// Copyright 2025 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Generic CLI tool for exploring external data sources and reading via
// the manifest-based loon storage system.
//
// Usage:
//   loon demo-table --type <type> --path <dir> [--rows N] [--deletes pos1,...]
//   loon create     --format <format> --source <uri> --target <base_path>
//                   --columns col1,col2,...  [--prop key=value ...]
//   loon describe   <manifest_path> [--prop key=value ...]
//   loon read       <manifest_path> --columns col1,col2,...
//                   [--take pos1,pos2,...] [--prop key=value ...]

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <arrow/api.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/transaction/transaction.h"
#include "milvus-storage/format/lance/lance_common.h"
#include <folly/dynamic.h>
#include <folly/json.h>
#include "milvus-storage/format/iceberg/iceberg_common.h"
#include "iceberg_bridge.h"
#include "lance_bridge.h"

using milvus_storage::api::ColumnGroup;
using milvus_storage::api::ColumnGroupFile;
using milvus_storage::api::ColumnGroups;
using milvus_storage::api::GetValue;
using milvus_storage::api::kPropertyMetadata;
using milvus_storage::api::Manifest;
using milvus_storage::api::Properties;
using milvus_storage::api::SetValue;
using milvus_storage::api::transaction::Transaction;
using milvus_storage::FilesystemCache;
using milvus_storage::FormatReader;
using milvus_storage::StorageUri;

// ─── helpers ─────────────────────────────────────────────────────────

/// Resolve a local path to absolute (no-op for URIs with scheme).
static std::string ResolvePath(const std::string& path) {
  // Skip URI-style paths (s3://..., gs://..., etc.)
  if (path.find("://") != std::string::npos) return path;
  return std::filesystem::absolute(path).string();
}

static std::vector<std::string> Split(const std::string& s, char delim) {
  std::vector<std::string> parts;
  std::istringstream stream(s);
  std::string item;
  while (std::getline(stream, item, delim)) {
    if (!item.empty()) {
      parts.push_back(item);
    }
  }
  return parts;
}

static std::vector<int64_t> ParseInt64List(const std::string& s) {
  std::vector<int64_t> result;
  for (const auto& part : Split(s, ',')) {
    result.push_back(std::stoll(part));
  }
  return result;
}

static void PrintBatch(const std::shared_ptr<arrow::RecordBatch>& batch) {
  std::cout << batch->schema()->ToString() << std::endl;
  std::cout << "  rows: " << batch->num_rows() << std::endl;
  for (int c = 0; c < batch->num_columns(); ++c) {
    std::cout << "  " << batch->schema()->field(c)->name() << ": "
              << batch->column(c)->ToString() << std::endl;
  }
}

static void PrintTable(const std::shared_ptr<arrow::Table>& table) {
  std::cout << table->schema()->ToString() << std::endl;
  std::cout << "  rows: " << table->num_rows() << std::endl;
  for (int c = 0; c < table->num_columns(); ++c) {
    std::cout << "  " << table->schema()->field(c)->name() << ": "
              << table->column(c)->ToString() << std::endl;
  }
}

/// Parse --prop key=value pairs into properties
static void ApplyProps(Properties& properties, const std::vector<std::string>& props) {
  for (const auto& kv : props) {
    auto eq = kv.find('=');
    if (eq == std::string::npos) {
      std::cerr << "Warning: ignoring malformed --prop '" << kv
                << "' (expected key=value)" << std::endl;
      continue;
    }
    SetValue(properties, kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str());
  }
}

// ─── demo-table ──────────────────────────────────────────────────────

static int DoDemoTable(int argc, char** argv) {
  std::string type;
  std::string path;
  uint64_t rows = 100;
  std::vector<int64_t> deletes;
  std::vector<std::string> extra_props;

  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--type" && i + 1 < argc) {
      type = argv[++i];
    } else if (arg == "--path" && i + 1 < argc) {
      path = argv[++i];
    } else if (arg == "--rows" && i + 1 < argc) {
      rows = std::stoull(argv[++i]);
    } else if (arg == "--deletes" && i + 1 < argc) {
      deletes = ParseInt64List(argv[++i]);
    } else if (arg == "--prop" && i + 1 < argc) {
      extra_props.emplace_back(argv[++i]);
    }
  }

  if (type.empty() || path.empty()) {
    std::cerr << "Usage: loon demo-table --type <type> --path <dir>"
              << " [--rows N] [--deletes pos1,pos2,...] [--prop key=value ...]"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "Types: iceberg" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Creates a demo table with schema (id int64, name string,"
              << " value float64)." << std::endl;
    std::cerr << R"(Data: id=0..N-1, name="row_0".."row_{N-1}", value=id*1.5)"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "For cloud storage, pass extfs.* properties via --prop."
              << std::endl;
    return 1;
  }

  if (type != "iceberg") {
    std::cerr << "Error: unsupported type '" << type
              << "'. Supported: iceberg" << std::endl;
    return 1;
  }

  try {
    // Build storage options from properties (if any)
    std::unordered_map<std::string, std::string> storage_options;
    std::string table_path = path;

    if (!extra_props.empty()) {
      Properties properties;
      SetValue(properties, PROPERTY_FS_ROOT_PATH, "/");
      ApplyProps(properties, extra_props);
      FilesystemCache::getInstance().clean();

      auto config_result = FilesystemCache::resolve_config(properties, path);
      if (config_result.ok()) {
        storage_options = milvus_storage::iceberg::ToStorageOptions(config_result.ValueOrDie());
        // Convert Milvus URI to standard format for iceberg-rust
        auto parsed = StorageUri::Parse(path);
        if (parsed.ok() && !parsed->scheme.empty()) {
          auto standard = StorageUri::Make(parsed.ValueOrDie(), false);
          if (standard.ok()) {
            table_path = standard.ValueOrDie();
          }
        }
      }
    } else {
      table_path = ResolvePath(path);
    }

    bool with_deletes = !deletes.empty();
    auto info = milvus_storage::iceberg::CreateTestTable(
        table_path, rows, with_deletes, deletes, storage_options);

    std::cout << "Created iceberg table:" << std::endl;
    std::cout << "  path:              " << table_path << std::endl;
    std::cout << "  metadata_location: " << info.metadata_location << std::endl;
    std::cout << "  snapshot_id:       " << info.snapshot_id << std::endl;
    std::cout << "  data_file:         " << info.data_file_uri << std::endl;
    std::cout << "  rows:              " << rows << std::endl;
    if (with_deletes) {
      std::cout << "  deletes:           [";
      for (size_t i = 0; i < deletes.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << deletes[i];
      }
      std::cout << "]" << std::endl;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

// ─── explore helpers (from exttable_c.cpp pattern) ───────────────────

static arrow::Result<std::vector<ColumnGroupFile>> ExploreParquetOrVortex(
    const std::string& source,
    const Properties& properties) {
  // Resolve explore_dir: if URI, extract key
  std::string resolved_dir = source;
  StorageUri explore_uri;
  auto uri_res = StorageUri::Parse(source);
  if (uri_res.ok() && !uri_res->scheme.empty()) {
    explore_uri = *uri_res;
    resolved_dir = explore_uri.key;
  }

  ARROW_ASSIGN_OR_RAISE(auto fs,
      FilesystemCache::getInstance().get(properties, source));

  arrow::fs::FileSelector selector;
  selector.base_dir = resolved_dir;
  selector.allow_not_found = false;
  selector.recursive = false;
  selector.max_recursion = 0;

  ARROW_ASSIGN_OR_RAISE(auto file_infos, fs->GetFileInfo(selector));

  bool is_local = milvus_storage::IsLocalFileSystem(fs);

  StorageUri uri_base;
  if (!explore_uri.scheme.empty()) {
    uri_base.scheme = explore_uri.scheme;
    uri_base.address = explore_uri.address;
    uri_base.bucket_name = explore_uri.bucket_name;
  } else if (!is_local) {
    uri_base.scheme = fs->type_name();
    ARROW_ASSIGN_OR_RAISE(uri_base.address,
        GetValue<std::string>(properties, PROPERTY_FS_ADDRESS));
    ARROW_ASSIGN_OR_RAISE(uri_base.bucket_name,
        GetValue<std::string>(properties, PROPERTY_FS_BUCKET_NAME));
  }

  std::vector<ColumnGroupFile> files;
  for (const auto& fi : file_infos) {
    if (fi.type() != arrow::fs::FileType::File) continue;
    if (is_local && explore_uri.scheme.empty()) {
      // For local files, use the absolute path directly.
      // SubTreeFileSystem at "/" strips the leading /, so prepend it.
      std::string abs_path = "/" + fi.path();
      files.emplace_back(ColumnGroupFile{
          std::move(abs_path), -1, -1, {}});
    } else {
      uri_base.key = fi.path();
      ARROW_ASSIGN_OR_RAISE(auto file_uri, StorageUri::Make(uri_base));
      files.emplace_back(ColumnGroupFile{
          std::move(file_uri), -1, -1, {}});
    }
  }
  return files;
}

static arrow::Result<std::vector<ColumnGroupFile>> ExploreLance(
    const std::string& source,
    const Properties& properties) {
  ARROW_ASSIGN_OR_RAISE(auto fs_config,
      FilesystemCache::resolve_config(properties, source));

  std::string resolved_dir = source;
  auto uri_res = StorageUri::Parse(source);
  if (uri_res.ok() && !uri_res->scheme.empty()) {
    resolved_dir = uri_res->key;
  }

  ARROW_ASSIGN_OR_RAISE(auto lance_base_uri,
      milvus_storage::lance::BuildLanceBaseUri(fs_config, resolved_dir));
  auto storage_options = milvus_storage::lance::ToStorageOptions(fs_config);

  auto dataset = milvus_storage::lance::BlockingDataset::Open(
      lance_base_uri, storage_options);
  auto fragment_ids = dataset->GetAllFragmentIds();

  std::vector<ColumnGroupFile> files;
  for (auto frag_id : fragment_ids) {
    auto row_count = dataset->GetFragmentRowCount(frag_id);
    files.emplace_back(ColumnGroupFile{
        milvus_storage::lance::MakeLanceUri(lance_base_uri, frag_id),
        0,
        static_cast<int64_t>(row_count),
        {}});
  }
  return files;
}

static arrow::Result<std::vector<ColumnGroupFile>> ExploreIceberg(
    const std::string& source,
    const Properties& properties) {
  ARROW_ASSIGN_OR_RAISE(auto fs_config,
      FilesystemCache::resolve_config(properties, source));
  auto storage_options = milvus_storage::iceberg::ToStorageOptions(fs_config);

  ARROW_ASSIGN_OR_RAISE(auto snapshot_str,
      GetValue<std::string>(properties, PROPERTY_ICEBERG_SNAPSHOT_ID));
  int64_t snapshot_id = std::stoll(snapshot_str);

  // Convert Milvus URI to standard format for iceberg-rust
  ARROW_ASSIGN_OR_RAISE(auto parsed_uri, StorageUri::Parse(source));
  ARROW_ASSIGN_OR_RAISE(auto iceberg_uri, StorageUri::Make(parsed_uri, false));

  auto file_infos = milvus_storage::iceberg::PlanFiles(
      iceberg_uri, snapshot_id, storage_options);

  std::vector<ColumnGroupFile> files;
  files.reserve(file_infos.size());
  for (const auto& info : file_infos) {
    auto milvus_path = milvus_storage::iceberg::ToMilvusUri(info.data_file_path, fs_config.address);

    std::unordered_map<std::string, std::string> file_props;
    if (!info.delete_metadata_json.empty()) {
      file_props[kPropertyMetadata] =
          milvus_storage::iceberg::ConvertDeleteMetadataPaths(info.delete_metadata_json, fs_config.address);
    }
    files.emplace_back(ColumnGroupFile{
        std::move(milvus_path),
        0,
        static_cast<int64_t>(info.record_count - info.num_deleted_rows),
        std::move(file_props)});
  }
  return files;
}

// ─── create ──────────────────────────────────────────────────────────

static int DoCreate(int argc, char** argv) {
  std::string format;
  std::string source;
  std::string target;
  std::vector<std::string> columns;
  std::vector<std::string> extra_props;

  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--format" && i + 1 < argc) {
      format = argv[++i];
    } else if (arg == "--source" && i + 1 < argc) {
      source = argv[++i];
    } else if (arg == "--target" && i + 1 < argc) {
      target = argv[++i];
    } else if (arg == "--columns" && i + 1 < argc) {
      columns = Split(argv[++i], ',');
    } else if (arg == "--prop" && i + 1 < argc) {
      extra_props.emplace_back(argv[++i]);
    }
  }

  if (format.empty() || source.empty() || target.empty() || columns.empty()) {
    std::cerr << "Usage: loon create --format <format> --source <uri> "
              << "--target <base_path> --columns col1,col2,... "
              << "[--prop key=value ...]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Formats: parquet, vortex, lance-table, iceberg-table"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "For iceberg-table, --prop iceberg.snapshot_id=N is required."
              << std::endl;
    std::cerr << "The --source is the metadata.json path for iceberg-table,"
              << std::endl;
    std::cerr << "or the data directory for parquet/vortex/lance-table."
              << std::endl;
    return 1;
  }

  try {
    source = ResolvePath(source);
    target = ResolvePath(target);

    // Build properties — default to local filesystem with root "/"
    Properties properties;
    SetValue(properties, PROPERTY_FS_ROOT_PATH, "/");
    ApplyProps(properties, extra_props);
    FilesystemCache::getInstance().clean();

    // 1. Explore source → get ColumnGroupFiles
    std::vector<ColumnGroupFile> files;
    if (format == LOON_FORMAT_LANCE_TABLE) {
      auto res = ExploreLance(source, properties);
      if (!res.ok()) {
        std::cerr << "Error exploring lance: " << res.status().ToString()
                  << std::endl;
        return 1;
      }
      files = std::move(*res);
    } else if (format == LOON_FORMAT_ICEBERG_TABLE) {
      auto res = ExploreIceberg(source, properties);
      if (!res.ok()) {
        std::cerr << "Error exploring iceberg: " << res.status().ToString()
                  << std::endl;
        return 1;
      }
      files = std::move(*res);
    } else {
      // parquet or vortex
      auto res = ExploreParquetOrVortex(source, properties);
      if (!res.ok()) {
        std::cerr << "Error exploring " << format << ": "
                  << res.status().ToString() << std::endl;
        return 1;
      }
      files = std::move(*res);
    }

    std::cout << "Explored " << files.size() << " file(s) from " << source
              << std::endl;
    for (size_t i = 0; i < files.size(); ++i) {
      std::cout << "  [" << i << "] " << files[i].path;
      if (files[i].end_index > 0) {
        std::cout << "  (rows: " << files[i].end_index << ")";
      }
      if (files[i].properties.count(kPropertyMetadata) > 0) {
        std::cout << "  (has metadata)";
      }
      std::cout << std::endl;
    }

    // 2. Get filesystem for target
    auto fs_res = FilesystemCache::getInstance().get(properties);
    if (!fs_res.ok()) {
      std::cerr << "Error getting filesystem: " << fs_res.status().ToString()
                << std::endl;
      return 1;
    }
    auto fs = *fs_res;

    // 3. Build ColumnGroup and commit manifest via Transaction
    ColumnGroups cgs;
    cgs.push_back(std::make_shared<ColumnGroup>(
        ColumnGroup{.columns = columns, .format = format, .files = files}));

    auto tx_res = Transaction::Open(fs, target);
    if (!tx_res.ok()) {
      std::cerr << "Error opening transaction: " << tx_res.status().ToString()
                << std::endl;
      return 1;
    }
    auto tx = std::move(*tx_res);
    tx->AppendFiles(cgs);

    auto commit_res = tx->Commit();
    if (!commit_res.ok()) {
      std::cerr << "Error committing: " << commit_res.status().ToString()
                << std::endl;
      return 1;
    }
    auto version = *commit_res;

    auto manifest_path = milvus_storage::get_manifest_filepath(target, version);
    std::cout << "Committed manifest: " << manifest_path << std::endl;
    std::cout << "  version: " << version << std::endl;
    std::cout << "  files: " << files.size() << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

// ─── describe ─────────────────────────────────────────────────────────

/// Read and deserialize a manifest file, returning the Manifest object.
static arrow::Result<std::shared_ptr<Manifest>> ReadManifest(
    const std::string& manifest_path, const Properties& properties) {
  ARROW_ASSIGN_OR_RAISE(auto fs,
      FilesystemCache::getInstance().get(properties, manifest_path));
  return Manifest::ReadFrom(fs, manifest_path);
}

static int DoDescribe(int argc, char** argv) {
  if (argc < 1) {
    std::cerr << "Usage: loon describe <manifest_path> [--prop key=value ...]"
              << std::endl;
    return 1;
  }
  std::string manifest_path = argv[0];
  std::vector<std::string> extra_props;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--prop" && i + 1 < argc) {
      extra_props.emplace_back(argv[++i]);
    }
  }

  try {
    manifest_path = ResolvePath(manifest_path);

    Properties properties;
    SetValue(properties, PROPERTY_FS_ROOT_PATH, "/");
    ApplyProps(properties, extra_props);
    FilesystemCache::getInstance().clean();

    auto manifest_res = ReadManifest(manifest_path, properties);
    if (!manifest_res.ok()) {
      std::cerr << "Error: " << manifest_res.status().ToString() << std::endl;
      return 1;
    }
    auto manifest = *manifest_res;

    // Build folly::dynamic tree
    folly::dynamic root = folly::dynamic::object
        ("path", manifest_path)
        ("version", manifest->version());

    // Column groups
    folly::dynamic cg_arr = folly::dynamic::array;
    for (auto& cg : manifest->columnGroups()) {
      folly::dynamic cols = folly::dynamic::array;
      for (auto& c : cg->columns) cols.push_back(c);

      folly::dynamic file_arr = folly::dynamic::array;
      for (auto& f : cg->files) {
        folly::dynamic fobj = folly::dynamic::object
            ("path", f.path)
            ("start_index", f.start_index)
            ("end_index", f.end_index);
        auto meta_it = f.properties.find(kPropertyMetadata);
        if (meta_it != f.properties.end()) {
          fobj["metadata"] = meta_it->second;
        } else {
          fobj["metadata"] = nullptr;
        }
        file_arr.push_back(std::move(fobj));
      }

      cg_arr.push_back(folly::dynamic::object
          ("format", cg->format)
          ("columns", std::move(cols))
          ("files", std::move(file_arr)));
    }
    root["column_groups"] = std::move(cg_arr);

    // Delta logs
    folly::dynamic dl_arr = folly::dynamic::array;
    for (auto& dl : manifest->deltaLogs()) {
      dl_arr.push_back(folly::dynamic::object
          ("path", dl.path)
          ("type", static_cast<int>(dl.type))
          ("num_entries", dl.num_entries));
    }
    root["delta_logs"] = std::move(dl_arr);

    // Stats
    folly::dynamic stats_obj = folly::dynamic::object;
    for (auto& [key, stat] : manifest->stats()) {
      folly::dynamic paths = folly::dynamic::array;
      for (auto& p : stat.paths) paths.push_back(p);
      folly::dynamic meta = folly::dynamic::object;
      for (auto& [mk, mv] : stat.metadata) meta[mk] = mv;
      stats_obj[key] = folly::dynamic::object
          ("paths", std::move(paths))
          ("metadata", std::move(meta));
    }
    root["stats"] = std::move(stats_obj);

    // Indexes
    folly::dynamic idx_arr = folly::dynamic::array;
    for (auto& idx : manifest->indexes()) {
      folly::dynamic props = folly::dynamic::object;
      for (auto& [pk, pv] : idx.properties) props[pk] = pv;
      idx_arr.push_back(folly::dynamic::object
          ("column_name", idx.column_name)
          ("index_type", idx.index_type)
          ("path", idx.path)
          ("properties", std::move(props)));
    }
    root["indexes"] = std::move(idx_arr);

    std::cout << folly::toPrettyJson(root) << std::endl;

    FilesystemCache::getInstance().clean();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

// ─── read ────────────────────────────────────────────────────────────

static int DoRead(int argc, char** argv) {
  if (argc < 1) {
    std::cerr << "Usage: loon read <manifest_path> --columns col1,col2,..."
              << " [--take pos1,pos2,...] [--prop key=value ...]" << std::endl;
    return 1;
  }
  std::string manifest_path = argv[0];
  std::vector<std::string> columns;
  std::vector<int64_t> take_positions;
  std::vector<std::string> extra_props;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--columns" && i + 1 < argc) {
      columns = Split(argv[++i], ',');
    } else if (arg == "--take" && i + 1 < argc) {
      take_positions = ParseInt64List(argv[++i]);
    } else if (arg == "--prop" && i + 1 < argc) {
      extra_props.emplace_back(argv[++i]);
    }
  }

  if (columns.empty()) {
    std::cerr << "Error: --columns is required" << std::endl;
    return 1;
  }

  try {
    manifest_path = ResolvePath(manifest_path);

    // Build properties — default to local filesystem with root "/"
    Properties properties;
    SetValue(properties, PROPERTY_FS_ROOT_PATH, "/");
    ApplyProps(properties, extra_props);
    FilesystemCache::getInstance().clean();

    // 1. Read manifest
    auto manifest_res = ReadManifest(manifest_path, properties);
    if (!manifest_res.ok()) {
      std::cerr << "Error: " << manifest_res.status().ToString() << std::endl;
      return 1;
    }
    auto manifest = *manifest_res;

    // 2. Print manifest summary
    auto& cgs = manifest->columnGroups();
    std::cout << "Manifest: " << manifest_path << std::endl;
    std::cout << "  version: " << manifest->version() << std::endl;
    std::cout << "  column_groups: " << cgs.size() << std::endl;
    for (size_t gi = 0; gi < cgs.size(); ++gi) {
      auto& cg = cgs[gi];
      std::cout << "  [" << gi << "] format=" << cg->format
                << "  columns=[";
      for (size_t ci = 0; ci < cg->columns.size(); ++ci) {
        if (ci > 0) std::cout << ",";
        std::cout << cg->columns[ci];
      }
      std::cout << "]  files=" << cg->files.size() << std::endl;
      for (size_t fi = 0; fi < cg->files.size(); ++fi) {
        auto& f = cg->files[fi];
        std::cout << "    file[" << fi << "] path=" << f.path
                  << "  range=[" << f.start_index << "," << f.end_index << ")"
                  << "  has_metadata=" << (f.properties.count(kPropertyMetadata) > 0 ? "true" : "false") << std::endl;
        auto meta_it = f.properties.find(kPropertyMetadata);
        if (meta_it != f.properties.end()) {
          std::cout << "    metadata: " << meta_it->second << std::endl;
        }
      }
    }

    // 3. Read data via FormatReader per file in each column group
    int64_t grand_total = 0;
    for (size_t gi = 0; gi < cgs.size(); ++gi) {
      auto& cg = cgs[gi];
      for (size_t fi = 0; fi < cg->files.size(); ++fi) {
        auto& f = cg->files[fi];
        auto reader_res = milvus_storage::FormatReader::create(
            nullptr, cg->format, f, properties, columns, nullptr);
        if (!reader_res.ok()) {
          std::cerr << "Error creating reader for file[" << fi << "]: "
                    << reader_res.status().ToString() << std::endl;
          return 1;
        }
        auto reader = *reader_res;

        if (!take_positions.empty()) {
          // Random access
          std::cout << "--- take [";
          for (size_t ti = 0; ti < take_positions.size(); ++ti) {
            if (ti > 0) std::cout << ",";
            std::cout << take_positions[ti];
          }
          std::cout << "] ---" << std::endl;

          auto table_res = reader->take(take_positions);
          if (!table_res.ok()) {
            std::cerr << "Error in take(): "
                      << table_res.status().ToString() << std::endl;
            return 1;
          }
          PrintTable(*table_res);
          grand_total += (*table_res)->num_rows();
        } else {
          // Sequential read via get_chunk
          auto rg_res = reader->get_row_group_infos();
          if (!rg_res.ok()) {
            std::cerr << "Error getting row groups: "
                      << rg_res.status().ToString() << std::endl;
            return 1;
          }
          auto rg_infos = *rg_res;
          int64_t file_total = 0;
          for (size_t rg = 0; rg < rg_infos.size(); ++rg) {
            auto batch_res = reader->get_chunk(rg);
            if (!batch_res.ok()) {
              std::cerr << "Error reading chunk " << rg << ": "
                        << batch_res.status().ToString() << std::endl;
              return 1;
            }
            auto batch = *batch_res;
            std::cout << "--- cg[" << gi << "] file[" << fi
                      << "] rg[" << rg << "] ---" << std::endl;
            PrintBatch(batch);
            file_total += batch->num_rows();
          }
          grand_total += file_total;
        }
      }
    }
    std::cout << "total_rows: " << grand_total << std::endl;

    FilesystemCache::getInstance().clean();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

// ─── main ────────────────────────────────────────────────────────────

static void PrintUsage() {
  std::cerr
      << "Usage: loon <command> [args...]" << std::endl
      << std::endl
      << "Commands:" << std::endl
      << "  demo-table  Create a demo table for testing."
      << std::endl
      << "  create      Explore an external data source and commit a manifest."
      << std::endl
      << "  describe    Dump a manifest file as formatted JSON."
      << std::endl
      << "  read        Read data from a manifest (sequential or random access)."
      << std::endl
      << std::endl
      << "demo-table:" << std::endl
      << "  loon demo-table --type <type> --path <dir> \\" << std::endl
      << "                  [--rows N] [--deletes pos1,pos2,...]" << std::endl
      << std::endl
      << "  Types: iceberg" << std::endl
      << std::endl
      << "create:" << std::endl
      << "  loon create --format <format> --source <uri> \\" << std::endl
      << "                   --target <base_path> --columns col1,col2,... \\"
      << std::endl
      << "                   [--prop key=value ...]" << std::endl
      << std::endl
      << "  Formats: parquet, vortex, lance-table, iceberg-table" << std::endl
      << std::endl
      << "  For iceberg-table:" << std::endl
      << "    --source is the metadata.json path" << std::endl
      << "    --prop iceberg.snapshot_id=N is required" << std::endl
      << std::endl
      << "  For parquet/vortex:" << std::endl
      << "    --source is the directory containing data files" << std::endl
      << std::endl
      << "  For lance-table:" << std::endl
      << "    --source is the lance dataset directory" << std::endl
      << std::endl
      << "describe:" << std::endl
      << "  loon describe <manifest_path> [--prop key=value ...]"
      << std::endl
      << std::endl
      << "read:" << std::endl
      << "  loon read <manifest_path> --columns col1,col2,... \\"
      << std::endl
      << "                 [--take pos1,pos2,...] [--prop key=value ...]"
      << std::endl
      << std::endl
      << "Examples:" << std::endl
      << "  # Explore an Iceberg table and create a manifest" << std::endl
      << "  loon create --format iceberg-table \\" << std::endl
      << "    --source /data/iceberg/metadata/v1.metadata.json \\"
      << std::endl
      << "    --target /tmp/my_manifest \\" << std::endl
      << "    --columns id,name,value \\" << std::endl
      << "    --prop iceberg.snapshot_id=1" << std::endl
      << std::endl
      << "  # Explore a parquet directory and create a manifest" << std::endl
      << "  loon create --format parquet \\" << std::endl
      << "    --source /data/parquets/ \\" << std::endl
      << "    --target /tmp/my_manifest \\" << std::endl
      << "    --columns id,name,value" << std::endl
      << std::endl
      << "  # Describe a manifest (dump as JSON)" << std::endl
      << "  loon describe /tmp/my_manifest/_metadata/manifest-1.avro"
      << std::endl
      << std::endl
      << "  # Read all data from the manifest" << std::endl
      << "  loon read /tmp/my_manifest/_metadata/manifest-1.avro \\"
      << std::endl
      << "    --columns id,name,value" << std::endl
      << std::endl
      << "  # Random access (take)" << std::endl
      << "  loon read /tmp/my_manifest/_metadata/manifest-1.avro \\"
      << std::endl
      << "    --columns id,name --take 0,5,10,15" << std::endl;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage();
    return 1;
  }

  std::string command = argv[1];
  if (command == "demo-table") {
    return DoDemoTable(argc - 2, argv + 2);
  } else if (command == "create") {
    return DoCreate(argc - 2, argv + 2);
  } else if (command == "describe") {
    return DoDescribe(argc - 2, argv + 2);
  } else if (command == "read") {
    return DoRead(argc - 2, argv + 2);
  } else if (command == "--help" || command == "-h") {
    PrintUsage();
    return 0;
  } else {
    std::cerr << "Unknown command: " << command << std::endl;
    PrintUsage();
    return 1;
  }
}
