// Copyright 2023 Zilliz
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

#include "milvus-storage/manifest.h"

#include <sstream>
#include <cstring>
#include <filesystem>

#include <arrow/status.h>
#include <arrow/result.h>
#include <avro/Compiler.hh>
#include <avro/DataFile.hh>
#include <avro/Decoder.hh>
#include <avro/Encoder.hh>
#include <avro/Specific.hh>
#include <avro/Stream.hh>
#include <fmt/format.h>

#include "milvus-storage/common/path_util.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/filesystem/fs.h"

// Specialize codec_traits for custom types in the avro namespace
namespace avro {

template <>
struct codec_traits<milvus_storage::api::ColumnGroupFile> {
  static void encode(Encoder& e, const milvus_storage::api::ColumnGroupFile& file) {
    avro::encode(e, file.path);
    avro::encode(e, file.start_index);
    avro::encode(e, file.end_index);
    avro::encode(e, file.metadata);
  }

  static void decode(Decoder& d, milvus_storage::api::ColumnGroupFile& file) {
    avro::decode(d, file.path);
    avro::decode(d, file.start_index);
    avro::decode(d, file.end_index);
    avro::decode(d, file.metadata);
  }
};

template <>
struct codec_traits<milvus_storage::api::ColumnGroup> {
  static void encode(Encoder& e, const milvus_storage::api::ColumnGroup& group) {
    avro::encode(e, group.columns);
    avro::encode(e, group.files);
    avro::encode(e, group.format);
  }

  static void decode(Decoder& d, milvus_storage::api::ColumnGroup& group) {
    avro::decode(d, group.columns);
    avro::decode(d, group.files);
    avro::decode(d, group.format);
  }
};

// Specialize codec_traits for std::shared_ptr<ColumnGroup> to enable automatic vector handling
template <>
struct codec_traits<std::shared_ptr<milvus_storage::api::ColumnGroup>> {
  static void encode(Encoder& e, const std::shared_ptr<milvus_storage::api::ColumnGroup>& group) {
    if (group) {
      avro::encode(e, *group);
    } else {
      // Encode empty fields for null group
      milvus_storage::api::ColumnGroup empty_group;
      avro::encode(e, empty_group);
    }
  }

  static void decode(Decoder& d, std::shared_ptr<milvus_storage::api::ColumnGroup>& group) {
    auto decoded_group = std::make_shared<milvus_storage::api::ColumnGroup>();
    avro::decode(d, *decoded_group);
    group = decoded_group;
  }
};

template <>
struct codec_traits<milvus_storage::api::DeltaLog> {
  static void encode(Encoder& e, const milvus_storage::api::DeltaLog& delta_log) {
    avro::encode(e, delta_log.path);
    auto type_int = static_cast<int32_t>(delta_log.type);
    avro::encode(e, type_int);
    avro::encode(e, delta_log.num_entries);
  }

  static void decode(Decoder& d, milvus_storage::api::DeltaLog& delta_log) {
    avro::decode(d, delta_log.path);
    int32_t type_int;
    avro::decode(d, type_int);
    delta_log.type = static_cast<milvus_storage::api::DeltaLogType>(type_int);
    avro::decode(d, delta_log.num_entries);
  }
};

template <>
struct codec_traits<milvus_storage::api::Index> {
  static void encode(Encoder& e, const milvus_storage::api::Index& idx) {
    avro::encode(e, idx.column_name);
    avro::encode(e, idx.index_type);
    avro::encode(e, idx.path);
    avro::encode(e, idx.properties);
  }

  static void decode(Decoder& d, milvus_storage::api::Index& idx) {
    avro::decode(d, idx.column_name);
    avro::decode(d, idx.index_type);
    avro::decode(d, idx.path);
    avro::decode(d, idx.properties);
  }
};

template <>
struct codec_traits<milvus_storage::api::Statistics> {
  static void encode(Encoder& e, const milvus_storage::api::Statistics& stat) {
    avro::encode(e, stat.paths);
    avro::encode(e, stat.metadata);
  }

  static void decode(Decoder& d, milvus_storage::api::Statistics& stat) {
    avro::decode(d, stat.paths);
    avro::decode(d, stat.metadata);
  }
};

template <>
struct codec_traits<milvus_storage::api::Manifest> {
  static void encode(Encoder& e, const milvus_storage::api::Manifest& m) {
    avro::encode(e, m.columnGroups());
    avro::encode(e, m.deltaLogs());
    avro::encode(e, m.stats());
    avro::encode(e, m.indexes());
  }

  static void decode(Decoder& d, milvus_storage::api::Manifest& m) {
    avro::decode(d, m.columnGroups());
    avro::decode(d, m.deltaLogs());
    avro::decode(d, m.stats());
    avro::decode(d, m.indexes());
  }
};

}  // namespace avro

namespace milvus_storage::api {

// Legacy manifest format magic number (used for detecting old format)
constexpr int32_t MANIFEST_MAGIC = 0x4D494C56;  // "MILV" in ASCII

// Avro schema describing the Manifest record structure.
// Field order must match codec_traits<Manifest> encode/decode order.
static const char* const MANIFEST_SCHEMA_JSON = R"({
  "type": "record",
  "name": "Manifest",
  "namespace": "milvus_storage",
  "fields": [
    {"name": "column_groups", "type": {"type": "array", "items": {
      "type": "record", "name": "ColumnGroup", "fields": [
        {"name": "columns", "type": {"type": "array", "items": "string"}},
        {"name": "files", "type": {"type": "array", "items": {
          "type": "record", "name": "ColumnGroupFile", "fields": [
            {"name": "path", "type": "string"},
            {"name": "start_index", "type": "long", "default": 0},
            {"name": "end_index", "type": "long", "default": 0},
            {"name": "metadata", "type": "bytes", "default": ""}
          ]
        }}},
        {"name": "format", "type": "string"}
      ]
    }}},
    {"name": "delta_logs", "type": {"type": "array", "items": {
      "type": "record", "name": "DeltaLog", "fields": [
        {"name": "path", "type": "string"},
        {"name": "type", "type": "int"},
        {"name": "num_entries", "type": "long"}
      ]
    }}},
    {"name": "stats", "type": {"type": "map", "values": {
      "type": "record", "name": "Statistics", "fields": [
        {"name": "paths", "type": {"type": "array", "items": "string"}},
        {"name": "metadata", "type": {"type": "map", "values": "string"}, "default": {}}
      ]
    }}, "default": {}},
    {"name": "indexes", "type": {"type": "array", "items": {
      "type": "record", "name": "Index", "fields": [
        {"name": "column_name", "type": "string"},
        {"name": "index_type", "type": "string"},
        {"name": "path", "type": "string"},
        {"name": "properties", "type": {"type": "map", "values": "string"}, "default": {}}
      ]
    }}, "default": []}
  ]
})";

static const avro::ValidSchema& getManifestSchema() {
  static const avro::ValidSchema schema = avro::compileJsonSchemaFromString(MANIFEST_SCHEMA_JSON);
  return schema;
}

static inline std::string ToRelative(const std::string& path,
                                     const std::optional<std::string>& base_path,
                                     const std::string& dir_path) {
  if (!base_path.has_value()) {
    return path;
  }

  std::filesystem::path full_dir_path(base_path.value());
  full_dir_path /= dir_path;
  std::string dir_path_str = full_dir_path.lexically_normal().string();

  if (path.size() >= dir_path_str.size() && path.substr(0, dir_path_str.size()) == dir_path_str) {
    return path.substr(dir_path_str.size());
  }

  // external table keep the absolute path
  return path;
}

static inline std::string ToAbsolute(const std::string& path,
                                     const std::optional<std::string>& base_path,
                                     const std::string& dir_path) {
  if (!base_path.has_value()) {
    return path;
  }

  auto uri_result = milvus_storage::StorageUri::Parse(path);
  if (uri_result.ok() && !uri_result.ValueOrDie().scheme.empty()) {
    return path;
  }

  std::filesystem::path p(path);
  if (p.is_relative()) {
    std::filesystem::path full_dir_path(base_path.value());
    full_dir_path /= dir_path;
    full_dir_path /= path;
    return full_dir_path.lexically_normal().string();
  }

  return path;
}

Manifest::Manifest(ColumnGroups column_groups,
                   const std::vector<DeltaLog>& delta_logs,
                   const std::map<std::string, Statistics>& stats,
                   const std::vector<Index>& indexes,
                   uint32_t version)
    : version_(version),
      column_groups_(std::move(column_groups)),
      delta_logs_(delta_logs),
      stats_(stats),
      indexes_(indexes) {}

Manifest::Manifest(const Manifest& other)
    : version_(other.version_),
      column_groups_(copy_column_groups(other.column_groups_)),
      delta_logs_(other.delta_logs_),
      stats_(other.stats_),
      indexes_(other.indexes_) {}

std::shared_ptr<Manifest> Manifest::DeepCopy() const { return std::shared_ptr<Manifest>(new Manifest(*this)); }

Manifest& Manifest::operator=(const Manifest& other) {
  if (this != &other) {
    version_ = other.version_;
    column_groups_ = copy_column_groups(other.column_groups_);
    delta_logs_ = other.delta_logs_;
    stats_ = other.stats_;
    indexes_ = other.indexes_;
  }
  return *this;
}

arrow::Status Manifest::serialize(std::ostream& output_stream) const {
  try {
    auto avro_output = avro::ostreamOutputStream(output_stream);
    avro::DataFileWriter<Manifest> writer(std::move(avro_output), getManifestSchema());
    writer.write(*this);
    writer.close();

    return arrow::Status::OK();
  } catch (const avro::Exception& e) {
    return arrow::Status::Invalid(fmt::format("Failed to serialize Manifest: {}", e.what()));  // NOLINT
  } catch (const std::exception& e) {
    return arrow::Status::Invalid(
        fmt::format("Failed to serialize Manifest (std::exception): {}", e.what()));  // NOLINT
  }
}

arrow::Status Manifest::deserialize(std::istream& input_stream) {
  auto error = [this](const std::string& msg) {
    column_groups_.clear();
    delta_logs_.clear();
    stats_.clear();
    indexes_.clear();
    return arrow::Status::Invalid(msg);
  };

  try {
    // Peek first 4 bytes to detect format
    char header[4] = {};
    input_stream.read(header, 4);
    if (!input_stream || input_stream.gcount() < 4) {
      return error("Cannot deserialize Manifest: stream is empty or too short");
    }
    input_stream.clear();
    input_stream.seekg(0);

    if (std::memcmp(header, "Obj\x01", 4) == 0) {
      // Standard Avro OCF format - schema resolution is automatic
      auto avro_input = avro::istreamInputStream(input_stream);
      avro::DataFileReader<Manifest> reader(std::move(avro_input), getManifestSchema());
      if (!reader.read(*this)) {
        return error("Failed to deserialize Manifest: no record in Avro file");
      }
      version_ = MANIFEST_VERSION;
    } else {
      deserializeLegacy(input_stream);
    }

    return arrow::Status::OK();
  } catch (const avro::Exception& e) {
    return error(fmt::format("Failed to deserialize Manifest: {}", e.what()));  // NOLINT
  } catch (const std::exception& e) {
    return error(fmt::format("Failed to deserialize Manifest: {}", e.what()));  // NOLINT
  } catch (...) {
    return error("Failed to deserialize Manifest: unknown error (possibly invalid or empty stream)");
  }
}

void Manifest::deserializeLegacy(std::istream& input_stream) {
  auto avro_input = avro::istreamInputStream(input_stream);
  auto decoder = avro::binaryDecoder();
  decoder->init(*avro_input);

  int32_t magic = 0;
  avro::decode(*decoder, magic);
  if (magic != MANIFEST_MAGIC) {
    throw avro::Exception("Invalid MILV magic number");
  }

  int32_t version = 0;
  avro::decode(*decoder, version);
  version_ = version;

  avro::decode(*decoder, column_groups_);
  avro::decode(*decoder, delta_logs_);

  if (version >= 3) {
    avro::decode(*decoder, stats_);
  } else {
    std::map<std::string, std::vector<std::string>> legacy_stats;
    avro::decode(*decoder, legacy_stats);
    stats_.clear();
    for (auto& [key, files] : legacy_stats) {
      Statistics stat;
      stat.paths = std::move(files);
      stats_[key] = std::move(stat);
    }
  }

  if (version >= 2) {
    avro::decode(*decoder, indexes_);
  } else {
    indexes_.clear();
  }
}

std::shared_ptr<ColumnGroup> Manifest::getColumnGroup(const std::string& column_name) const {
  for (const auto& cg : column_groups_) {
    for (const auto& col : cg->columns) {
      if (col == column_name) {
        return cg;
      }
    }
  }
  return nullptr;
}

const Index* Manifest::getIndex(const std::string& column_name, const std::string& index_type) const {
  for (const auto& idx : indexes_) {
    if (idx.column_name == column_name && idx.index_type == index_type) {
      return &idx;
    }
  }
  return nullptr;
}

Manifest Manifest::toRelativePaths(const std::string& base_path) const {
  Manifest copy_manifest(*this);

  for (auto& column_group : copy_manifest.column_groups_) {
    for (auto& file : column_group->files) {
      file.path = ToRelative(file.path, std::optional<std::string>(base_path), milvus_storage::kDataPath);
    }
  }

  // normalize delta log paths (convert absolute to relative)
  for (auto& delta_log : copy_manifest.delta_logs_) {
    delta_log.path = ToRelative(delta_log.path, std::optional<std::string>(base_path), milvus_storage::kDeltaPath);
  }

  // normalize stats paths (convert absolute to relative)
  for (auto& [key, stat] : copy_manifest.stats_) {
    for (auto& path : stat.paths) {
      path = ToRelative(path, std::optional<std::string>(base_path), milvus_storage::kStatsPath);
    }
  }

  // normalize index paths (convert absolute to relative)
  for (auto& idx : copy_manifest.indexes_) {
    idx.path = ToRelative(idx.path, std::optional<std::string>(base_path), milvus_storage::kIndexPath);
  }

  return copy_manifest;
}

void Manifest::ToAbsolutePaths(const std::string& base_path) {
  for (auto& column_group : column_groups_) {
    for (auto& file : column_group->files) {
      file.path = ToAbsolute(file.path, std::optional<std::string>(base_path), milvus_storage::kDataPath);
    }
  }

  // denormalize delta log paths (convert relative to absolute)
  for (auto& delta_log : delta_logs_) {
    delta_log.path = ToAbsolute(delta_log.path, std::optional<std::string>(base_path), milvus_storage::kDeltaPath);
  }

  // denormalize stats paths (convert relative to absolute)
  for (auto& [key, stat] : stats_) {
    for (auto& path : stat.paths) {
      path = ToAbsolute(path, std::optional<std::string>(base_path), milvus_storage::kStatsPath);
    }
  }

  // denormalize index paths (convert relative to absolute)
  for (auto& idx : indexes_) {
    idx.path = ToAbsolute(idx.path, std::optional<std::string>(base_path), milvus_storage::kIndexPath);
  }
}

// Static cache accessor
LRUCache<std::string, std::shared_ptr<Manifest>>& Manifest::getCache() {
  static milvus_storage::LRUCache<std::string, std::shared_ptr<Manifest>> cache(1024);
  return cache;
}

void Manifest::CleanCache() { getCache().clean(); }

// Build a globally unique cache key from the filesystem's root path and the file path.
// All our filesystems are SubTreeFileSystem (via FileSystemProxy), so base_path() gives
// the root (local absolute path, or S3 bucket name, etc.).
static std::string MakeCacheKey(const milvus_storage::ArrowFileSystemPtr& fs, const std::string& path) {
  auto* subtree = dynamic_cast<arrow::fs::SubTreeFileSystem*>(fs.get());
  if (subtree) {
    return subtree->base_path() + "/" + path;
  }
  // Fallback for non-SubTreeFileSystem (shouldn't happen in practice)
  return fs->type_name() + "://" + path;
}

arrow::Result<std::shared_ptr<Manifest>> Manifest::ReadFrom(const milvus_storage::ArrowFileSystemPtr& fs,
                                                            const std::string& path) {
  std::string cache_key = MakeCacheKey(fs, path);

  auto& cache = getCache();
  auto cached = cache.get(cache_key);
  if (cached.has_value()) {
    return std::move(cached.value());
  }

  // Read file
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(path));
  ARROW_ASSIGN_OR_RAISE(auto file_size, input_file->GetSize());
  ARROW_ASSIGN_OR_RAISE(auto buffer, input_file->Read(file_size));

  if (buffer->size() != file_size) {
    return arrow::Status::IOError(
        fmt::format("Failed to read the complete file, expected size={}, actual size={}", file_size, buffer->size()));
  }
  ARROW_RETURN_NOT_OK(input_file->Close());

  // Deserialize
  auto manifest = std::make_shared<Manifest>();
  std::string data(reinterpret_cast<const char*>(buffer->data()), buffer->size());
  std::istringstream in(data);
  ARROW_RETURN_NOT_OK(manifest->deserialize(in));

  // Derive base_path from manifest file path by stripping _metadata/manifest-*.avro
  // Path format: {base_path}/_metadata/manifest-{version}.avro
  std::filesystem::path p(path);
  std::string base_path = p.parent_path().parent_path().string();
  manifest->ToAbsolutePaths(base_path);

  cache.put(cache_key, manifest);
  return manifest;
}

arrow::Status Manifest::WriteTo(const milvus_storage::ArrowFileSystemPtr& fs,
                                const std::string& path,
                                const Manifest& manifest) {
  // Derive base_path from path
  std::filesystem::path p(path);
  std::string base_path = p.parent_path().parent_path().string();

  // Convert to relative paths and serialize
  Manifest relative_manifest = manifest.toRelativePaths(base_path);
  std::ostringstream oss;
  ARROW_RETURN_NOT_OK(relative_manifest.serialize(oss));
  std::string data = oss.str();

  // Try conditional write first if filesystem supports it
  auto conditional_fs = std::dynamic_pointer_cast<milvus_storage::UploadConditional>(fs);
  if (conditional_fs) {
    auto res = conditional_fs->OpenConditionalOutputStream(path, nullptr);
    if (res.ok()) {
      auto output_stream = res.ValueOrDie();
      ARROW_RETURN_NOT_OK(output_stream->Write(data.data(), data.size()));
      auto result = output_stream->Close();
      if (!result.ok()) {
        if (result.code() == arrow::StatusCode::IOError) {
          return arrow::Status::AlreadyExists("File already exists: ", path);
        }
        return result;
      }
      return arrow::Status::OK();
    }
  }

  // Fall back: check existence then write
  ARROW_ASSIGN_OR_RAISE(auto file_info, fs->GetFileInfo(path));
  if (file_info.type() != arrow::fs::FileType::NotFound) {
    return arrow::Status::AlreadyExists("File already exists: ", path);
  }

  auto [parent, _] = milvus_storage::GetAbstractPathParent(path);
  if (!parent.empty()) {
    ARROW_RETURN_NOT_OK(fs->CreateDir(parent));
  }

  ARROW_ASSIGN_OR_RAISE(auto output_stream, fs->OpenOutputStream(path));
  ARROW_RETURN_NOT_OK(output_stream->Write(data.data(), data.size()));
  ARROW_RETURN_NOT_OK(output_stream->Close());

  return arrow::Status::OK();
}

}  // namespace milvus_storage::api
