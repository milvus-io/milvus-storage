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
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/upload_conditional.h"

// Specialize codec_traits for custom types in the avro namespace
namespace avro {

// Avro natively supports std::map but not std::unordered_map.
// This codec_traits bridges unordered_map to Avro's map wire format.
template <typename V>
struct codec_traits<std::unordered_map<std::string, V>> {
  static void encode(Encoder& e, const std::unordered_map<std::string, V>& m) {
    e.mapStart();
    if (!m.empty()) {
      e.setItemCount(m.size());
      for (const auto& [k, v] : m) {
        e.startItem();
        avro::encode(e, k);
        avro::encode(e, v);
      }
    }
    e.mapEnd();
  }

  static void decode(Decoder& d, std::unordered_map<std::string, V>& m) {
    m.clear();
    for (size_t n = d.mapStart(); n != 0; n = d.mapNext()) {
      for (size_t i = 0; i < n; ++i) {
        std::string key;
        avro::decode(d, key);
        V val;
        avro::decode(d, val);
        m[std::move(key)] = std::move(val);
      }
    }
  }
};

template <>
struct codec_traits<milvus_storage::api::ColumnGroupFile> {
  static void encode(Encoder& e, const milvus_storage::api::ColumnGroupFile& file) {
    avro::encode(e, file.path);
    avro::encode(e, file.start_index);
    avro::encode(e, file.end_index);
    avro::encode(e, file.properties);
  }

  static void decode(Decoder& d, milvus_storage::api::ColumnGroupFile& file) {
    avro::decode(d, file.path);
    avro::decode(d, file.start_index);
    avro::decode(d, file.end_index);
    avro::decode(d, file.properties);
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
struct codec_traits<milvus_storage::api::LobFileInfo> {
  static void encode(Encoder& e, const milvus_storage::api::LobFileInfo& lob_file) {
    avro::encode(e, lob_file.path);
    avro::encode(e, lob_file.field_id);
    avro::encode(e, lob_file.total_rows);
    avro::encode(e, lob_file.valid_rows);
    avro::encode(e, lob_file.file_size_bytes);
  }

  static void decode(Decoder& d, milvus_storage::api::LobFileInfo& lob_file) {
    avro::decode(d, lob_file.path);
    avro::decode(d, lob_file.field_id);
    avro::decode(d, lob_file.total_rows);
    avro::decode(d, lob_file.valid_rows);
    avro::decode(d, lob_file.file_size_bytes);
  }
};

template <>
struct codec_traits<milvus_storage::api::Manifest> {
  static void encode(Encoder& e, const milvus_storage::api::Manifest& m) {
    avro::encode(e, m.columnGroups());
    avro::encode(e, m.deltaLogs());
    avro::encode(e, m.stats());
    avro::encode(e, m.indexes());
    avro::encode(e, m.lobFiles());
  }

  static void decode(Decoder& d, milvus_storage::api::Manifest& m) {
    avro::decode(d, m.columnGroups());
    avro::decode(d, m.deltaLogs());
    avro::decode(d, m.stats());
    avro::decode(d, m.indexes());
    avro::decode(d, m.lobFiles());
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
            {"name": "properties", "type": {"type": "map", "values": "string"}, "default": {}}
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
    }}, "default": []},
    {"name": "lob_files", "type": {"type": "array", "items": {
      "type": "record", "name": "LobFileInfo", "fields": [
        {"name": "path", "type": "string"},
        {"name": "field_id", "type": "long"},
        {"name": "total_rows", "type": "long"},
        {"name": "valid_rows", "type": "long"},
        {"name": "file_size_bytes", "type": "long"}
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
                   const std::vector<LobFileInfo>& lob_files,
                   uint32_t version)
    : version_(version),
      column_groups_(std::move(column_groups)),
      delta_logs_(delta_logs),
      stats_(stats),
      indexes_(indexes),
      lob_files_(lob_files) {}

Manifest::Manifest(const Manifest& other)
    : version_(other.version_),
      column_groups_(copy_column_groups(other.column_groups_)),
      delta_logs_(other.delta_logs_),
      stats_(other.stats_),
      indexes_(other.indexes_),
      lob_files_(other.lob_files_) {}

Manifest& Manifest::operator=(const Manifest& other) {
  if (this != &other) {
    version_ = other.version_;
    column_groups_ = copy_column_groups(other.column_groups_);
    delta_logs_ = other.delta_logs_;
    stats_ = other.stats_;
    indexes_ = other.indexes_;
    lob_files_ = other.lob_files_;
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
  } catch (...) {
    return arrow::Status::Invalid("Failed to serialize Manifest: unknown error");
  }
}

arrow::Status Manifest::deserialize(std::istream& input_stream) {
  auto error = [this](const std::string& msg) {
    column_groups_.clear();
    delta_logs_.clear();
    stats_.clear();
    indexes_.clear();
    lob_files_.clear();
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

  // Manually decode column groups for legacy format.
  // Legacy ColumnGroupFile does not contain file_size/footer_size fields,
  // so we cannot use codec_traits<ColumnGroupFile> which expects those fields.
  column_groups_.clear();
  for (size_t n = decoder->arrayStart(); n != 0; n = decoder->arrayNext()) {
    for (size_t i = 0; i < n; ++i) {
      auto cg = std::make_shared<ColumnGroup>();
      avro::decode(*decoder, cg->columns);
      // Decode files from legacy format (had metadata bytes, no file_size/footer_size)
      cg->files.clear();
      for (size_t fn = decoder->arrayStart(); fn != 0; fn = decoder->arrayNext()) {
        for (size_t fi = 0; fi < fn; ++fi) {
          ColumnGroupFile file;
          avro::decode(*decoder, file.path);
          avro::decode(*decoder, file.start_index);
          avro::decode(*decoder, file.end_index);
          std::vector<uint8_t> unused_bytes;
          avro::decode(*decoder, unused_bytes);  // legacy metadata field, skipped
          cg->files.push_back(std::move(file));
        }
      }
      avro::decode(*decoder, cg->format);
      column_groups_.push_back(std::move(cg));
    }
  }
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

  if (version >= 5) {
    avro::decode(*decoder, lob_files_);
  } else {
    lob_files_.clear();
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

  // normalize LOB file paths (convert absolute to relative)
  // LOB files live at partition level ({partition}/lobs/{field_id}/_data/), use kLobPath to resolve
  for (auto& lob_file : copy_manifest.lob_files_) {
    lob_file.path = ToRelative(lob_file.path, std::optional<std::string>(base_path), milvus_storage::kLobPath);
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

  // denormalize LOB file paths (convert relative to absolute)
  for (auto& lob_file : lob_files_) {
    lob_file.path = ToAbsolute(lob_file.path, std::optional<std::string>(base_path), milvus_storage::kLobPath);
  }
}

// Manifest files are small (~KB) and immutable once written, so caching
// up to 1024 entries keeps hot-path reads fast without significant memory.
static constexpr size_t kManifestCacheCapacity = 1024;

// Static cache accessor
LRUCache<std::string, std::shared_ptr<Manifest>>& Manifest::getCache() {
  static milvus_storage::LRUCache<std::string, std::shared_ptr<Manifest>> cache(kManifestCacheCapacity);
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

  // Read file into string (single allocation, avro needs std::istream)
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(path));
  ARROW_ASSIGN_OR_RAISE(auto file_size, input_file->GetSize());
  std::string data(file_size, '\0');
  ARROW_ASSIGN_OR_RAISE(auto bytes_read, input_file->Read(file_size, data.data()));
  if (bytes_read != file_size) {
    return arrow::Status::IOError(
        fmt::format("Failed to read the complete file, expected size={}, actual size={}", file_size, bytes_read));
  }
  ARROW_RETURN_NOT_OK(input_file->Close());
  std::istringstream in(data);

  auto manifest = std::make_shared<Manifest>();
  ARROW_RETURN_NOT_OK(manifest->deserialize(in));

  std::string base_path = milvus_storage::base_path_for_manifest(path);
  manifest->ToAbsolutePaths(base_path);

  cache.put(cache_key, manifest);
  return manifest;
}

static bool IsConditionalWriteConflict(const arrow::Status& status) {
  if (!status.IsIOError()) {
    return false;
  }
  auto detail = milvus_storage::ExtendStatusDetail::UnwrapStatus(status);
  if (!detail) {
    return false;
  }
  return detail->code() == milvus_storage::ExtendStatusCode::AwsErrorPreConditionFailed ||
         detail->code() == milvus_storage::ExtendStatusCode::AwsErrorConflict;
}

arrow::Status Manifest::WriteTo(const milvus_storage::ArrowFileSystemPtr& fs,
                                const std::string& path,
                                const Manifest& manifest) {
  std::string base_path = milvus_storage::base_path_for_manifest(path);

  // Convert to relative paths and serialize
  Manifest relative_manifest = manifest.toRelativePaths(base_path);
  std::ostringstream oss;
  ARROW_RETURN_NOT_OK(relative_manifest.serialize(oss));
  std::string data = oss.str();

  // Local filesystem needs parent directory created upfront
  if (milvus_storage::IsLocalFileSystem(fs)) {
    auto [parent, _] = milvus_storage::GetAbstractPathParent(path);
    if (!parent.empty()) {
      ARROW_RETURN_NOT_OK(fs->CreateDir(parent));
    }
  }

  // Try conditional write first if filesystem supports it
  auto conditional_fs = std::dynamic_pointer_cast<milvus_storage::UploadConditional>(fs);
  if (conditional_fs) {
    auto res = conditional_fs->OpenConditionalOutputStream(path, nullptr);
    if (res.ok()) {
      auto output_stream = std::move(res).ValueUnsafe();
      ARROW_RETURN_NOT_OK(output_stream->Write(data.data(), data.size()));
      auto close_status = output_stream->Close();
      if (!close_status.ok()) {
        if (IsConditionalWriteConflict(close_status)) {
          return arrow::Status::AlreadyExists("File already exists: ", path);
        }
        return close_status;
      }
      return arrow::Status::OK();
    }
    if (IsConditionalWriteConflict(res.status())) {
      return arrow::Status::AlreadyExists("File already exists: ", path);
    }
    // NotImplemented means the underlying fs has no conditional write support;
    // fall through to the plain write path below.
    if (!res.status().IsNotImplemented()) {
      return res.status();
    }
  }

  // Fall back: check existence then write
  ARROW_ASSIGN_OR_RAISE(auto file_info, fs->GetFileInfo(path));
  if (file_info.type() != arrow::fs::FileType::NotFound) {
    return arrow::Status::AlreadyExists("File already exists: ", path);
  }

  ARROW_ASSIGN_OR_RAISE(auto output_stream, fs->OpenOutputStream(path));
  ARROW_RETURN_NOT_OK(output_stream->Write(data.data(), data.size()));
  return output_stream->Close();
}

}  // namespace milvus_storage::api
