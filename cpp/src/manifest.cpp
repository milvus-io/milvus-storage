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
#include <algorithm>
#include <filesystem>

#include <arrow/status.h>
#include <arrow/result.h>
#include <avro/Decoder.hh>
#include <avro/Encoder.hh>
#include <avro/Specific.hh>
#include <avro/Stream.hh>

#include "milvus-storage/common/path_util.h"
#include "milvus-storage/common/layout.h"

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

}  // namespace avro

namespace milvus_storage::api {

// Manifest format constants (implementation detail)
constexpr int32_t MANIFEST_MAGIC = 0x4D494C56;  // "MILV" in ASCII

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
                   const std::map<std::string, std::vector<std::string>>& stats,
                   const std::vector<LobFileInfo>& lob_files,
                   uint32_t version)
    : version_(version),
      column_groups_(std::move(column_groups)),
      delta_logs_(delta_logs),
      stats_(stats),
      lob_files_(lob_files) {}

Manifest::Manifest(const Manifest& other)
    : version_(other.version_),
      column_groups_(copy_column_groups(other.column_groups_)),
      delta_logs_(other.delta_logs_),
      stats_(other.stats_),
      lob_files_(other.lob_files_) {}

Manifest& Manifest::operator=(const Manifest& other) {
  if (this != &other) {
    version_ = other.version_;
    column_groups_ = copy_column_groups(other.column_groups_);
    delta_logs_ = other.delta_logs_;
    stats_ = other.stats_;
    lob_files_ = other.lob_files_;
  }
  return *this;
}

arrow::Status Manifest::serialize(std::ostream& output_stream, const std::optional<std::string>& base_path) const {
  try {
    if (base_path.has_value()) {
      Manifest normalized_manifest = ToRelativePaths(base_path.value());
      return normalized_manifest.serialize(output_stream, std::nullopt);
    }

    // Convert std::ostream to avro::OutputStream
    std::unique_ptr<avro::OutputStream> avro_output = avro::ostreamOutputStream(output_stream);

    // Create encoder from output stream
    avro::EncoderPtr encoder = avro::binaryEncoder();
    encoder->init(*avro_output);

    // Encode MAGIC number first (through Avro encoder to maintain Avro compatibility)
    avro::encode(*encoder, MANIFEST_MAGIC);

    // Encode version second
    avro::encode(*encoder, version_);

    // Encode column groups
    avro::encode(*encoder, column_groups_);

    // Encode delta logs
    avro::encode(*encoder, delta_logs_);

    // Encode stats
    avro::encode(*encoder, stats_);

    // Encode LOB files (v2+)
    avro::encode(*encoder, lob_files_);

    // Flush encoder to ensure all data is written
    encoder->flush();

    return arrow::Status::OK();
  } catch (const avro::Exception& e) {
    return arrow::Status::Invalid("Failed to serialize Manifest: " + std::string(e.what()));
  } catch (const std::exception& e) {
    return arrow::Status::Invalid("Failed to serialize Manifest (std::exception): " + std::string(e.what()));
  }
}

arrow::Status Manifest::deserialize(std::istream& input_stream, const std::optional<std::string>& base_path) {
  // Helper to clear state and return error
  auto error = [this](const std::string& msg) {
    column_groups_.clear();
    delta_logs_.clear();
    stats_.clear();
    lob_files_.clear();
    return arrow::Status::Invalid(msg);
  };

  try {
    // Check the stream length.
    input_stream.seekg(0, std::ios::end);
    std::streampos end_pos = input_stream.tellg();
    input_stream.seekg(0, std::ios::beg);

    if (end_pos <= 0 || (end_pos == std::streampos(-1) && input_stream.fail())) {
      return error("Cannot deserialize Manifest: stream is empty or invalid");
    }

    // Reset stream state after size check
    input_stream.clear();
    input_stream.seekg(0, std::ios::beg);

    // Create Avro input stream and decoder
    std::unique_ptr<avro::InputStream> avro_input = avro::istreamInputStream(input_stream);
    avro::DecoderPtr decoder = avro::binaryDecoder();
    decoder->init(*avro_input);

    // Read and validate MAGIC number
    int32_t magic = 0;
    avro::decode(*decoder, magic);
    if (magic != MANIFEST_MAGIC) {
      return error("Invalid MAGIC number: not a valid Manifest file (expected " + std::to_string(MANIFEST_MAGIC) +
                   ", got " + std::to_string(magic) + ")");
    }

    // Read and validate version
    int32_t version = 0;
    avro::decode(*decoder, version);
    version_ = version;
    if (version < MANIFEST_VERSION_MIN || version > MANIFEST_VERSION) {
      return error("Unsupported manifest version: " + std::to_string(version) + " (supported: " +
                   std::to_string(MANIFEST_VERSION_MIN) + "-" + std::to_string(MANIFEST_VERSION) + ")");
    }

    // Decode v1 fields (always present)
    avro::decode(*decoder, column_groups_);
    avro::decode(*decoder, delta_logs_);
    avro::decode(*decoder, stats_);

    // Decode v2+ fields
    if (version >= 2) {
      avro::decode(*decoder, lob_files_);
    } else {
      lob_files_.clear();  // v1 manifest has no LOB files
    }

    // resolve the absolute path to relative path
    // direct copy modify the original column groups
    if (base_path.has_value()) {
      ToAbsolutePaths(base_path.value());
    }

    return arrow::Status::OK();
  } catch (const avro::Exception& e) {
    return error("Failed to deserialize Manifest: " + std::string(e.what()));
  } catch (const std::exception& e) {
    return error("Failed to deserialize Manifest: " + std::string(e.what()));
  } catch (...) {
    return error("Failed to deserialize Manifest: unknown error (possibly invalid or empty stream)");
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

Manifest Manifest::ToRelativePaths(const std::string& base_path) const {
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
  for (auto& [key, files] : copy_manifest.stats_) {
    for (auto& file : files) {
      file = ToRelative(file, std::optional<std::string>(base_path), milvus_storage::kStatsPath);
    }
  }

  // normalize LOB file paths (convert absolute to relative)
  for (auto& lob_file : copy_manifest.lob_files_) {
    lob_file.path = ToRelative(lob_file.path, std::optional<std::string>(base_path), milvus_storage::kDataPath);
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
  for (auto& [key, files] : stats_) {
    for (auto& file : files) {
      file = ToAbsolute(file, std::optional<std::string>(base_path), milvus_storage::kStatsPath);
    }
  }

  // denormalize LOB file paths (convert relative to absolute)
  for (auto& lob_file : lob_files_) {
    lob_file.path = ToAbsolute(lob_file.path, std::optional<std::string>(base_path), milvus_storage::kDataPath);
  }
}

}  // namespace milvus_storage::api
