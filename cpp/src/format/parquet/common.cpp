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

#include "milvus-storage/format/parquet/common.h"

namespace milvus_storage::parquet {

::parquet::Compression::type convert_compression_type(const std::string& compression) {
  if (compression == "uncompressed") {
    return ::parquet::Compression::UNCOMPRESSED;
  } else if (compression == "snappy") {
    return ::parquet::Compression::SNAPPY;
  } else if (compression == "gzip") {
    return ::parquet::Compression::GZIP;
  } else if (compression == "lz4") {
    return ::parquet::Compression::LZ4;
  } else if (compression == "zstd") {
    return ::parquet::Compression::ZSTD;
  } else if (compression == "brotli") {
    return ::parquet::Compression::BROTLI;
  } else {
    return ::parquet::Compression::ZSTD;
  }
}

std::shared_ptr<::parquet::WriterProperties> convert_write_properties(
    const milvus_storage::api::Properties& properties) {
  ::parquet::WriterProperties::Builder builder;

  // Set compression
  auto compression = milvus_storage::api::GetValue(properties, milvus_storage::api::WriteCompressionKey);
  builder.compression(convert_compression_type(compression));

  auto compression_level = milvus_storage::api::GetValue(properties, milvus_storage::api::WriteCompressionLevelKey);
  if (compression_level >= 0) {
    builder.compression_level(compression_level);
  }

  auto enable_dictionary = milvus_storage::api::GetValue(properties, milvus_storage::api::WriteEnableDictionaryKey);
  if (enable_dictionary) {
    builder.enable_dictionary();
  } else {
    builder.disable_dictionary();
  }

  return builder.build();
}

}  // namespace milvus_storage::parquet