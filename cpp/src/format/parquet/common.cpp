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

::parquet::Compression::type convert_compression_type(milvus_storage::api::CompressionType compression) {
  switch (compression) {
    case milvus_storage::api::CompressionType::UNCOMPRESSED:
      return ::parquet::Compression::UNCOMPRESSED;
    case milvus_storage::api::CompressionType::SNAPPY:
      return ::parquet::Compression::SNAPPY;
    case milvus_storage::api::CompressionType::GZIP:
      return ::parquet::Compression::GZIP;
    case milvus_storage::api::CompressionType::LZ4:
      return ::parquet::Compression::LZ4;
    case milvus_storage::api::CompressionType::ZSTD:
      return ::parquet::Compression::ZSTD;
    case milvus_storage::api::CompressionType::BROTLI:
      return ::parquet::Compression::BROTLI;
    default:
      return ::parquet::Compression::ZSTD;
  }
}

std::shared_ptr<::parquet::WriterProperties> convert_write_properties(
    const milvus_storage::api::WriteProperties& properties) {
  ::parquet::WriterProperties::Builder builder;

  // Set compression
  builder.compression(convert_compression_type(properties.compression));

  if (properties.compression_level >= 0) {
    builder.compression_level(properties.compression_level);
  }

  if (properties.enable_dictionary) {
    builder.enable_dictionary();
  } else {
    builder.disable_dictionary();
  }

  return builder.build();
}

}  // namespace milvus_storage::parquet