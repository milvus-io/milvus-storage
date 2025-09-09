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

#pragma once

#include <memory>
#include <parquet/properties.h>
#include "milvus-storage/common/config.h"
#include "milvus-storage/writer.h"

namespace milvus_storage::parquet {

/**
 * @brief Converts API compression type to parquet compression type
 *
 * @param compression The API compression type
 * @return The corresponding parquet compression type
 */
::parquet::Compression::type convert_compression_type(milvus_storage::api::CompressionType compression);

/**
 * @brief Converts WriteProperties to parquet::WriterProperties
 *
 * @param properties The API write properties
 * @return The corresponding parquet writer properties
 */
std::shared_ptr<::parquet::WriterProperties> convert_write_properties(
    const milvus_storage::api::WriteProperties& properties);

}  // namespace milvus_storage::parquet