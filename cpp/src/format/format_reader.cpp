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

#include "milvus-storage/format/format_reader.h"

#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage {

std::string RowGroupInfo::ToString() const {
  std::stringstream ss;
  ss << "RowGroupInfo{"
     << "start_offset=" << start_offset << ", end_offset=" << end_offset << ", memory_size=" << memory_size << "}";
  return ss.str();
}

arrow::Result<std::unique_ptr<FormatReader>> FormatReader::create(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::string& format,
    const std::string& path,
    const milvus_storage::api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever) {
  ARROW_ASSIGN_OR_RAISE(auto file_system, milvus_storage::FilesystemCache::getInstance().get(properties, path));
  if (format == LOON_FORMAT_PARQUET) {
    return std::make_unique<milvus_storage::parquet::ParquetFormatReader>(file_system, path, properties, needed_columns,
                                                                          key_retriever);
  }
#ifdef BUILD_VORTEX_BRIDGE
  else if (format == LOON_FORMAT_VORTEX) {
    return std::make_unique<milvus_storage::vortex::VortexFormatReader>(file_system, schema, path, properties,
                                                                        needed_columns);
  }
#endif  // BUILD_VORTEX_BRIDGE
  else {
    return arrow::Status::Invalid("Unsupported file format: " + format);
  }
}

}  // namespace milvus_storage