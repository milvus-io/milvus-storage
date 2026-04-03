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

#include "milvus-storage/format/vortex/vortex_format.h"

#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/format/vortex/vortex_writer.h"

namespace milvus_storage {

std::shared_ptr<FormatReader> VortexFormat::make_reader(
    const std::shared_ptr<arrow::fs::FileSystem>& fs,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::string& resolved_path,
    const api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& /*key_retriever*/,
    uint64_t file_size,
    uint64_t footer_size) {
  return std::make_shared<vortex::VortexFormatReader>(fs, read_schema, resolved_path, properties, needed_columns,
                                                      file_size, footer_size);
}

arrow::Result<std::unique_ptr<FormatWriter>> VortexFormat::make_writer(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                                       const std::shared_ptr<arrow::Schema>& schema,
                                                                       const std::string& file_path,
                                                                       const api::Properties& properties) {
  return std::make_unique<vortex::VortexFileWriter>(fs, schema, file_path, properties);
}

}  // namespace milvus_storage
