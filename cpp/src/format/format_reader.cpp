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

#include <sstream>

#include "milvus-storage/format/format.h"

namespace milvus_storage {

std::string RowGroupInfo::ToString() const {
  std::stringstream ss;
  ss << "RowGroupInfo{"
     << "start_offset=" << start_offset << ", end_offset=" << end_offset << ", memory_size=" << memory_size << "}";
  return ss.str();
}

arrow::Result<std::shared_ptr<FormatReader>> FormatReader::create(
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::string& format,
    const api::ColumnGroupFile& file,
    const api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever) {
  ARROW_ASSIGN_OR_RAISE(auto* fmt, Format::get(format));
  return fmt->create_reader(read_schema, file, properties, needed_columns, key_retriever);
}

}  // namespace milvus_storage
