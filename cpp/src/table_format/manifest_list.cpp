// Copyright 2024 Zilliz
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

#include "milvus-storage/table_format/manifest_list.h"
#include "milvus-storage/table_format/layout.h"
#include "milvus-storage/table_format/types_codec.h"

#include <sstream>

#include <arrow/buffer.h>
#include <arrow/filesystem/filesystem.h>
#include <avro/Compiler.hh>
#include <avro/DataFile.hh>
#include <avro/Stream.hh>
#include <fmt/format.h>

namespace milvus_storage::api::table_format {

static const char* const PARTITION_MANIFEST_ENTRY_SCHEMA_JSON = R"({
  "type": "record",
  "name": "ManifestListEntry",
  "namespace": "milvus_storage.table_format",
  "fields": [
    {"name": "partition_id", "type": "long"},
    {"name": "partition_name", "type": "string"},
    {"name": "segments", "type": {"type": "array", "items": {
      "type": "record", "name": "SegmentInfo", "fields": [
        {"name": "segment_id", "type": "long"},
        {"name": "manifest", "type": "string"},
        {"name": "level", "type": "int", "default": 0},
        {"name": "num_rows", "type": "long", "default": 0},
        {"name": "file_size", "type": "long", "default": 0},
        {"name": "index_size", "type": "long", "default": 0},
        {"name": "sorted", "type": "boolean", "default": false},
        {"name": "partition_key_sorted", "type": "boolean", "default": false}
      ]
    }}}
  ]
})";

static const avro::ValidSchema& getManifestListSchema() {
  static const avro::ValidSchema schema = avro::compileJsonSchemaFromString(PARTITION_MANIFEST_ENTRY_SCHEMA_JSON);
  return schema;
}

ManifestList::ManifestList(std::vector<ManifestListEntry> entries) : entries_(std::move(entries)) {}

arrow::Status ManifestList::serialize(std::ostream& output_stream) const {
  try {
    auto avro_output = avro::ostreamOutputStream(output_stream);
    avro::DataFileWriter<ManifestListEntry> writer(std::move(avro_output), getManifestListSchema());
    for (const auto& entry : entries_) {
      writer.write(entry);
    }
    writer.close();
    return arrow::Status::OK();
  } catch (const avro::Exception& e) {
    return arrow::Status::Invalid(fmt::format("Failed to serialize ManifestList: {}", e.what()));
  } catch (const std::exception& e) {
    return arrow::Status::Invalid(fmt::format("Failed to serialize ManifestList: {}", e.what()));
  }
}

arrow::Status ManifestList::deserialize(std::istream& input_stream) {
  try {
    entries_.clear();
    auto avro_input = avro::istreamInputStream(input_stream);
    avro::DataFileReader<ManifestListEntry> reader(std::move(avro_input), getManifestListSchema());
    ManifestListEntry entry;
    while (reader.read(entry)) {
      entries_.push_back(std::move(entry));
      entry = ManifestListEntry{};
    }
    return arrow::Status::OK();
  } catch (const avro::Exception& e) {
    return arrow::Status::Invalid(fmt::format("Failed to deserialize ManifestList: {}", e.what()));
  } catch (const std::exception& e) {
    return arrow::Status::Invalid(fmt::format("Failed to deserialize ManifestList: {}", e.what()));
  }
}

arrow::Result<ManifestList> ReadManifestListFromFile(const milvus_storage::ArrowFileSystemPtr& fs,
                                                     const std::string& path) {
  ARROW_ASSIGN_OR_RAISE(auto input, fs->OpenInputFile(path));
  ARROW_ASSIGN_OR_RAISE(auto size, input->GetSize());
  ARROW_ASSIGN_OR_RAISE(auto buffer, input->Read(size));
  ARROW_RETURN_NOT_OK(input->Close());
  std::string data(reinterpret_cast<const char*>(buffer->data()), buffer->size());
  std::istringstream iss(data);
  ManifestList ml;
  ARROW_RETURN_NOT_OK(ml.deserialize(iss));
  return std::move(ml);
}

arrow::Result<std::string> WriteManifestListToFile(const milvus_storage::ArrowFileSystemPtr& fs,
                                                   const std::string& base_path,
                                                   const ManifestList& ml) {
  ARROW_RETURN_NOT_OK(fs->CreateDir(GetCollManifestsDir(base_path)));
  std::string path = GetManifestListFilepath(base_path, GenerateUniqueId());
  std::ostringstream oss;
  ARROW_RETURN_NOT_OK(ml.serialize(oss));
  std::string data = oss.str();
  ARROW_ASSIGN_OR_RAISE(auto out, fs->OpenOutputStream(path));
  ARROW_RETURN_NOT_OK(out->Write(data.data(), static_cast<int64_t>(data.size())));
  ARROW_RETURN_NOT_OK(out->Close());
  return path;
}

}  // namespace milvus_storage::api::table_format
