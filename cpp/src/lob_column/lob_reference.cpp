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

#include "milvus-storage/lob_column/lob_reference.h"

#include <cstring>
#include <filesystem>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace milvus_storage::lob_column {

std::vector<uint8_t> EncodeInlineData(const uint8_t* payload, size_t size) {
  std::vector<uint8_t> data(1 + size);
  data[0] = FLAG_INLINE_DATA;
  if (size > 0) {
    std::memcpy(&data[1], payload, size);
  }
  return data;
}

void DecodeInlineData(const uint8_t* data, size_t data_size, const uint8_t** out_payload, size_t* out_size) {
  if (data_size <= 1) {
    *out_payload = nullptr;
    *out_size = 0;
    return;
  }
  *out_payload = data + 1;
  *out_size = data_size - 1;
}

EncodedLOBRef EncodeLOBReference(const uint8_t* file_id, int32_t row_offset) {
  EncodedLOBRef ref;  // zero-initialized by default
  ref[0] = FLAG_LOB_REFERENCE;
  std::memcpy(&ref[4], file_id, UUID_BINARY_SIZE);
  std::memcpy(&ref[20], &row_offset, sizeof(int32_t));
  return ref;
}

LOBReference DecodeLOBReference(const uint8_t* data) {
  LOBReference ref;
  ref.file_id = UUIDToString(data + 4);
  std::memcpy(&ref.row_offset, data + 20, sizeof(int32_t));
  return ref;
}

void GenerateUUIDBinary(uint8_t* out) {
  // use boost::uuids for consistent UUID generation with the rest of the project
  static thread_local boost::uuids::random_generator gen;
  boost::uuids::uuid boost_uuid = gen();
  std::memcpy(out, boost_uuid.data, UUID_BINARY_SIZE);
}

std::string UUIDToString(const uint8_t* uuid) {
  // convert binary UUID to string format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
  boost::uuids::uuid boost_uuid;
  std::memcpy(boost_uuid.data, uuid, UUID_BINARY_SIZE);
  return boost::uuids::to_string(boost_uuid);
}

std::string BuildLOBFilePath(const std::string& lob_base_path, const std::string& file_id_str) {
  std::filesystem::path path = std::filesystem::path(lob_base_path) / "_data" / (file_id_str + ".vx");
  return path.string();
}

}  // namespace milvus_storage::lob_column
