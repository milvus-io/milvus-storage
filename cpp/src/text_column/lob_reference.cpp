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

#include "milvus-storage/text_column/lob_reference.h"

#include <cstring>
#include <filesystem>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace milvus_storage::text_column {

std::vector<uint8_t> EncodeInlineText(const std::string& text) {
  // inline text: [flag=0x00] + [text bytes]
  // variable length, stores complete text
  std::vector<uint8_t> data(1 + text.size());
  data[0] = FLAG_INLINE_DATA;
  std::memcpy(&data[1], text.data(), text.size());
  return data;
}

std::vector<uint8_t> EncodeLOBReference(const uint8_t* file_id, int32_t row_offset) {
  // LOB reference (24 bytes, 4-byte aligned):
  // [flag=0x01 (1B)] [padding (3B)] [file_id (16B)] [row_offset (4B)]
  std::vector<uint8_t> data(LOB_REFERENCE_SIZE, 0);  // zero-initialize for padding
  data[0] = FLAG_LOB_REFERENCE;
  // bytes 1-3 are padding (already zeroed)
  // copy UUID binary (16 bytes) at offset 4
  std::memcpy(&data[4], file_id, UUID_BINARY_SIZE);
  // row_offset at offset 20 (4 + 16)
  std::memcpy(&data[20], &row_offset, sizeof(int32_t));
  return data;
}

std::string DecodeInlineText(const uint8_t* data, size_t data_size) {
  // skip flag byte, return the rest as text
  if (data_size <= 1) {
    return "";
  }
  return std::string(reinterpret_cast<const char*>(data + 1), data_size - 1);
}

void DecodeLOBReference(const uint8_t* data, std::string* file_id_str, int32_t* row_offset) {
  // skip flag (1B) and padding (3B), read file_id (binary) and row_offset
  // file_id is at offset 4, length 16 bytes (binary UUID)
  *file_id_str = UUIDToString(data + 4);
  // row_offset at offset 20 (4 + 16)
  std::memcpy(row_offset, data + 20, sizeof(int32_t));
}

bool IsLOBReference(const uint8_t* data) { return data[0] == FLAG_LOB_REFERENCE; }

bool IsInlineData(const uint8_t* data) { return data[0] == FLAG_INLINE_DATA; }

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

}  // namespace milvus_storage::text_column
