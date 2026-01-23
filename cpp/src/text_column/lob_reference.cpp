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

std::vector<uint8_t> EncodeLOBReference(const std::string& file_id_str, int32_t row_offset) {
  // LOB reference (44 bytes, 4-byte aligned):
  // [flag=0x01 (1B)] [padding (3B)] [file_id_str (36B)] [row_offset (4B)]
  std::vector<uint8_t> data(LOB_REFERENCE_SIZE, 0);  // zero-initialize for padding
  data[0] = FLAG_LOB_REFERENCE;
  // bytes 1-3 are padding (already zeroed)
  // copy UUID string (36 bytes) at offset 4
  size_t copy_len = std::min(file_id_str.size(), UUID_STRING_SIZE);
  std::memcpy(&data[4], file_id_str.data(), copy_len);
  // row_offset at offset 40 (4 + 36)
  std::memcpy(&data[40], &row_offset, sizeof(int32_t));
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
  // skip flag (1B) and padding (3B), read file_id_str and row_offset
  // file_id_str is at offset 4, length 36 bytes
  *file_id_str = std::string(reinterpret_cast<const char*>(data + 4), UUID_STRING_SIZE);
  // row_offset at offset 40 (4 + 36)
  std::memcpy(row_offset, data + 40, sizeof(int32_t));
}

bool IsLOBReference(const uint8_t* data) { return data[0] == FLAG_LOB_REFERENCE; }

bool IsInlineData(const uint8_t* data) { return data[0] == FLAG_INLINE_DATA; }

std::string GenerateUUIDString() {
  // use boost::uuids for consistent UUID generation with the rest of the project
  static thread_local boost::uuids::random_generator gen;
  boost::uuids::uuid boost_uuid = gen();
  return boost::uuids::to_string(boost_uuid);
}

std::string BuildLOBFilePath(const std::string& lob_base_path, const std::string& file_id_str) {
  std::filesystem::path path = std::filesystem::path(lob_base_path) / "_data" / (file_id_str + ".vx");
  return path.string();
}

}  // namespace milvus_storage::text_column
