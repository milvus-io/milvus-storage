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

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace milvus_storage::lob_column {

// encoding format:
//
// 1. inline data (variable length):
//    [flag=0x00 (1 byte)] [payload bytes...]
//    total size = 1 + payload.length()
//    payload can be text (UTF-8) or binary data; the caller determines the type.
//
// 2. LOB reference (fixed 24 bytes, 4-byte aligned):
//    [flag=0x01 (1 byte)] [padding (3 bytes)] [file_id (16 bytes)] [row_offset (4 bytes)]
//    total size = 1 + 3 + 16 + 4 = 24 bytes
//
//    file_id is stored as binary UUID (16 bytes).
//    Converted to string format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) when building file paths.
//
//    ┌─────────┬─────────┬──────────────────┬────────────────┐
//    │ Flag    │ Padding │ File ID (binary) │ Offset in File │
//    │ (1B)    │ (3B)    │ (16B)            │ (4B)           │
//    │ 0x01    │ 0x00    │ UUID bytes       │ int32          │
//    └─────────┴─────────┴──────────────────┴────────────────┘

// flag values
constexpr uint8_t FLAG_INLINE_DATA = 0x00;
constexpr uint8_t FLAG_LOB_REFERENCE = 0x01;

// UUID binary size (16 bytes)
constexpr size_t UUID_BINARY_SIZE = 16;

// UUID string length (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
constexpr size_t UUID_STRING_SIZE = 36;

// LOB reference size (fixed): flag + padding + file_id_binary + row_offset = 1 + 3 + 16 + 4 = 24 bytes
constexpr size_t LOB_REFERENCE_SIZE = 24;

// fixed-size encoded LOB reference (24 bytes, lives entirely on the stack)
struct EncodedLOBRef {
  std::array<uint8_t, LOB_REFERENCE_SIZE> bytes = {};

  const uint8_t* data() const { return bytes.data(); }
  size_t size() const { return LOB_REFERENCE_SIZE; }
  uint8_t& operator[](size_t i) { return bytes[i]; }
  uint8_t operator[](size_t i) const { return bytes[i]; }

  operator std::vector<uint8_t>() const { return {bytes.begin(), bytes.end()}; }
};

struct LOBReference {
  std::string file_id;
  int32_t row_offset;
};

// --- generic inline data encode/decode (works for both text and binary) ---

/**
 * @brief encode inline data (variable length)
 * @param payload raw bytes to encode
 * @param size number of bytes in payload
 * @return encoded bytes: [0x00] + [payload bytes], length = 1 + size
 */
std::vector<uint8_t> EncodeInlineData(const uint8_t* payload, size_t size);

/**
 * @brief decode inline data from encoded bytes (zero-copy)
 * @param data encoded bytes starting with flag
 * @param data_size total size of the encoded data
 * @param out_payload output: pointer to the payload within data (after flag byte)
 * @param out_size output: size of the payload
 */
void DecodeInlineData(const uint8_t* data, size_t data_size, const uint8_t** out_payload, size_t* out_size);

// --- text convenience wrappers ---

/**
 * @brief encode inline text (variable length), convenience wrapper over EncodeInlineData
 * @param text the text string to encode
 * @return encoded bytes: [0x00] + [text bytes], length = 1 + text.size()
 */
inline std::vector<uint8_t> EncodeInlineText(const std::string& text) {
  return EncodeInlineData(reinterpret_cast<const uint8_t*>(text.data()), text.size());
}

/**
 * @brief decode inline text from encoded data, convenience wrapper over DecodeInlineData
 * @param data encoded bytes starting with flag
 * @param data_size total size of the encoded data
 * @return decoded text string (without the flag byte)
 */
inline std::string DecodeInlineText(const uint8_t* data, size_t data_size) {
  const uint8_t* payload;
  size_t payload_size;
  DecodeInlineData(data, data_size, &payload, &payload_size);
  return std::string(reinterpret_cast<const char*>(payload), payload_size);
}

// --- LOB reference encode/decode ---

/**
 * @brief encode LOB reference (fixed 24 bytes)
 * @param file_id binary UUID of the LOB file (16 bytes)
 * @param row_offset row offset in the Vortex file
 * @return stack-allocated 24-byte encoded reference
 */
EncodedLOBRef EncodeLOBReference(const uint8_t* file_id, int32_t row_offset);

/**
 * @brief decode LOB reference from encoded data
 * @param data encoded bytes (must be at least 24 bytes)
 * @return LOBReference with file_id (UUID string, 36 chars) and row_offset
 */
LOBReference DecodeLOBReference(const uint8_t* data);

/**
 * @brief check if data is a LOB reference (flag == 0x01)
 * @param data encoded bytes
 * @return true if it's a LOB reference
 */
inline bool IsLOBReference(const uint8_t* data) { return data[0] == FLAG_LOB_REFERENCE; }

/**
 * @brief check if data is inline data (flag == 0x00)
 * @param data encoded bytes
 * @return true if it's inline data
 */
inline bool IsInlineData(const uint8_t* data) { return data[0] == FLAG_INLINE_DATA; };

/**
 * @brief build LOB file path from file_id UUID string
 * @param lob_base_path base path for LOB files (e.g., {partition}/lobs/{field_id})
 * @param file_id_str UUID string of the LOB file (36 chars)
 * @return full path to the LOB file (e.g., {lob_base_path}/_data/{uuid}.vx)
 */
std::string BuildLOBFilePath(const std::string& lob_base_path, const std::string& file_id_str);

/**
 * @brief generate a random UUID v4 as binary
 * @param out output buffer (must be at least 16 bytes)
 */
void GenerateUUIDBinary(uint8_t* out);

/**
 * @brief convert binary UUID to string format
 * @param uuid binary UUID (16 bytes)
 * @return UUID string (36 chars: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
 */
std::string UUIDToString(const uint8_t* uuid);

}  // namespace milvus_storage::lob_column
