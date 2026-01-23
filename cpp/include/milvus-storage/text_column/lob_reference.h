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

#include <cstdint>
#include <string>
#include <vector>

namespace milvus_storage::text_column {

// encoding format:
//
// 1. inline text (variable length):
//    [flag=0x00 (1 byte)] [text bytes...]
//    total size = 1 + text.length()
//
// 2. LOB reference (fixed 44 bytes, 4-byte aligned):
//    [flag=0x01 (1 byte)] [padding (3 bytes)] [file_id (36 bytes)] [row_offset (4 bytes)]
//    total size = 1 + 3 + 36 + 4 = 44 bytes
//
//    file_id is stored as UUID string format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (36 chars)
//    This eliminates the need for binary-to-string conversion during reads.
//
//    ┌─────────┬─────────┬──────────────────────────────────────────┬────────────────┐
//    │ Flag    │ Padding │ File ID (UUID string)                    │ Offset in File │
//    │ (1B)    │ (3B)    │ (36B)                                    │ (4B)           │
//    │ 0x01    │ 0x00    │ "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"   │ int32          │
//    └─────────┴─────────┴──────────────────────────────────────────┴────────────────┘

// flag values
constexpr uint8_t FLAG_INLINE_DATA = 0x00;
constexpr uint8_t FLAG_LOB_REFERENCE = 0x01;

// UUID string length (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
constexpr size_t UUID_STRING_SIZE = 36;

// LOB reference size (fixed): flag + padding + file_id_string + row_offset = 1 + 3 + 36 + 4 = 44 bytes
constexpr size_t LOB_REFERENCE_SIZE = 44;

// default threshold for inline text (texts smaller than this are stored inline)
// this is a suggested default, actual threshold is configurable
constexpr size_t DEFAULT_INLINE_THRESHOLD = 65536;  // 64KB

/**
 * @brief encode inline text (variable length)
 * @param text the text string to encode
 * @return encoded bytes: [0x00] + [text bytes], length = 1 + text.size()
 */
std::vector<uint8_t> EncodeInlineText(const std::string& text);

/**
 * @brief encode LOB reference (fixed 44 bytes)
 * @param file_id_str UUID string of the LOB file (36 chars: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
 * @param row_offset row offset in the Vortex file
 * @return encoded bytes: [0x01] + [padding] + [file_id_str] + [row_offset], length = 44
 */
std::vector<uint8_t> EncodeLOBReference(const std::string& file_id_str, int32_t row_offset);

/**
 * @brief decode inline text from encoded data
 * @param data encoded bytes starting with flag
 * @param data_size total size of the encoded data
 * @return decoded text string (without the flag byte)
 */
std::string DecodeInlineText(const uint8_t* data, size_t data_size);

/**
 * @brief decode LOB reference from encoded data
 * @param data encoded bytes (must be at least 44 bytes)
 * @param file_id_str output: UUID string (36 chars)
 * @param row_offset output: row offset
 */
void DecodeLOBReference(const uint8_t* data, std::string* file_id_str, int32_t* row_offset);

/**
 * @brief check if data is a LOB reference (flag == 0x01)
 * @param data encoded bytes
 * @return true if it's a LOB reference
 */
bool IsLOBReference(const uint8_t* data);

/**
 * @brief check if data is inline text (flag == 0x00)
 * @param data encoded bytes
 * @return true if it's inline text
 */
bool IsInlineData(const uint8_t* data);

/**
 * @brief build LOB file path from file_id UUID string
 * @param lob_base_path base path for LOB files (e.g., {partition}/lobs/{field_id})
 * @param file_id_str UUID string of the LOB file (36 chars)
 * @return full path to the LOB file (e.g., {lob_base_path}/_data/{uuid}.vx)
 */
std::string BuildLOBFilePath(const std::string& lob_base_path, const std::string& file_id_str);

/**
 * @brief generate a random UUID v4 and return as string
 * @return UUID string (36 chars: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
 */
std::string GenerateUUIDString();

}  // namespace milvus_storage::text_column
