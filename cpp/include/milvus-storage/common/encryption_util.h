// Copyright 2026 Zilliz
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

#include <string>
#include <arrow/result.h>
#include <arrow/util/base64.h>

namespace milvus_storage {

// Decode text carried by encryption properties and C callbacks. Never include
// key material in errors. Native C++ reader callbacks already return raw bytes.
inline arrow::Result<std::string> DecodeBase64EncryptionKey(const std::string& encoded) {
  constexpr const char* error = "Encryption key must be standard padded Base64 of a 16, 24 or 32 byte AES key";
  if ((encoded.size() != 24 && encoded.size() != 32 && encoded.size() != 44) ||
      encoded.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=") !=
          std::string::npos) {
    return arrow::Status::Invalid(error);
  }
  auto key = arrow::util::base64_decode(encoded);
  // Arrow accepts prefixes and ignores trailing input; require a canonical
  // round trip as well as a valid AES key length.
  if (arrow::util::base64_encode(key) != encoded || (key.size() != 16 && key.size() != 24 && key.size() != 32)) {
    return arrow::Status::Invalid(error);
  }
  return key;
}

}  // namespace milvus_storage
