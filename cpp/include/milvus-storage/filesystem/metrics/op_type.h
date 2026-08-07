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

namespace milvus_storage {

// Backend filesystem operations. Order is fixed and mirrored by the FFI
// snapshot layout; do not reorder.
enum class OpType : int {
  Read = 0,
  Write,
  List,
  Head,
  OpenInput,
  OpenOutput,
  CreateDir,
  DeleteDir,
  DeleteFile,
  Move,
  Copy,
  MultipartCreate,
  MultipartUploadPart,
  MultipartComplete,
  MultipartAbort
};
inline constexpr int kOpTypeCount = 15;

// Operation outcome classification. Order is fixed and mirrored by the FFI
// snapshot layout; do not reorder.
enum class OpStatus : int { Ok = 0, NotFound, Throttled, Auth, Timeout, Network, ServerError, ClientError, Unknown };
inline constexpr int kStatusCount = 9;

inline constexpr int kTransferCount = 3;  // Read, Write, MultipartUploadPart

// Index into transfer arrays, or -1 for non-transfer ops.
inline int TransferIndex(OpType op) {
  switch (op) {
    case OpType::Read:
      return 0;
    case OpType::Write:
      return 1;
    case OpType::MultipartUploadPart:
      return 2;
    default:
      return -1;
  }
}

}  // namespace milvus_storage
