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

#include <arrow/filesystem/filesystem.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "milvus-storage/lob_column/lob_reference.h"
#include "milvus-storage/lob_column/lob_column_manager.h"

namespace milvus_storage::lob_column {

// encoded reference with size information
// needed because inline data is variable length
struct EncodedRef {
  const uint8_t* data;
  size_t size;
};

// reader for a single LOB column (TEXT or BINARY)
// resolves encoded references and reads data from LOB files
//
// the reader handles both inline and LOB references:
//   - inline (flag=0x00): data is directly in the reference, variable length
//   - LOB (flag=0x01): data is in a Vortex LOB file, fixed 24 bytes
//
// for batch reads, the reader optimizes by:
//   1. grouping references by file_id_str (UUID string)
//   2. using Vortex take() API for efficient random access
//   3. caching open file handles
//
// thread safety: not thread-safe, use one reader per thread
class LobColumnReader {
  public:
  virtual ~LobColumnReader() = default;

  // --- generic data interface (works for both text and binary) ---

  // read a single entry from an encoded reference
  // handles both inline (variable length) and LOB (24 bytes) references
  virtual arrow::Result<std::vector<uint8_t>> ReadData(const uint8_t* encoded_ref, size_t ref_size) = 0;

  // read multiple entries from encoded references
  // more efficient than calling ReadData() repeatedly as it groups reads by file_id
  virtual arrow::Result<std::vector<std::vector<uint8_t>>> ReadBatchData(
      const std::vector<EncodedRef>& encoded_refs) = 0;

  // read data from a BinaryArray of encoded references
  // returns a BinaryArray with the decoded data
  virtual arrow::Result<std::shared_ptr<arrow::BinaryArray>> ReadArrowArray(
      const std::shared_ptr<arrow::BinaryArray>& encoded_refs) = 0;

  // take entries by row indices from a specific LOB file
  // returns data in the order of row_offsets
  virtual arrow::Result<std::vector<std::vector<uint8_t>>> TakeData(const std::string& file_id_str,
                                                                    const std::vector<int32_t>& row_offsets) = 0;

  // testonly: production code uses ReadData() directly
  arrow::Result<std::string> ReadText(const uint8_t* encoded_ref, size_t ref_size) {
    ARROW_ASSIGN_OR_RAISE(auto data, ReadData(encoded_ref, ref_size));
    return std::string(reinterpret_cast<const char*>(data.data()), data.size());
  }

  // testonly: production code uses ReadBatchData() directly
  arrow::Result<std::vector<std::string>> ReadBatch(const std::vector<EncodedRef>& encoded_refs) {
    ARROW_ASSIGN_OR_RAISE(auto data_vec, ReadBatchData(encoded_refs));
    std::vector<std::string> results;
    results.reserve(data_vec.size());
    for (const auto& d : data_vec) {
      results.emplace_back(reinterpret_cast<const char*>(d.data()), d.size());
    }
    return results;
  }

  arrow::Result<std::vector<std::string>> Take(const std::string& file_id_str,
                                               const std::vector<int32_t>& row_offsets) {
    ARROW_ASSIGN_OR_RAISE(auto data_vec, TakeData(file_id_str, row_offsets));
    std::vector<std::string> results;
    results.reserve(data_vec.size());
    for (const auto& d : data_vec) {
      results.emplace_back(reinterpret_cast<const char*>(d.data()), d.size());
    }
    return results;
  }

  // --- lifecycle methods ---

  // close the reader and release resources
  virtual arrow::Status Close() = 0;

  // check if the reader is closed
  virtual bool IsClosed() const = 0;

  // clear the file handle cache
  virtual void ClearCache() = 0;

  protected:
  LobColumnReader() = default;
};

// factory function to create LobColumnReader
arrow::Result<std::unique_ptr<LobColumnReader>> CreateLobColumnReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                      const LobColumnConfig& config);

}  // namespace milvus_storage::lob_column
