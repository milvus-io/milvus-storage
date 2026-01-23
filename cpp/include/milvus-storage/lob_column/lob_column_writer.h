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
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "milvus-storage/lob_column/lob_column_manager.h"

namespace milvus_storage::lob_column {

// statistics for LOB column writer
struct LobColumnWriterStats {
  int64_t total_entries = 0;      // total number of entries written
  int64_t inline_entries = 0;     // number of entries stored inline
  int64_t lob_entries = 0;        // number of entries stored as LOB
  int64_t total_bytes = 0;        // total bytes of data
  int64_t lob_files_created = 0;  // number of LOB files created
};

// result for a single LOB file created by LobColumnWriter
struct LobFileResult {
  std::string path;         // file path
  int64_t total_rows;       // total number of rows in this file
  int64_t valid_rows;       // number of valid (non-deleted) rows
  int64_t file_size_bytes;  // size of the file in bytes
};

// writer for a single LOB column (TEXT or BINARY)
// handles the decision of inline vs LOB storage and generates LOBReferences
//
// the writer outputs encoded references (24 bytes each for LOB, variable for inline)
// that should be stored in the segment's reference binlog (parquet format)
//
// lifecycle:
//   1. create writer via LobColumnManager::CreateWriter()
//   2. call WriteData()/WriteText() or WriteBatchData()/WriteBatch() to write data
//   3. call Flush() periodically if needed
//   4. call Close() to finalize and close LOB files
//   5. use the returned references to write reference binlog
//
// thread safety: not thread-safe, use one writer per thread
class LobColumnWriter {
  public:
  virtual ~LobColumnWriter() = default;

  // --- generic data interface (works for both text and binary) ---

  // write a single data entry and return the encoded reference
  // the reference is either:
  //   - inline data (flag=0x00): data stored directly, size = 1 + data_size
  //   - LOB reference (flag=0x01): pointer to data in a Vortex LOB file, fixed 24 bytes
  virtual arrow::Result<std::vector<uint8_t>> WriteData(const uint8_t* data, size_t data_size) = 0;

  // write multiple data entries and return encoded references
  // more efficient than calling WriteData() repeatedly as it can batch writes
  virtual arrow::Result<std::vector<std::vector<uint8_t>>> WriteBatchData(
      const std::vector<std::pair<const uint8_t*, size_t>>& items) = 0;

  // write data from an Arrow BinaryArray
  // returns encoded references as a BinaryArray
  virtual arrow::Result<std::shared_ptr<arrow::BinaryArray>> WriteArrowArray(
      const std::shared_ptr<arrow::BinaryArray>& data) = 0;

  // --- text convenience wrappers (non-virtual, delegate to generic interface) ---

  arrow::Result<std::vector<uint8_t>> WriteText(const std::string& text) {
    return WriteData(reinterpret_cast<const uint8_t*>(text.data()), text.size());
  }

  arrow::Result<std::vector<std::vector<uint8_t>>> WriteBatch(const std::vector<std::string>& texts) {
    std::vector<std::pair<const uint8_t*, size_t>> items;
    items.reserve(texts.size());
    for (const auto& t : texts) {
      items.emplace_back(reinterpret_cast<const uint8_t*>(t.data()), t.size());
    }
    return WriteBatchData(items);
  }

  // --- lifecycle methods ---

  // flush buffered data to storage (does not close the current LOB file)
  virtual arrow::Status Flush() = 0;

  // close the writer and finalize all LOB files
  // after calling this, the writer cannot be used anymore
  // returns the list of LOB file results with metadata
  virtual arrow::Result<std::vector<LobFileResult>> Close() = 0;

  // abort the writer and discard all written data
  // this will delete any LOB files that were created
  virtual arrow::Status Abort() = 0;

  // get the number of rows written so far
  virtual int64_t WrittenRows() const = 0;

  // get writer statistics
  virtual LobColumnWriterStats GetStats() const = 0;

  // check if the writer is closed
  virtual bool IsClosed() const = 0;

  protected:
  LobColumnWriter() = default;
};

// factory function to create LobColumnWriter
arrow::Result<std::unique_ptr<LobColumnWriter>> CreateLobColumnWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                      const LobColumnConfig& config);

}  // namespace milvus_storage::lob_column
