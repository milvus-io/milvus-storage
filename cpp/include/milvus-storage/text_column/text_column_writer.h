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

#include "milvus-storage/text_column/text_column_manager.h"

namespace milvus_storage::text_column {

// statistics for text column writer
struct TextColumnWriterStats {
  int64_t total_texts = 0;        // total number of texts written
  int64_t inline_texts = 0;       // number of texts stored inline
  int64_t lob_texts = 0;          // number of texts stored as LOB
  int64_t total_bytes = 0;        // total bytes of text data
  int64_t lob_files_created = 0;  // number of LOB files created
};

// result for a single LOB file created by TextColumnWriter
struct LobFileResult {
  std::string path;         // file path
  int64_t total_rows;       // total number of rows in this file
  int64_t valid_rows;       // number of valid (non-deleted) rows
  int64_t file_size_bytes;  // size of the file in bytes
};

// writer for a single TEXT column
// handles the decision of inline vs LOB storage and generates LOBReferences
//
// the writer outputs encoded references (44 bytes each for LOB, variable for inline)
// that should be stored in the segment's reference binlog (parquet format)
//
// lifecycle:
//   1. create writer via TextColumnManager::CreateWriter()
//   2. call WriteText() or WriteBatch() to write texts
//   3. call Flush() periodically if needed
//   4. call Close() to finalize and close LOB files
//   5. use the returned references to write reference binlog
//
// thread safety: not thread-safe, use one writer per thread
class TextColumnWriter {
  public:
  virtual ~TextColumnWriter() = default;

  // write a single text and return the encoded reference
  // the reference is either:
  //   - inline text (flag=0x00): text stored directly, size = 1 + text.length()
  //   - LOB reference (flag=0x01): pointer to text in a Vortex LOB file, fixed 44 bytes
  virtual arrow::Result<std::vector<uint8_t>> WriteText(const std::string& text) = 0;

  // write multiple texts and return encoded references
  // this is more efficient than calling WriteText() repeatedly
  // as it can batch writes to the LOB file
  virtual arrow::Result<std::vector<std::vector<uint8_t>>> WriteBatch(const std::vector<std::string>& texts) = 0;

  // write texts from an Arrow StringArray
  // returns encoded references as a BinaryArray (each element is 44 bytes for LOB, variable for inline)
  virtual arrow::Result<std::shared_ptr<arrow::BinaryArray>> WriteArrowArray(
      const std::shared_ptr<arrow::StringArray>& texts) = 0;

  // flush buffered data to storage
  // this does not close the current LOB file
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
  virtual TextColumnWriterStats GetStats() const = 0;

  // check if the writer is closed
  virtual bool IsClosed() const = 0;

  protected:
  TextColumnWriter() = default;
};

// factory function to create TextColumnWriter
arrow::Result<std::unique_ptr<TextColumnWriter>> CreateTextColumnWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                        const TextColumnConfig& config);

}  // namespace milvus_storage::text_column
