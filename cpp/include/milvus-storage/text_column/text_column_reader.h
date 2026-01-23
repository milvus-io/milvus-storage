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

#include "milvus-storage/text_column/lob_reference.h"
#include "milvus-storage/text_column/text_column_manager.h"

namespace milvus_storage::text_column {

// encoded reference with size information
// needed because inline text is variable length
struct EncodedRef {
  const uint8_t* data;
  size_t size;
};

// reader for a single TEXT column
// resolves encoded references and reads text data from LOB files
//
// the reader handles both inline and LOB references:
//   - inline (flag=0x00): text is directly in the reference, variable length
//   - LOB (flag=0x01): text is in a Vortex LOB file, fixed 44 bytes
//
// for batch reads, the reader optimizes by:
//   1. grouping references by file_id_str (UUID string)
//   2. using Vortex take() API for efficient random access
//   3. caching open file handles
//
// thread safety: not thread-safe, use one reader per thread
class TextColumnReader {
  public:
  virtual ~TextColumnReader() = default;

  // read a single text from an encoded reference
  // handles both inline (variable length) and LOB (44 bytes) references
  virtual arrow::Result<std::string> ReadText(const uint8_t* encoded_ref, size_t ref_size) = 0;

  // read multiple texts from encoded references
  // this is more efficient than calling ReadText() repeatedly
  // as it groups reads by file_id and uses batch I/O
  virtual arrow::Result<std::vector<std::string>> ReadBatch(const std::vector<EncodedRef>& encoded_refs) = 0;

  // read texts from a BinaryArray of encoded references
  // each element can be either inline (variable) or LOB (44 bytes)
  // returns a StringArray with the decoded texts
  virtual arrow::Result<std::shared_ptr<arrow::StringArray>> ReadArrowArray(
      const std::shared_ptr<arrow::BinaryArray>& encoded_refs) = 0;

  // take texts by row indices from a specific LOB file
  // this is useful when you know the file_id_str (UUID string) and row offsets
  // returns texts in the order of row_indices
  virtual arrow::Result<std::vector<std::string>> Take(const std::string& file_id_str,
                                                       const std::vector<int32_t>& row_offsets) = 0;

  // close the reader and release resources
  // this closes any cached file handles
  virtual arrow::Status Close() = 0;

  // check if the reader is closed
  virtual bool IsClosed() const = 0;

  // clear the file handle cache
  // useful to release memory without closing the reader
  virtual void ClearCache() = 0;

  protected:
  TextColumnReader() = default;
};

// factory function to create TextColumnReader
arrow::Result<std::unique_ptr<TextColumnReader>> CreateTextColumnReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                        const TextColumnConfig& config);

}  // namespace milvus_storage::text_column
