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
#include <arrow/status.h>
#include <arrow/result.h>

#include <cstdint>
#include <memory>
#include <string>

#include "milvus-storage/properties.h"

namespace milvus_storage::text_column {

class TextColumnWriter;
class TextColumnReader;

// configuration for a single TEXT column
struct TextColumnConfig {
  // lob base path: {partition_path}/lobs/{field_id}
  // Go layer constructs the full path and passes it to C++
  std::string lob_base_path;

  // field id of the TEXT column
  int64_t field_id = 0;

  // inline threshold in bytes (texts smaller than this are stored inline)
  // inline text is variable length, no maximum limit
  // default: 64KB
  size_t inline_threshold = 64 * 1024;

  // maximum bytes per LOB file before rolling to a new file
  // default: 512MB
  size_t max_lob_file_bytes = 512 * 1024 * 1024;

  // flush threshold: when buffered LOB data exceeds this size, flush to file
  // this prevents memory buildup during batch writes
  // default: 64MB
  size_t flush_threshold_bytes = 64 * 1024 * 1024;

  // storage properties (filesystem config, compression, etc.)
  api::Properties properties;
};

// manager for a single TEXT column
// each TEXT field in a collection should have its own manager instance
//
// usage:
//   auto manager = TextColumnManager::Create(fs, config);
//   auto writer = manager->CreateWriter();
//   auto refs = writer->WriteBatch(texts);
//   writer->Close();
//
//   auto reader = manager->CreateReader();
//   auto texts = reader->ReadBatch(encoded_refs);
class TextColumnManager {
  public:
  virtual ~TextColumnManager() = default;

  // create a new manager for the specified TEXT column
  static arrow::Result<std::unique_ptr<TextColumnManager>> Create(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                  const TextColumnConfig& config);

  // create a writer for this TEXT column
  // the writer manages LOB file lifecycle and generates LOBReferences
  virtual arrow::Result<std::unique_ptr<TextColumnWriter>> CreateWriter() = 0;

  // create a reader for this TEXT column
  // the reader can resolve LOBReferences and read text data from LOB files
  virtual arrow::Result<std::unique_ptr<TextColumnReader>> CreateReader() = 0;

  // get the configuration
  virtual const TextColumnConfig& GetConfig() const = 0;

  // get the filesystem
  virtual std::shared_ptr<arrow::fs::FileSystem> GetFileSystem() const = 0;

  protected:
  TextColumnManager() = default;
};

}  // namespace milvus_storage::text_column
