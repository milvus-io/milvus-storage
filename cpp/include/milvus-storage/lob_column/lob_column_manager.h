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

namespace milvus_storage::lob_column {

class LobColumnWriter;
class LobColumnReader;

// data type stored in the LOB column
enum class LobDataType {
  kText,    // UTF-8 text
  kBinary,  // arbitrary bytes
};

// configuration for a single LOB column (TEXT or BINARY)
struct LobColumnConfig {
  // data type determines Arrow type and field name in Vortex files
  LobDataType data_type = LobDataType::kText;

  // lob base path: {partition_path}/lobs/{field_id}
  // Go layer constructs the full path and passes it to C++
  std::string lob_base_path;

  // field id of the LOB column
  int64_t field_id = 0;

  // inline threshold in bytes (data smaller than this is stored inline)
  // inline data is variable length, no maximum limit
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

// manager for a single LOB column (TEXT or BINARY)
// each LOB field in a collection should have its own manager instance
//
// usage:
//   auto manager = LobColumnManager::Create(fs, config);
//   auto writer = manager->CreateWriter();
//   auto refs = writer->WriteBatch(texts);  // or WriteData() for binary
//   writer->Close();
//
//   auto reader = manager->CreateReader();
//   auto texts = reader->ReadBatch(encoded_refs);  // or ReadData() for binary
class LobColumnManager {
  public:
  virtual ~LobColumnManager() = default;

  // create a new manager for the specified LOB column
  static arrow::Result<std::unique_ptr<LobColumnManager>> Create(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                 const LobColumnConfig& config);

  // create a writer for this LOB column
  // the writer manages LOB file lifecycle and generates LOBReferences
  virtual arrow::Result<std::unique_ptr<LobColumnWriter>> CreateWriter() = 0;

  // create a reader for this LOB column
  // the reader can resolve LOBReferences and read data from LOB files
  virtual arrow::Result<std::unique_ptr<LobColumnReader>> CreateReader() = 0;

  // get the configuration
  virtual const LobColumnConfig& GetConfig() const = 0;

  // get the filesystem
  virtual std::shared_ptr<arrow::fs::FileSystem> GetFileSystem() const = 0;

  protected:
  LobColumnManager() = default;
};

}  // namespace milvus_storage::lob_column
