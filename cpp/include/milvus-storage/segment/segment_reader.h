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
#include <arrow/table.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/text_column/text_column_manager.h"

namespace milvus_storage::segment {

// configuration for SegmentReader
struct SegmentReaderConfig {
  // TEXT column field IDs and their configurations
  // used to create TextColumnReader for resolving LOBReferences
  std::map<int64_t, text_column::TextColumnConfig> text_columns;

  // read buffer size for underlying readers
  size_t read_buffer_size = 64 * 1024 * 1024;  // 64MB default

  // storage properties
  api::Properties properties;
};

// SegmentReader reads row data from storage with optional column selection
// - supports extracting specific columns from segment data
// - LOBReferences are automatically resolved to TEXT
// - supports both sequential read and random row access (Take)
//
// usage (read all columns):
//   auto reader = SegmentReader::Open(fs, segment_path, version, schema, {}, config);
//   std::shared_ptr<arrow::RecordBatch> batch;
//   while (reader->ReadNext(&batch).ok() && batch) {
//     // process batch
//   }
//   reader->Close();
//
// usage (read specific columns):
//   std::vector<std::string> columns = {"id", "content", "value"};
//   auto reader = SegmentReader::Open(fs, segment_path, version, schema, columns, config);
//
// usage (random access):
//   auto reader = SegmentReader::Open(fs, segment_path, version, schema, columns, config);
//   std::vector<int64_t> row_indices = {0, 5, 10, 15};
//   auto table = reader->Take(row_indices);
//
// thread safety: not thread-safe, use one reader per thread
class SegmentReader : public arrow::RecordBatchReader {
  public:
  virtual ~SegmentReader() = default;

  // create a SegmentReader from ColumnGroups (directly)
  // column_groups: the ColumnGroups containing column group metadata
  // schema: the original schema with TEXT columns as utf8() type
  // columns: list of column names to extract (empty = all columns)
  static arrow::Result<std::unique_ptr<SegmentReader>> Create(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                              const std::shared_ptr<api::ColumnGroups>& column_groups,
                                                              const std::shared_ptr<arrow::Schema>& schema,
                                                              const std::vector<std::string>& columns,
                                                              const SegmentReaderConfig& config);

  // open a SegmentReader from manifest (reads manifest file first)
  // segment_path: base path where manifest and data files are stored
  // version: manifest version to read (-1 = latest version)
  // schema: the original schema with TEXT columns as utf8() type
  // columns: list of column names to extract (empty = all columns)
  static arrow::Result<std::unique_ptr<SegmentReader>> Open(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                            const std::string& segment_path,
                                                            int64_t version,
                                                            const std::shared_ptr<arrow::Schema>& schema,
                                                            const std::vector<std::string>& columns,
                                                            const SegmentReaderConfig& config);

  // read the next RecordBatch with selected columns
  // LOBReferences are automatically resolved to TEXT
  // returns nullptr when no more data
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override = 0;

  // extract specific rows by indices (random access)
  // row_indices: global row indices to extract (0-based)
  // parallelism: number of parallel I/O threads
  // returns Table containing only selected rows and columns
  virtual arrow::Result<std::shared_ptr<arrow::Table>> Take(const std::vector<int64_t>& row_indices,
                                                            size_t parallelism = 1) = 0;

  // get the schema of extracted columns only
  std::shared_ptr<arrow::Schema> schema() const override = 0;

  // get the original full schema
  virtual std::shared_ptr<arrow::Schema> GetOriginalSchema() const = 0;

  // get the list of extracted column names
  virtual const std::vector<std::string>& GetExtractedColumns() const = 0;

  // get total number of rows read so far
  virtual int64_t GetTotalRows() const = 0;

  // close the reader and release resources
  arrow::Status Close() override = 0;

  // check if the reader is closed
  virtual bool IsClosed() const = 0;

  // get the manifest version being read (if opened from manifest)
  virtual int64_t GetVersion() const = 0;

  protected:
  SegmentReader() = default;
};

}  // namespace milvus_storage::segment
