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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/text_column/text_column_manager.h"
#include "milvus-storage/transaction/transaction.h"

namespace milvus_storage::segment {

// configuration for SegmentWriter
struct SegmentWriterConfig {
  // LOB base path - used for LOB files at partition level
  // e.g., "/data/partitions/lobs"
  // LOB files: {lob_base_path}/{field_id}/_data/{uuid}.vx
  std::string lob_base_path;

  // segment base path - this is where manifest and parquet data files are stored
  // e.g., "/data/partitions/segments/seg-001"
  // manifest: {segment_path}/_metadata/manifest-{version}.avro
  // data:     {segment_path}/_data/cg{N}_{uuid}.parquet
  std::string segment_path;

  // TEXT column field IDs and their configurations
  // Note: lob_base_path in TextColumnConfig should be {lob_base_path}/{field_id}
  // The caller (Go layer) is responsible for constructing the full path
  std::map<int64_t, text_column::TextColumnConfig> text_columns;

  // storage properties (filesystem config, compression, column group policy, etc.)
  // must include PROPERTY_WRITER_POLICY and PROPERTY_FORMAT for ColumnGroupPolicy
  // see writer.h for available policy options:
  //   - LOON_COLUMN_GROUP_POLICY_SINGLE: all columns in one group
  //   - LOON_COLUMN_GROUP_POLICY_SCHEMA_BASED: group by column name patterns
  //   - LOON_COLUMN_GROUP_POLICY_SIZE_BASED: group by column sizes
  api::Properties properties;

  // transaction related configuration
  // read_version: version to read from (-1 = LATEST, fetch greatest version)
  int64_t read_version = -1;

  // retry_limit: maximum number of retry attempts on commit conflicts
  uint32_t retry_limit = 3;
};

// result returned by SegmentWriter::Close()
struct SegmentWriterResult {
  // full path to the committed manifest file
  // format: {segment_path}/_metadata/manifest-{version}.avro
  std::string manifest_path;

  // committed version number
  int64_t committed_version = 0;

  // total number of rows written
  int64_t rows_written = 0;
};

// statistics for segment writer
struct SegmentWriterStats {
  int64_t total_rows = 0;             // total number of rows written
  int64_t total_bytes = 0;            // total bytes written (approximate)
  int64_t parquet_files_created = 0;  // number of parquet files created
  int64_t lob_files_created = 0;      // number of LOB files created
};

// SegmentWriter writes complete row data to storage with transaction support
// - TEXT columns are stored in Vortex LOB files (via TextColumnManager)
// - Regular columns are stored in Parquet files (via PackedRecordBatchWriter)
// - Automatically manages Transaction for manifest versioning
//
// Usage:
//   auto writer = SegmentWriter::Create(fs, schema, config);
//   writer->Write(batch1);
//   writer->Write(batch2);
//   auto result = writer->Close();  // commits transaction, writes manifest
//   // result.manifest_path = "segment_path/_metadata/manifest-2.avro"
//   // result.committed_version = 2
//
// File layout after Close():
//   partition_path/
//   ├── segments/
//   │   └── {segment_id}/              <- segment_path points here
//   │       ├── _metadata/
//   │       │   └── manifest-{version}.avro    <- manifest at segment level
//   │       └── _data/
//   │           └── cg{N}_{uuid}.parquet       <- parquet files for regular columns
//   └── lobs/
//       └── {field_id}/                <- lob_base_path points here
//           └── _data/
//               └── {uuid}.vx          <- LOB files for TEXT columns (partition level)
//
// Thread safety: not thread-safe, use one writer per thread
class SegmentWriter {
  public:
  virtual ~SegmentWriter() = default;

  // create a SegmentWriter
  // schema: the original schema with TEXT columns as utf8() type
  // the writer will convert TEXT columns to binary() for storage
  // internally opens a Transaction for manifest management
  static arrow::Result<std::unique_ptr<SegmentWriter>> Create(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                              const std::shared_ptr<arrow::Schema>& schema,
                                                              const SegmentWriterConfig& config);

  // write a RecordBatch
  // TEXT columns are automatically extracted and written to LOB files
  // the LOBReferences are stored in the Parquet files
  virtual arrow::Status Write(const std::shared_ptr<arrow::RecordBatch>& batch) = 0;

  // flush all buffered data to storage
  // this does not close the writer or commit the transaction
  virtual arrow::Status Flush() = 0;

  // close the writer and commit the transaction
  // returns SegmentWriterResult containing:
  //   - manifest_path: full path to the committed manifest file
  //   - committed_version: the new version number
  //   - rows_written: total rows written
  // the manifest contains:
  //   - column groups for regular columns (Parquet based on policy)
  //   - Vortex column groups for TEXT LOB files
  virtual arrow::Result<SegmentWriterResult> Close() = 0;

  // abort the writer and clean up all created files
  // this will delete any LOB files and Parquet files that were created
  // does NOT commit the transaction
  virtual arrow::Status Abort() = 0;

  // get the number of rows written so far
  virtual int64_t WrittenRows() const = 0;

  // get writer statistics
  virtual SegmentWriterStats GetStats() const = 0;

  // get the storage schema (TEXT columns converted to BINARY)
  virtual std::shared_ptr<arrow::Schema> GetStorageSchema() const = 0;

  // get the original schema
  virtual std::shared_ptr<arrow::Schema> GetOriginalSchema() const = 0;

  // check if the writer is closed
  virtual bool IsClosed() const = 0;

  // get the read version of the underlying transaction
  virtual int64_t GetReadVersion() const = 0;

  protected:
  SegmentWriter() = default;
};

}  // namespace milvus_storage::segment
