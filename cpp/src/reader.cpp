// Copyright 2023 Zilliz
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

#include "milvus-storage/reader.h"

#include <arrow/array.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/iterator.h>
#include <parquet/properties.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/status.h"
#include "milvus-storage/packed/reader.h"

namespace milvus_storage::api {

// ==================== ColumnGroupReader Implementation ====================

ColumnGroupReader::ColumnGroupReader(const std::shared_ptr<arrow::fs::FileSystem>& fs, const std::shared_ptr<ColumnGroup>& column_group)
    : fs_(fs), column_group_(column_group) {
  // Initialize column group reader with filesystem and column group metadata
  // The column group contains information about which columns are stored together
  // and their physical storage location (file paths, chunk boundaries, etc.)
}

arrow::Result<std::vector<int64_t>> ColumnGroupReader::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  // TODO: Implement row-to-chunk mapping for column groups
  // This should map global row indices to local chunk indices within this column group
  // considering the column group's row boundaries and chunk organization
  return arrow::Status::NotImplemented("Row-to-chunk mapping not yet implemented for ColumnGroupReader");
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ColumnGroupReader::get_chunk(const int64_t chunk_index) {
  // TODO: Implement single chunk reading for column groups
  // This should read a specific chunk (row group) from the column group's storage files
  // and return the data as an Arrow RecordBatch
  return arrow::Status::NotImplemented("Single chunk reading not yet implemented for ColumnGroupReader");
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ColumnGroupReader::get_chunks(const std::vector<int64_t>& chunk_indices, 
                                                                                               int64_t parallelism) {
  // TODO: Implement multi-chunk reading for column groups with parallel support
  // This should efficiently read multiple chunks, potentially in parallel,
  // and return them as a vector of RecordBatches
  return arrow::Status::NotImplemented("Multi-chunk reading not yet implemented for ColumnGroupReader");
}

// ==================== Reader Implementation ====================

Reader::Reader(const std::shared_ptr<arrow::fs::FileSystem>& fs, 
               const std::shared_ptr<Manifest>& manifest, 
               const std::shared_ptr<arrow::Schema>& schema,
               const std::shared_ptr<std::vector<std::string>>& needed_columns,
               const ReadProperties& properties)
    : fs_(fs), manifest_(manifest), schema_(schema), properties_(properties) {
  
  // Initialize the list of columns to read from the dataset
  if (needed_columns != nullptr) {
    needed_columns_ = *needed_columns;
  } else {
    // If no specific columns requested, read all columns from the schema
    needed_columns_.clear();
    for (int i = 0; i < schema_->num_fields(); ++i) {
      needed_columns_.push_back(schema_->field(i)->name());
    }
  }

  // Determine which column groups are needed based on the requested columns
  // This optimization allows reading only the column groups that contain
  // the requested columns, reducing I/O and improving performance
  auto visited_column_groups = std::set<int64_t>();
  for (const auto& column_name : needed_columns_) {
    auto column_group = manifest_->get_column_group(column_name);
    if (column_group != nullptr && visited_column_groups.find(column_group->id) == visited_column_groups.end()) {
      needed_column_groups_.push_back(column_group);
      visited_column_groups.insert(column_group->id);
    }
  }
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> Reader::scan(const std::string& predicate, 
                                                                                         int64_t batch_size, 
                                                                                         int64_t buffer_size) {
  // Collect file paths from all column groups in the manifest
  // This provides the PackedRecordBatchReader with all necessary data files
  auto paths = std::vector<std::string>();
  for (const auto& column_group : manifest_->get_column_groups()) {
    paths.push_back(column_group->path);
  }

  // Create and return a PackedRecordBatchReader for sequential scanning
  // The packed reader handles coordination across multiple column group files
  // and provides efficient streaming access to the entire dataset
  
  // TODO: Implement predicate pushdown for server-side filtering
  // TODO: Implement batch_size parameter to control memory usage per batch
  // TODO: Implement column projection to read only needed_columns_
  // TODO: Apply encryption properties from properties_ if configured
  
  return std::make_shared<PackedRecordBatchReader>(fs_, paths, schema_, buffer_size);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> Reader::take(const std::vector<int64_t>& row_indices, 
                                                                int64_t parallelism) {
  // TODO: Implement random access reading by row indices
  // This should:
  // 1. Map row indices to their corresponding column groups and chunks
  // 2. Group indices by column group for efficient batch reading
  // 3. Read the required chunks from each column group (potentially in parallel)
  // 4. Extract the specific rows from each chunk
  // 5. Reconstruct the final RecordBatch maintaining original row order
  // 6. Handle cross-column-group row assembly for complete records
  
  return arrow::Status::NotImplemented("Random access by row indices not yet implemented");
}

Reader::~Reader() {
  // Destructor performs cleanup of resources
  // Column group readers and cached metadata are automatically cleaned up
  // by their respective destructors when the Reader object is destroyed
}

}  // namespace milvus_storage::api