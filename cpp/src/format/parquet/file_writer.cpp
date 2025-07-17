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

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/format/parquet/file_writer.h"
#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"

#include <parquet/properties.h>
#include <boost/variant.hpp>
#include "boost/filesystem/path.hpp"
#include <boost/filesystem/operations.hpp>

namespace milvus_storage {

ParquetFileWriter::ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                     const std::string& file_path,
                                     const StorageConfig& storage_config,
                                     std::shared_ptr<parquet::WriterProperties> writer_props)
    : schema_(std::move(schema)),
      fs_(std::move(fs)),
      file_path_(file_path),
      storage_config_(storage_config),
      writer_props_(std::move(writer_props)),
      count_(0),
      cached_size_(0) {}

Status ParquetFileWriter::Init() {
  boost::filesystem::path dir_path(file_path_);
  if (!boost::filesystem::exists(dir_path.parent_path())) {
    boost::filesystem::create_directories(dir_path.parent_path());
  }
  if (!fs_) {
    return Status::InvalidArgument("Invalid file system for parquet file writer");
  }
  auto s3fs = std::dynamic_pointer_cast<MultiPartUploadS3FS>(fs_);
  std::shared_ptr<arrow::io::OutputStream> sink;
  if (storage_config_.part_size > 0 && s3fs) {
    // azure does not support custom part upload size output stream
    ASSIGN_OR_RETURN_ARROW_NOT_OK(sink, s3fs->OpenOutputStreamWithUploadSize(file_path_, storage_config_.part_size));
  } else {
    ASSIGN_OR_RETURN_ARROW_NOT_OK(sink, fs_->OpenOutputStream(file_path_));
  }

  ASSIGN_OR_RETURN_ARROW_NOT_OK(
      auto writer, parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(), sink, writer_props_));

  writer_ = std::move(writer);
  kv_metadata_ = std::make_shared<arrow::KeyValueMetadata>();
  return Status::OK();
}

Status ParquetFileWriter::Write(const arrow::RecordBatch& record) {
  RETURN_ARROW_NOT_OK(writer_->WriteRecordBatch(record));
  count_ += record.num_rows();
  return Status::OK();
}

Status ParquetFileWriter::WriteTable(const arrow::Table& table) {
  RETURN_ARROW_NOT_OK(writer_->WriteTable(table));
  count_ += table.num_rows();
  return Status::OK();
}

Status ParquetFileWriter::WriteRecordBatches(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
                                             const std::vector<size_t>& batch_memory_sizes) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> current_batches = cached_batches_;
  size_t rg_size = cached_size_;

  for (size_t i = 0; i < batches.size(); ++i) {
    auto batch = batches[i];
    size_t batch_size = batch_memory_sizes[i];
    int64_t total_rows = batch->num_rows();
    double avg_row_size = static_cast<double>(batch_size) / total_rows;
    int64_t offset = 0;

    while (offset < total_rows) {
      // Check if current row group is already full
      if (rg_size >= DEFAULT_MAX_ROW_GROUP_SIZE) {
        RETURN_ARROW_NOT_OK(WriteRowGroup(current_batches, rg_size));
        current_batches.clear();
        rg_size = 0;
      }

      size_t remain_size = 0;
      if (rg_size < DEFAULT_MAX_ROW_GROUP_SIZE) {
        remain_size = DEFAULT_MAX_ROW_GROUP_SIZE - rg_size;
      }

      int64_t max_rows = static_cast<int64_t>(remain_size / avg_row_size);
      if (max_rows <= 0) {
        max_rows = 1;
      }
      int64_t slice_len = std::min(max_rows, total_rows - offset);
      auto slice = batch->Slice(offset, slice_len);
      size_t slice_size = avg_row_size * slice_len;
      current_batches.push_back(slice);
      rg_size += slice_size;
      offset += slice_len;
    }
  }
  cached_batches_ = current_batches;
  cached_size_ = rg_size;
  return Status::OK();
}

Status ParquetFileWriter::WriteRowGroup(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batch,
                                        size_t row_group_size) {
  int64_t row_offset = count_;

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto table, arrow::Table::FromRecordBatches(batch));
  RETURN_ARROW_NOT_OK(writer_->WriteTable(*table));

  // Add row group metadata after writing
  row_group_metadata_.Add(RowGroupMetadata(row_group_size, table->num_rows(), row_offset));
  count_ += table->num_rows();
  return Status::OK();
}

int64_t ParquetFileWriter::count() { return count_; }

void ParquetFileWriter::AppendKVMetadata(const std::string& key, const std::string& value) {
  kv_metadata_->Append(key, value);
}

Status ParquetFileWriter::Close() {
  // Flush any remaining cached batches before closing
  if (!cached_batches_.empty()) {
    RETURN_ARROW_NOT_OK(WriteRowGroup(cached_batches_, cached_size_));
    cached_batches_.clear();
    cached_size_ = 0;
  }

  AppendKVMetadata(ROW_GROUP_META_KEY, row_group_metadata_.Serialize());
  AppendKVMetadata(STORAGE_VERSION_KEY, "1.0.0");
  RETURN_ARROW_NOT_OK(writer_->AddKeyValueMetadata(kv_metadata_));
  RETURN_ARROW_NOT_OK(writer_->Close());
  return Status::OK();
}
}  // namespace milvus_storage
