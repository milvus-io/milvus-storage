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
#include "milvus-storage/format/parquet/common.h"
#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"

#include <parquet/properties.h>
#include <boost/variant.hpp>
#include "boost/filesystem/path.hpp"
#include <boost/filesystem/operations.hpp>
#include <memory>

namespace milvus_storage::parquet {

ParquetFileWriter::ParquetFileWriter(std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                     std::shared_ptr<arrow::Schema> schema,
                                     const milvus_storage::api::Properties& properties)
    : ParquetFileWriter(schema,
                        fs,
                        column_group->paths[0],
                        milvus_storage::StorageConfig{
                            milvus_storage::api::GetValue(properties, milvus_storage::api::MultiPartUploadSizeKey)},
                        milvus_storage::parquet::convert_write_properties(properties)) {}

ParquetFileWriter::ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                     const std::string& file_path,
                                     const milvus_storage::StorageConfig& storage_config,
                                     std::shared_ptr<::parquet::WriterProperties> writer_props)
    : schema_(std::move(schema)),
      fs_(std::move(fs)),
      file_path_(file_path),
      storage_config_(storage_config),
      count_(0),
      bytes_written_(0),
      cached_size_(0),
      cached_batches_(),
      cached_batch_sizes_() {
  auto builder = ::parquet::WriterProperties::Builder(*writer_props);
  builder.max_row_group_length(
      std::numeric_limits<int64_t>::max());  // no limit on row group size, let the writer handle it
  if (writer_props->file_encryption_properties()) {
    auto deep_copied_decryption = writer_props->file_encryption_properties()->DeepClone();
    builder.encryption(std::move(deep_copied_decryption));
  }
  if (writer_props->default_column_properties().compression() == ::parquet::Compression::UNCOMPRESSED) {
    builder.compression(::parquet::Compression::ZSTD);
    builder.compression_level(3);
  }
  writer_props_ = builder.build();

  if (!fs_) {
    throw std::runtime_error("Invalid file system for parquet file writer");
  }
  bool is_local_fs = fs_->type_name() == "local";
  // create parent dir if not exist only for local file system
  if (is_local_fs) {
    boost::filesystem::path dir_path(file_path_);
    auto create_dir_result = fs_->CreateDir(dir_path.parent_path().string());
    if (!create_dir_result.ok()) {
      throw std::runtime_error("Failed to create directory: " + create_dir_result.ToString());
    }
  }

  std::shared_ptr<arrow::io::OutputStream> sink;
  if (storage_config_.part_size > 0 && fs_->type_name() == MULTI_PART_UPLOAD_S3_FILESYSTEM_NAME) {
    auto s3fs = std::dynamic_pointer_cast<milvus_storage::MultiPartUploadS3FS>(fs_);
    // azure does not support custom part upload size output stream
    auto sink_result = s3fs->OpenOutputStreamWithUploadSize(file_path_, storage_config_.part_size);
    if (!sink_result.ok()) {
      throw std::runtime_error("Failed to open output stream: " + sink_result.status().ToString());
    }
    sink = std::move(sink_result).ValueOrDie();
  } else {
    auto sink_result = fs_->OpenOutputStream(file_path_);
    if (!sink_result.ok()) {
      throw std::runtime_error("Failed to open output stream: " + sink_result.status().ToString());
    }
    sink = std::move(sink_result).ValueOrDie();
  }

  auto writer_result = ::parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(), sink, writer_props_);
  if (!writer_result.ok()) {
    throw std::runtime_error("Failed to create parquet writer: " + writer_result.status().ToString());
  }
  writer_ = std::move(writer_result).ValueOrDie();
  kv_metadata_ = std::make_shared<arrow::KeyValueMetadata>();
}

arrow::Status ParquetFileWriter::Write(const std::shared_ptr<arrow::RecordBatch> record) {
  if (!record) {
    return arrow::Status::OK();
  }
  cached_batches_.push_back(record);
  auto batch_size = milvus_storage::GetRecordBatchMemorySize(record);
  cached_batch_sizes_.push_back(batch_size);
  cached_size_ += batch_size;
  return arrow::Status::OK();
}

arrow::Status ParquetFileWriter::Flush() {
  std::vector<std::shared_ptr<arrow::RecordBatch>> row_group_batches;
  std::vector<size_t> row_group_batch_sizes;
  size_t rg_size = 0;

  for (size_t i = 0; i < cached_batches_.size(); ++i) {
    const auto& batch = cached_batches_[i];
    size_t batch_size = cached_batch_sizes_[i];
    int64_t total_rows = batch->num_rows();
    double avg_row_size = static_cast<double>(batch_size) / total_rows;
    int64_t offset = 0;

    while (offset < total_rows) {
      // Check if current row group is already full
      if (rg_size >= milvus_storage::DEFAULT_MAX_ROW_GROUP_SIZE) {
        ARROW_RETURN_NOT_OK(write_row_group(row_group_batches, rg_size));
        row_group_batches.clear();
        row_group_batch_sizes.clear();
        rg_size = 0;
      }

      size_t remain_size = 0;
      if (rg_size < milvus_storage::DEFAULT_MAX_ROW_GROUP_SIZE) {
        remain_size = milvus_storage::DEFAULT_MAX_ROW_GROUP_SIZE - rg_size;
      }

      auto max_rows = static_cast<int64_t>(remain_size / avg_row_size);
      if (max_rows <= 0) {
        max_rows = 1;
      }
      int64_t slice_len = std::min(max_rows, total_rows - offset);
      auto slice = batch->Slice(offset, slice_len);
      size_t slice_size = avg_row_size * slice_len;
      row_group_batches.emplace_back(slice);
      row_group_batch_sizes.emplace_back(slice_size);
      rg_size += slice_size;
      offset += slice_len;
    }
  }

  // Keep remaining batches for next flush
  cached_batches_ = row_group_batches;
  cached_batch_sizes_ = row_group_batch_sizes;
  cached_size_ = rg_size;

  return arrow::Status::OK();
}

arrow::Status ParquetFileWriter::write_row_group(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batch,
                                                 size_t row_group_size) {
  ARROW_RETURN_NOT_OK(writer_->NewBufferedRowGroup());
  size_t num_rows = 0;
  for (const auto& b : batch) {
    ARROW_RETURN_NOT_OK(writer_->WriteRecordBatch(*b));
    num_rows += b->num_rows();
  }
  for (auto& builder : metadata_builders_) {
    builder.second->Append(batch);
  }
  // Add row group metadata after writing
  row_group_metadata_.Add(milvus_storage::RowGroupMetadata(row_group_size, num_rows, count_));
  count_ += num_rows;
  bytes_written_ += row_group_size;
  return arrow::Status::OK();
}

arrow::Status ParquetFileWriter::AppendKVMetadata(const std::string& key, const std::string& value) {
  kv_metadata_->Append(key, value);
  return arrow::Status::OK();
}

arrow::Status ParquetFileWriter::AddUserMetadata(const std::vector<std::pair<std::string, std::string>>& metadata) {
  for (const auto& [key, value] : metadata) {
    ARROW_RETURN_NOT_OK(AppendKVMetadata(key, value));
  }
  return arrow::Status::OK();
}

arrow::Status ParquetFileWriter::AddMetadataBuilder(const std::string& key, std::unique_ptr<MetadataBuilder> builder) {
  for (const auto& [existing_key, _] : metadata_builders_) {
    if (existing_key == key) {
      return arrow::Status::Invalid("Metadata builder with key ", key, " already exists");
    }
  }
  metadata_builders_.emplace_back(key, std::move(builder));
  return arrow::Status::OK();
}

arrow::Status ParquetFileWriter::Close() {
  if (closed_ || !writer_) {
    return arrow::Status::OK();
  }
  // Flush any pending batches first
  ARROW_RETURN_NOT_OK(Flush());

  // Write any remaining cached batches that are smaller than DEFAULT_MAX_ROW_GROUP_SIZE
  if (!cached_batches_.empty()) {
    ARROW_RETURN_NOT_OK(write_row_group(cached_batches_, cached_size_));
    cached_batches_.clear();
    cached_batch_sizes_.clear();
    cached_size_ = 0;
  }

  for (const auto& [key, builder] : metadata_builders_) {
    ARROW_RETURN_NOT_OK(AppendKVMetadata(key, builder->Finish()));
  }

  ARROW_RETURN_NOT_OK(AppendKVMetadata(milvus_storage::ROW_GROUP_META_KEY, row_group_metadata_.Serialize()));
  ARROW_RETURN_NOT_OK(AppendKVMetadata(milvus_storage::STORAGE_VERSION_KEY, "1.0.0"));
  ARROW_RETURN_NOT_OK(writer_->AddKeyValueMetadata(kv_metadata_));
  ARROW_RETURN_NOT_OK(writer_->Close());
  closed_ = true;
  return arrow::Status::OK();
}

}  // namespace milvus_storage::parquet
