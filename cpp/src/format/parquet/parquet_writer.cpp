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

#include "milvus-storage/format/parquet/parquet_writer.h"

#include <memory>

#include <parquet/properties.h>
#include <boost/variant.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/upload_sizable.h"

namespace milvus_storage::parquet {

static ::parquet::Compression::type convert_compression_type(const std::string& compression) {
  if (compression == "uncompressed") {
    return ::parquet::Compression::UNCOMPRESSED;
  } else if (compression == "snappy") {
    return ::parquet::Compression::SNAPPY;
  } else if (compression == "gzip") {
    return ::parquet::Compression::GZIP;
  } else if (compression == "lz4") {
    return ::parquet::Compression::LZ4;
  } else if (compression == "zstd") {
    return ::parquet::Compression::ZSTD;
  } else if (compression == "brotli") {
    return ::parquet::Compression::BROTLI;
  } else {
    return ::parquet::Compression::ZSTD;
  }
}

static std::shared_ptr<::parquet::WriterProperties> convert_write_properties(
    const milvus_storage::api::Properties& properties) {
  ::parquet::WriterProperties::Builder builder;

  bool enc_enable = api::GetValueNoError<bool>(properties, PROPERTY_WRITER_ENC_ENABLE);
  if (enc_enable) {
    std::string enc_key = api::GetValueNoError<std::string>(properties, PROPERTY_WRITER_ENC_KEY);
    std::string enc_meta = api::GetValueNoError<std::string>(properties, PROPERTY_WRITER_ENC_META);
    std::string enc_algorithm = api::GetValueNoError<std::string>(properties, PROPERTY_WRITER_ENC_ALGORITHM);

    // create builder with key
    ::parquet::FileEncryptionProperties::Builder file_encryption_builder(enc_key);
    // set metadata
    file_encryption_builder.footer_key_metadata(enc_meta);

    // set algorithm
    if (enc_algorithm == ENCRYPTION_ALGORITHM_AES_GCM_V1) {
      file_encryption_builder.algorithm(::parquet::ParquetCipher::AES_GCM_V1);
    } else if (enc_algorithm == ENCRYPTION_ALGORITHM_AES_GCM_CTR_V1) {
      file_encryption_builder.algorithm(::parquet::ParquetCipher::AES_GCM_CTR_V1);
    } else {
      // impossible case
      assert(false);
    }

    builder.encryption(file_encryption_builder.build());
  }

  // Set compression
  auto compression = milvus_storage::api::GetValueNoError<std::string>(properties, PROPERTY_WRITER_COMPRESSION);
  builder.compression(convert_compression_type(compression));

  auto compression_level = milvus_storage::api::GetValueNoError<int32_t>(properties, PROPERTY_WRITER_COMPRESSION_LEVEL);
  if (compression_level >= 0) {
    builder.compression_level(compression_level);
  }

  auto enable_dictionary = milvus_storage::api::GetValueNoError<bool>(properties, PROPERTY_WRITER_ENABLE_DICTIONARY);
  if (enable_dictionary) {
    builder.enable_dictionary();
  } else {
    builder.disable_dictionary();
  }

  return builder.build();
}

ParquetFileWriter::ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                     const std::string& file_path,
                                     const milvus_storage::StorageConfig& storage_config,
                                     std::shared_ptr<::parquet::WriterProperties> writer_props)
    : schema_(std::move(schema)),
      fs_(std::move(fs)),
      file_path_(file_path),
      storage_config_(storage_config),
      sink_(nullptr),
      writer_(nullptr),
      cached_size_(0),
      cached_batches_(),
      cached_batch_sizes_(),
      written_rows_(0) {
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

  for (int i = 0; i < schema_->num_fields(); ++i) {
    auto field = schema_->field(i);
    switch (field->type()->id()) {
      case arrow::Type::FIXED_SIZE_BINARY:
      case arrow::Type::BINARY:
        // Disable statistics for vector columns
        builder.disable_statistics(field->name());
        break;
      default:
        // TODO: truncate statistics for long varible length columns when arrow support it.
        // See: https://github.com/apache/arrow/issues/36139
        break;
    }
  }

  writer_props_ = builder.build();
}

arrow::Result<std::unique_ptr<ParquetFileWriter>> ParquetFileWriter::Make(
    std::shared_ptr<arrow::fs::FileSystem> fs,
    std::shared_ptr<arrow::Schema> schema,
    const std::string& file_path,
    const milvus_storage::api::Properties& properties) {
  ARROW_ASSIGN_OR_RAISE(auto part_size,
                        milvus_storage::api::GetValue<int64_t>(properties, PROPERTY_FS_MULTI_PART_UPLOAD_SIZE));
  return ParquetFileWriter::Make(schema, fs, file_path, milvus_storage::StorageConfig{part_size},
                                 std::move(convert_write_properties(properties)));
}

arrow::Result<std::unique_ptr<ParquetFileWriter>> ParquetFileWriter::Make(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const std::string& file_path,
    const milvus_storage::StorageConfig& storage_config,
    std::shared_ptr<::parquet::WriterProperties> writer_props) {
  auto writer =
      std::unique_ptr<ParquetFileWriter>(new ParquetFileWriter(schema, fs, file_path, storage_config, writer_props));
  ARROW_RETURN_NOT_OK(writer->init());
  return writer;
}

arrow::Status ParquetFileWriter::init() {
  if (!fs_) {
    return arrow::Status::Invalid("Invalid file system for parquet file writer");
  }

  // Although the DIR is created in `column_group_writer`,
  // the current logic cannot be removed. It is still dependent
  // by `packed/`.
  if (IsLocalFileSystem(fs_)) {
    boost::filesystem::path dir_path(file_path_);
    auto parent_dir_path = dir_path.parent_path();
    auto create_dir_result = fs_->CreateDir(parent_dir_path.string());
    if (!create_dir_result.ok()) {
      return arrow::Status::IOError("Failed to create directory: " + create_dir_result.ToString());
    }
  }

  // Try OpenOutputStreamWithUploadSize first, fall back to normal OpenOutputStream if not supported
  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> sink_result;
  auto upload_size_fs = std::dynamic_pointer_cast<UploadSizable>(fs_);
  if (upload_size_fs) {
    sink_result = upload_size_fs->OpenOutputStreamWithUploadSize(file_path_, nullptr, storage_config_.part_size);
    // If not supported, fall back to normal OpenOutputStream
    if (!sink_result.ok() && sink_result.status().code() == arrow::StatusCode::NotImplemented) {
      sink_result = fs_->OpenOutputStream(file_path_);
    }
  } else {
    // Not an UploadSizable filesystem, use normal OpenOutputStream
    sink_result = fs_->OpenOutputStream(file_path_);
  }

  if (!sink_result.ok()) {
    return arrow::Status::IOError("Failed to open output stream: " + sink_result.status().ToString());
  }
  sink_ = std::move(sink_result).ValueOrDie();

  auto writer_result = ::parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(), sink_, writer_props_);
  if (!writer_result.ok()) {
    return arrow::Status::IOError("Failed to create parquet writer: " + writer_result.status().ToString());
  }
  writer_ = std::move(writer_result).ValueOrDie();
  kv_metadata_ = std::make_shared<arrow::KeyValueMetadata>();
  return arrow::Status::OK();
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
  // Add row group metadata after writing
  row_group_metadata_.Add(milvus_storage::RowGroupMetadata(row_group_size, num_rows, written_rows_));
  written_rows_ += num_rows;
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

arrow::Result<api::ColumnGroupFile> ParquetFileWriter::Close() {
  if (closed_ || !writer_) {
    return arrow::Status::Invalid("Current writer is closed or writer is not initialized. file_path=" + file_path_);
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

  ARROW_RETURN_NOT_OK(AppendKVMetadata(milvus_storage::ROW_GROUP_META_KEY, row_group_metadata_.Serialize()));
  ARROW_RETURN_NOT_OK(AppendKVMetadata(milvus_storage::STORAGE_VERSION_KEY, "1.0.0"));
  ARROW_RETURN_NOT_OK(writer_->AddKeyValueMetadata(kv_metadata_));
  ARROW_RETURN_NOT_OK(writer_->Close());
  ARROW_RETURN_NOT_OK(sink_->Flush());
  ARROW_RETURN_NOT_OK(sink_->Close());

  closed_ = true;
  return api::ColumnGroupFile{
      .path = file_path_,
      .start_index = 0,
      .end_index = written_rows_,
      .metadata = {},
  };
}

}  // namespace milvus_storage::parquet
