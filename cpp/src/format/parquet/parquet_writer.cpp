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

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <arrow/io/buffered.h>
#include <arrow/io/memory.h>
#include <parquet/properties.h>
#include <parquet/metadata.h>
#include <boost/variant.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <utility>

#include <fmt/format.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/upload_sizable.h"

namespace milvus_storage::parquet {

namespace properties {

static ::parquet::Compression::type ConvertCompressionType(const std::string& compression) {
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

static bool ContainsColumnName(const std::vector<std::string>& column_names, const std::string& column_name) {
  return std::find(column_names.begin(), column_names.end(), column_name) != column_names.end();
}

static bool ShouldDisableStatistics(const std::shared_ptr<arrow::Field>& field,
                                    const std::vector<std::string>& disabled_stats_columns) {
  if (ContainsColumnName(disabled_stats_columns, field->name())) {
    return true;
  }

  switch (field->type()->id()) {
    case arrow::Type::FIXED_SIZE_BINARY:
    case arrow::Type::BINARY:
      return true;
    default:
      // TODO: truncate statistics for long varible length columns when arrow support it.
      // See: https://github.com/apache/arrow/issues/36139
      return false;
  }
}

static void DisableStatisticsForPartColumns(const std::shared_ptr<arrow::Schema>& schema,
                                            const std::vector<std::string>& disabled_stats_columns,
                                            ::parquet::WriterProperties::Builder* builder) {
  for (int i = 0; i < schema->num_fields(); ++i) {
    auto field = schema->field(i);
    if (ShouldDisableStatistics(field, disabled_stats_columns)) {
      builder->disable_statistics(field->name());
    }
  }
}

static bool ShouldDisableCompression(const std::shared_ptr<arrow::Field>& field,
                                     const std::vector<std::string>& disabled_compression_columns) {
  return ContainsColumnName(disabled_compression_columns, field->name());
}

static void DisableCompressionForPartColumns(const std::shared_ptr<arrow::Schema>& schema,
                                             const std::vector<std::string>& disabled_compression_columns,
                                             ::parquet::WriterProperties::Builder* builder) {
  for (int i = 0; i < schema->num_fields(); ++i) {
    auto field = schema->field(i);
    if (!ShouldDisableCompression(field, disabled_compression_columns)) {
      continue;
    }

    builder->compression(field->name(), ::parquet::Compression::UNCOMPRESSED);
    builder->codec_options(field->name(), std::shared_ptr<::arrow::util::CodecOptions>());
    builder->disable_dictionary(field->name());
    builder->encoding(field->name(), ::parquet::Encoding::PLAIN);
  }
}

static void NormalizeWriterProperties(const std::shared_ptr<arrow::Schema>& schema,
                                      const std::shared_ptr<::parquet::WriterProperties>& writer_props,
                                      const std::vector<std::string>& disabled_compression_columns,
                                      const std::vector<std::string>& disabled_stats_columns,
                                      ::parquet::WriterProperties::Builder* builder) {
  builder->max_row_group_length(
      std::numeric_limits<int64_t>::max());  // no limit on row group size, let the writer handle it
  if (writer_props->file_encryption_properties()) {
    auto deep_copied_decryption = writer_props->file_encryption_properties()->DeepClone();
    builder->encryption(std::move(deep_copied_decryption));
  }
  if (writer_props->default_column_properties().compression() == ::parquet::Compression::UNCOMPRESSED) {
    builder->compression(::parquet::Compression::ZSTD);
    builder->compression_level(3);
  }

  // Fall back to PLAIN once the per-column dictionary reaches 1/16 of the
  // row group size. With parquet's default 1 MB limit and our ~1 MB row
  // groups, high-cardinality columns end up with a dictionary page that
  // mirrors the column data instead of falling back, inflating the file.
  builder->dictionary_pagesize_limit(DEFAULT_MAX_ROW_GROUP_SIZE / 16);

  DisableStatisticsForPartColumns(schema, disabled_stats_columns, builder);
  DisableCompressionForPartColumns(schema, disabled_compression_columns, builder);
}

static std::shared_ptr<::parquet::WriterProperties> NormalizeWriterProperties(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<::parquet::WriterProperties>& writer_props,
    const std::vector<std::string>& disabled_compression_columns = {},
    const std::vector<std::string>& disabled_stats_columns = {}) {
  auto builder = ::parquet::WriterProperties::Builder(*writer_props);
  NormalizeWriterProperties(schema, writer_props, disabled_compression_columns, disabled_stats_columns, &builder);

  return builder.build();
}

static std::shared_ptr<::parquet::WriterProperties> ConvertFromApiProperties(
    const milvus_storage::api::Properties& api_properties, const std::shared_ptr<arrow::Schema>& schema) {
  ::parquet::WriterProperties::Builder builder;

  bool enc_enable = api::GetValueNoError<bool>(api_properties, PROPERTY_WRITER_ENC_ENABLE);
  if (enc_enable) {
    auto enc_key = api::GetValueNoError<std::string>(api_properties, PROPERTY_WRITER_ENC_KEY);
    auto enc_meta = api::GetValueNoError<std::string>(api_properties, PROPERTY_WRITER_ENC_META);
    auto enc_algorithm = api::GetValueNoError<std::string>(api_properties, PROPERTY_WRITER_ENC_ALGORITHM);

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
  auto compression = milvus_storage::api::GetValueNoError<std::string>(api_properties, PROPERTY_WRITER_COMPRESSION);
  builder.compression(ConvertCompressionType(compression));

  auto compression_level =
      milvus_storage::api::GetValueNoError<int32_t>(api_properties, PROPERTY_WRITER_COMPRESSION_LEVEL);
  if (compression_level >= 0) {
    builder.compression_level(compression_level);
  }

  auto enable_dictionary =
      milvus_storage::api::GetValueNoError<bool>(api_properties, PROPERTY_WRITER_ENABLE_DICTIONARY);
  if (enable_dictionary) {
    builder.enable_dictionary();
  } else {
    builder.disable_dictionary();
  }

  auto writer_props = builder.build();
  auto finalized_builder = ::parquet::WriterProperties::Builder(*writer_props);
  auto disabled_compression_columns = milvus_storage::api::GetValueNoError<std::vector<std::string>>(
      api_properties, PROPERTY_WRITER_DISABLE_COMPRESSION_COLUMNS);
  auto disabled_stats_columns = milvus_storage::api::GetValueNoError<std::vector<std::string>>(
      api_properties, PROPERTY_WRITER_DISABLE_STATS_COLUMNS);
  NormalizeWriterProperties(schema, writer_props, disabled_compression_columns, disabled_stats_columns,
                            &finalized_builder);

  return finalized_builder.build();
}

}  // namespace properties

ParquetFileWriter::ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                     const std::string& file_path,
                                     const milvus_storage::StorageConfig& storage_config,
                                     const std::shared_ptr<::parquet::WriterProperties>& writer_props)
    : schema_(std::move(schema)),
      fs_(std::move(fs)),
      file_path_(file_path),
      storage_config_(storage_config),
      sink_(nullptr),
      writer_(nullptr),

      cached_batches_(),
      cached_batch_sizes_() {
  writer_props_ = writer_props;
}

arrow::Result<std::unique_ptr<ParquetFileWriter>> ParquetFileWriter::Make(
    std::shared_ptr<arrow::fs::FileSystem> fs,
    std::shared_ptr<arrow::Schema> schema,
    const std::string& file_path,
    const milvus_storage::api::Properties& api_properties) {
  ARROW_ASSIGN_OR_RAISE(auto part_size,
                        milvus_storage::api::GetValue<int64_t>(api_properties, PROPERTY_FS_MULTI_PART_UPLOAD_SIZE));
  auto writer_props = properties::ConvertFromApiProperties(api_properties, schema);
  auto writer = std::unique_ptr<ParquetFileWriter>(new ParquetFileWriter(
      std::move(schema), std::move(fs), file_path, milvus_storage::StorageConfig{part_size}, writer_props));
  ARROW_RETURN_NOT_OK(writer->init());
  return writer;
}

arrow::Result<std::unique_ptr<ParquetFileWriter>> ParquetFileWriter::Make(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const std::string& file_path,
    const milvus_storage::StorageConfig& storage_config,
    const std::shared_ptr<::parquet::WriterProperties>& writer_props) {
  auto finalized_writer_props = properties::NormalizeWriterProperties(schema, writer_props);
  auto writer = std::unique_ptr<ParquetFileWriter>(
      new ParquetFileWriter(std::move(schema), std::move(fs), file_path, storage_config, finalized_writer_props));
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
      return arrow::Status::IOError(fmt::format("Failed to create directory [path={}, details: {}]",
                                                parent_dir_path.string(),  // NOLINT
                                                create_dir_result.ToString()));
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
    return arrow::Status::IOError(fmt::format("Failed to open output stream: {}", sink_result.status().ToString()));
  }
  sink_ = std::move(sink_result).ValueOrDie();

  auto writer_result = ::parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(), sink_, writer_props_);
  if (!writer_result.ok()) {
    return arrow::Status::IOError(
        fmt::format("Failed to create parquet writer: {}", writer_result.status().ToString()));
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

arrow::Result<size_t> ParquetFileWriter::Tell() const {
  if (closed_) {
    return cached_tell_;
  }
  ARROW_ASSIGN_OR_RAISE(auto pos, sink_->Tell());
  return static_cast<size_t>(pos);
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
    return arrow::Status::Invalid(
        fmt::format("Current writer is closed or writer is not initialized. [file_path={}]", file_path_));
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
  ARROW_ASSIGN_OR_RAISE(auto pos, sink_->Tell());

  // Measure footer size by re-serializing the metadata to a buffer.
  // Parquet footer = [Thrift FileMetaData][4B footer_length][4B magic "PAR1"]
  // Note: FileMetaData::size() only works for read-path metadata, not writer-created metadata.
  ARROW_ASSIGN_OR_RAISE(auto meta_sink, arrow::io::BufferOutputStream::Create());
  writer_->metadata()->WriteTo(meta_sink.get());
  ARROW_ASSIGN_OR_RAISE(auto meta_buffer, meta_sink->Finish());
  auto footer_size = static_cast<uint64_t>(meta_buffer->size()) + 8;

  cached_tell_ = static_cast<size_t>(pos);
  ARROW_RETURN_NOT_OK(sink_->Flush());
  ARROW_ASSIGN_OR_RAISE(auto file_size, sink_->Tell());
  ARROW_RETURN_NOT_OK(sink_->Close());

  closed_ = true;
  return api::ColumnGroupFile{
      .path = file_path_,
      .start_index = 0,
      .end_index = written_rows_,
      .properties = {{api::kPropertyFileSize, std::to_string(file_size)},
                     {api::kPropertyFooterSize, std::to_string(footer_size)}},
  };
}

}  // namespace milvus_storage::parquet
