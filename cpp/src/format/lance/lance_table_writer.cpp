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

#ifdef BUILD_GTEST

#include "milvus-storage/format/lance/lance_table_writer.h"

#include <string>
#include <iostream>
#include <utility>

#include <arrow/chunked_array.h>  // keep this line before other arrow header
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/status.h>
#include <arrow/result.h>
#include <fmt/format.h>

#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/common/arrow_util.h"

namespace milvus_storage::lance {

LanceTableWriter::LanceTableWriter(const std::string& base_path,
                                   std::shared_ptr<arrow::Schema> schema,
                                   const api::Properties& properties,
                                   LanceDataStorageFormat data_storage_format)
    : closed_(false),
      base_path_(base_path),
      schema_(std::move(schema)),
      properties_(properties),
      data_storage_format_(data_storage_format) {
  assert(schema_);
}

class BatchIterator : public arrow::RecordBatchReader {
  public:
  BatchIterator(const std::shared_ptr<arrow::Schema>& schema,
                const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches)
      : schema_(schema), batches_(batches) {}

  [[nodiscard]] std::shared_ptr<arrow::Schema> schema() const override { return schema_; }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out) override {
    if (position_ >= batches_.size()) {
      *out = nullptr;
    } else {
      *out = batches_[position_++];
    }
    return arrow::Status::OK();
  }

  private:
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  size_t position_{0};
};

arrow::Status LanceTableWriter::Write(const std::shared_ptr<arrow::RecordBatch> batch) {
  assert(!closed_);
  assert(batch->schema()->Equals(*schema_, false));
  written_rows_ += batch->num_rows();

  record_batches_.emplace_back(batch);
  return arrow::Status::OK();
}

arrow::Status LanceTableWriter::Flush() { return arrow::Status::OK(); }

size_t LanceTableWriter::GetRetainedBufferSize() const {
  size_t size = 0;
  for (const auto& batch : record_batches_) {
    size += GetRecordBatchMemorySize(batch);
  }
  return size;
}

arrow::Result<api::ColumnGroupFile> LanceTableWriter::Close() {
  assert(!closed_);

  // Get storage options from properties for cloud storage support
  ArrowFileSystemConfig fs_config;
  ARROW_RETURN_NOT_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  ARROW_ASSIGN_OR_RAISE(auto storage_options, ToWriterOptions(fs_config));

  // Build full Lance URI from relative path
  ARROW_ASSIGN_OR_RAISE(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));

  struct ArrowArrayStream array_stream;
  auto batch_iterator = std::make_shared<BatchIterator>(schema_, record_batches_);
  ARROW_RETURN_NOT_OK(ExportRecordBatchReader(batch_iterator, &array_stream));

  ARROW_ASSIGN_OR_RAISE(auto written_fragment_ids,
                        BlockingDataset::WriteDataset(lance_uri, &array_stream, storage_options, data_storage_format_));
  record_batches_.clear();

  // Store Milvus-format URI (scheme://address/bucket/key) in ColumnGroupFile.path
  // so the reader can resolve the right extfs.<alias>.* by address+bucket. The
  // reader strips address back to standard form before handing to Lance.
  auto milvus_lance_uri = ToMilvusLanceUri(lance_uri, fs_config.address);

  if (written_fragment_ids.size() != 1) {
    return arrow::Status::Invalid(
        fmt::format("LanceTableWriter expected one new fragment, got {}", written_fragment_ids.size()));
  }
  closed_ = true;
  return api::ColumnGroupFile{
      .path = MakeLanceUri(milvus_lance_uri, written_fragment_ids[0]),
      .start_index = 0,
      .end_index = written_rows_,
  };
}

}  // namespace milvus_storage::lance

#endif  // BUILD_GTEST
