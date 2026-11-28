// Copyright 2026 Zilliz
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

#include "milvus-storage/format/source_format_reader.h"

#include "milvus-storage/filesystem/azure/source_sas_token_policy.h"

namespace milvus_storage {
namespace {
arrow::Status Check(const SourceReadContextPtr& context) {
  return fs::SourceCredentialArrowStatus(context ? context->CheckState() : LOON_SOURCE_CREDENTIAL_DENIED);
}

class SourceBatchReader final : public arrow::RecordBatchReader {
  public:
  SourceBatchReader(std::shared_ptr<arrow::RecordBatchReader> reader, SourceReadContextPtr context)
      : reader_(std::move(reader)), context_(std::move(context)) {}
  std::shared_ptr<arrow::Schema> schema() const override { return reader_->schema(); }
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
    if (!batch)
      return arrow::Status::Invalid("Missing output batch");
    batch->reset();
    ARROW_RETURN_NOT_OK(Check(context_));
    std::shared_ptr<arrow::RecordBatch> result;
    ARROW_RETURN_NOT_OK(reader_->ReadNext(&result));
    ARROW_RETURN_NOT_OK(Check(context_));
    *batch = std::move(result);
    return arrow::Status::OK();
  }
  // Cancellation never prevents releasing resources.
  arrow::Status Close() override { return reader_->Close(); }

  private:
  std::shared_ptr<arrow::RecordBatchReader> reader_;
  SourceReadContextPtr context_;
};

class SourceFormatReader final : public FormatReader {
  public:
  SourceFormatReader(std::shared_ptr<FormatReader> reader, SourceReadContextPtr context)
      : reader_(std::move(reader)), context_(std::move(context)) {}
  arrow::Status open() override {
    return Checked([&] { return reader_->open(); });
  }
  arrow::Result<std::vector<RowGroupInfo>> get_row_group_infos() override {
    return Checked([&] { return reader_->get_row_group_infos(); });
  }
  arrow::Result<std::vector<uint64_t>> get_rg_column_memsz(int64_t index) const override {
    return Checked([&] { return reader_->get_rg_column_memsz(index); });
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(const int& index) override {
    return Checked([&] { return reader_->get_chunk(index); });
  }
  arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(const std::vector<int>& indices) override {
    return Checked([&] { return reader_->get_chunks(indices); });
  }
  arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& indices) override {
    return Checked([&] { return reader_->take(indices); });
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> read_with_range(const uint64_t& start,
                                                                           const uint64_t& end) override {
    ARROW_ASSIGN_OR_RAISE(auto stream, Checked([&] { return reader_->read_with_range(start, end); }));
    return std::make_shared<SourceBatchReader>(std::move(stream), context_);
  }
  arrow::Result<std::shared_ptr<FormatReader>> clone_reader() override {
    ARROW_ASSIGN_OR_RAISE(auto clone, Checked([&] { return reader_->clone_reader(); }));
    return GuardSourceFormatReader(std::move(clone), context_);
  }
  std::shared_ptr<arrow::Schema> get_schema() const override { return reader_->get_schema(); }
  arrow::Status set_predicate(const std::string& predicate) override {
    return Checked([&] { return reader_->set_predicate(predicate); });
  }
  folly::SemiFuture<arrow::Status> open_async() override {
    return folly::makeSemiFuture().deferValue(
        [reader = reader_, context = context_](folly::Unit) -> folly::SemiFuture<arrow::Status> {
          auto status = Check(context);
          if (!status.ok())
            return folly::makeSemiFuture(status);
          return reader->open_async().deferValue([reader, context](arrow::Status result) {
            auto status = Check(context);
            return status.ok() ? result : status;
          });
        });
  }
  folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> take_async(
      const std::vector<int64_t>& indices) override {
    using Result = arrow::Result<std::shared_ptr<arrow::Table>>;
    return folly::makeSemiFuture().deferValue(
        [reader = reader_, context = context_, indices](folly::Unit) -> folly::SemiFuture<Result> {
          auto status = Check(context);
          if (!status.ok())
            return folly::makeSemiFuture(Result(status));
          return reader->take_async(indices).deferValue([reader, context](Result result) -> Result {
            ARROW_RETURN_NOT_OK(Check(context));
            return result;
          });
        });
  }
  folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::RecordBatchReader>>> read_with_range_async(
      uint64_t start, uint64_t end) override {
    using Result = arrow::Result<std::shared_ptr<arrow::RecordBatchReader>>;
    return folly::makeSemiFuture().deferValue(
        [reader = reader_, context = context_, start, end](folly::Unit) -> folly::SemiFuture<Result> {
          auto status = Check(context);
          if (!status.ok())
            return folly::makeSemiFuture(Result(status));
          return reader->read_with_range_async(start, end).deferValue([reader, context](Result result) -> Result {
            ARROW_RETURN_NOT_OK(Check(context));
            ARROW_ASSIGN_OR_RAISE(auto stream, std::move(result));
            return std::make_shared<SourceBatchReader>(std::move(stream), context);
          });
        });
  }

  private:
  template <typename F>
  auto Checked(F operation) const -> decltype(operation()) {
    ARROW_RETURN_NOT_OK(Check(context_));
    auto result = operation();
    ARROW_RETURN_NOT_OK(Check(context_));
    return result;
  }
  std::shared_ptr<FormatReader> reader_;
  SourceReadContextPtr context_;
};
}  // namespace

std::shared_ptr<FormatReader> GuardSourceFormatReader(std::shared_ptr<FormatReader> reader,
                                                      SourceReadContextPtr context) {
  return std::make_shared<SourceFormatReader>(std::move(reader), std::move(context));
}
std::shared_ptr<arrow::RecordBatchReader> GuardSourceBatchReader(std::shared_ptr<arrow::RecordBatchReader> reader,
                                                                 SourceReadContextPtr context) {
  return std::make_shared<SourceBatchReader>(std::move(reader), std::move(context));
}
}  // namespace milvus_storage
