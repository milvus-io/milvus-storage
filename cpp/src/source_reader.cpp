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

#include "milvus-storage/source_reader.h"
#include "milvus-storage/format/source_format_reader.h"

namespace milvus_storage {
namespace {
arrow::Status CheckSource(const SourceReadContextPtr& context) {
  return fs::SourceCredentialArrowStatus(context ? context->CheckState() : LOON_SOURCE_CREDENTIAL_DENIED);
}

template <typename Fn>
auto CheckedSource(const SourceReadContextPtr& context, Fn&& fn) -> decltype(fn()) {
  ARROW_RETURN_NOT_OK(CheckSource(context));
  auto result = fn();
  ARROW_RETURN_NOT_OK(CheckSource(context));
  return result;
}

template <typename Result, typename Owner, typename Work>
folly::SemiFuture<Result> CheckedSourceAsync(SourceReadContextPtr context, Owner owner, Work work) {
  return folly::makeSemiFuture().deferValue(
      [context, owner, work = std::move(work)](folly::Unit) mutable -> folly::SemiFuture<Result> {
        auto status = CheckSource(context);
        if (!status.ok())
          return folly::makeSemiFuture(Result(status));
        return work(*owner).deferValue([context, owner](Result result) -> Result {
          ARROW_RETURN_NOT_OK(CheckSource(context));
          return result;
        });
      });
}

class SourceChunkReader final : public api::ChunkReader {
  public:
  SourceChunkReader(std::unique_ptr<api::ChunkReader> reader, SourceReadContextPtr context)
      : reader_(std::move(reader)), context_(std::move(context)) {}
  size_t total_number_of_chunks() const override { return reader_->total_number_of_chunks(); }
  arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& rows) override {
    return CheckedSource(context_, [&] { return reader_->get_chunk_indices(rows); });
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t index) override {
    return CheckedSource(context_, [&] { return reader_->get_chunk(index); });
  }
  arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(const std::vector<int64_t>& indices,
                                                                             size_t parallelism) override {
    return CheckedSource(context_, [&] { return reader_->get_chunks(indices, parallelism); });
  }
  folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>> get_chunks_async(
      const std::vector<int64_t>& indices, size_t parallelism, folly::CancellationToken cancellation) override {
    using Result = arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>;
    return CheckedSourceAsync<Result>(context_, reader_,
                                      [indices, parallelism, cancellation](api::ChunkReader& reader) {
                                        return reader.get_chunks_async(indices, parallelism, cancellation);
                                      });
  }
  arrow::Result<std::vector<uint64_t>> get_chunk_estimated_size() override {
    return CheckedSource(context_, [&] { return reader_->get_chunk_estimated_size(); });
  }
  arrow::Result<std::vector<std::vector<uint64_t>>> get_chunk_column_estimated_size() override {
    return CheckedSource(context_, [&] { return reader_->get_chunk_column_estimated_size(); });
  }
  arrow::Result<std::vector<uint64_t>> get_chunk_rows() override {
    return CheckedSource(context_, [&] { return reader_->get_chunk_rows(); });
  }

  private:
  std::shared_ptr<api::ChunkReader> reader_;
  SourceReadContextPtr context_;
};

class SourceReader final : public api::Reader {
  public:
  SourceReader(std::unique_ptr<api::Reader> reader, SourceReadContextPtr context)
      : reader_(std::move(reader)), context_(std::move(context)) {}
  std::shared_ptr<api::ColumnGroups> get_column_groups() const override { return reader_->get_column_groups(); }
  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& predicate) const override {
    ARROW_ASSIGN_OR_RAISE(auto stream,
                          CheckedSource(context_, [&] { return reader_->get_record_batch_reader(predicate); }));
    return GuardSourceBatchReader(std::move(stream), context_);
  }
  arrow::Result<std::unique_ptr<api::ChunkReader>> get_chunk_reader(
      int64_t index, const std::shared_ptr<std::vector<std::string>>& columns) const override {
    ARROW_ASSIGN_OR_RAISE(auto reader,
                          CheckedSource(context_, [&] { return reader_->get_chunk_reader(index, columns); }));
    return std::make_unique<SourceChunkReader>(std::move(reader), context_);
  }
  folly::SemiFuture<arrow::Result<std::unique_ptr<api::ChunkReader>>> get_chunk_reader_async(
      int64_t index,
      const std::shared_ptr<std::vector<std::string>>& columns,
      folly::CancellationToken cancellation) const override {
    using Result = arrow::Result<std::unique_ptr<api::ChunkReader>>;
    return CheckedSourceAsync<Result>(context_, reader_,
                                      [index, columns, cancellation](api::Reader& reader) {
                                        return reader.get_chunk_reader_async(index, columns, cancellation);
                                      })
        .deferValue([context = context_](Result result) -> Result {
          ARROW_ASSIGN_OR_RAISE(auto reader, std::move(result));
          return std::make_unique<SourceChunkReader>(std::move(reader), context);
        });
  }
  arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& rows,
                                                    size_t parallelism,
                                                    const std::shared_ptr<std::vector<std::string>>& columns) override {
    return CheckedSource(context_, [&] { return reader_->take(rows, parallelism, columns); });
  }
  folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> take_async(
      const std::vector<int64_t>& rows,
      size_t parallelism,
      const std::shared_ptr<std::vector<std::string>>& columns,
      folly::CancellationToken cancellation) override {
    using Result = arrow::Result<std::shared_ptr<arrow::Table>>;
    return CheckedSourceAsync<Result>(context_, reader_,
                                      [rows, parallelism, columns, cancellation](api::Reader& reader) {
                                        return reader.take_async(rows, parallelism, columns, cancellation);
                                      });
  }
  void set_keyretriever(const KeyRetriever& callback) override { reader_->set_keyretriever(callback); }

  private:
  std::shared_ptr<api::Reader> reader_;
  SourceReadContextPtr context_;
};
}  // namespace

std::unique_ptr<api::Reader> GuardSourceReader(std::unique_ptr<api::Reader> reader, SourceReadContextPtr context) {
  return std::make_unique<SourceReader>(std::move(reader), std::move(context));
}
}  // namespace milvus_storage
