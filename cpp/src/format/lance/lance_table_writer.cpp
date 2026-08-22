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

#include <new>
#include <string>
#include <iostream>
#include <unordered_set>
#include <utility>

#include <arrow/chunked_array.h>  // keep this line before other arrow header
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/util/io_util.h>
#include <fmt/format.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/format/lance/lance_common.h"

namespace milvus_storage::lance {

LanceTableWriter::LanceTableWriter(const std::string& base_path,
                                   std::shared_ptr<arrow::Schema> schema,
                                   const api::Properties& properties,
                                   LanceDataStorageFormat data_storage_format)
    : closed_(false),
      base_path_(base_path),
      schema_(std::move(schema)),
      properties_(properties),
      data_storage_format_(data_storage_format),
      dataset_(nullptr) {
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
  ARROW_RETURN_NOT_OK(writer_status_.Check());
  return writer_status_.Fail(WriteImpl(batch));
}

arrow::Status LanceTableWriter::WriteImpl(const std::shared_ptr<arrow::RecordBatch>& batch) {
  assert(!closed_);
  assert(batch->schema()->Equals(*schema_, false));
  written_rows_ += batch->num_rows();

  record_batches_.emplace_back(batch);
  return arrow::Status::OK();
}

arrow::Status LanceTableWriter::Flush() {
  ARROW_RETURN_NOT_OK(writer_status_.Check());
  return writer_status_.Fail(FlushImpl());
}

arrow::Status LanceTableWriter::FlushImpl() { return arrow::Status::OK(); }

bool fids_contains(const std::vector<uint64_t>& origin, const std::vector<uint64_t>& current) {
  assert(current.size() > origin.size());
  std::unordered_set<uint64_t> current_set(current.begin(), current.end());
  for (uint64_t elem : origin) {
    if (current_set.find(elem) == current_set.end()) {
      return false;
    }
  }
  return true;
}

/// Whether a failed open means "there is nothing here yet", so the first write
/// must create the dataset rather than fail.
///
/// ONLY a positively identified not-found qualifies. This used to also accept
/// any unclassified IO failure, on the theory that a store whose not-found we
/// failed to classify would otherwise break first-write. The cost of that
/// tolerance was not a worse error message: taking this branch against a
/// dataset that DOES exist makes the writer believe it started from zero
/// fragments, and everything downstream is computed from that belief (see the
/// guard in CloseImpl). A transient open failure would have been laundered
/// into a manifest entry pointing at somebody else's fragment. Failing the
/// first write is recoverable; that is not.
bool IsMissingDatasetStatus(const arrow::Status& status) {
  if (auto detail = ExtendStatusDetail::UnwrapStatus(status); detail != nullptr) {
    return detail->code() == ExtendStatusCode::StorageNotFound;
  }
  // The filesystem layer reports a missing object through errno rather than the
  // extend taxonomy; the bridges report it through the taxonomy above.
  return arrow::internal::ErrnoFromStatus(status) == ENOENT;
}

std::vector<uint64_t> fids_diff(const std::vector<uint64_t>& origin, const std::vector<uint64_t>& current) {
  assert(current.size() > origin.size());

  std::unordered_set<uint64_t> origin_set(origin.begin(), origin.end());
  std::vector<uint64_t> diff;

  for (uint64_t elem : current) {
    if (origin_set.find(elem) == origin_set.end()) {
      diff.emplace_back(elem);
    }
  }
  return diff;
}

void LanceTableWriter::Abort() noexcept {
  // closed_ before BeginDiscard(): abort after a successful Close must not
  // leave a writer that finished cleanly reading as Cancelled.
  if (closed_) {
    return;
  }
  writer_status_.BeginDiscard();
  closed_ = true;
  // Buffered batches never reached the store, so dropping them is the whole of
  // the local cleanup. Fragments already written through the Rust dataset are
  // left behind: lance publishes them by committing a manifest version, so an
  // abandoned write leaves fragments no version references, and removing those
  // needs a lance-side API this bridge does not expose.
  //
  // Accepted, because this writer does not exist outside test builds -- the
  // whole class is behind #ifdef BUILD_GTEST and LanceFormat::create_writer
  // returns NotImplemented("Lance writer is only available in test builds")
  // otherwise. Production reads lance, it never writes it, so there is no
  // deployment in which these fragments accumulate.
  record_batches_.clear();
  dataset_.reset();
}

arrow::Result<api::ColumnGroupFile> LanceTableWriter::Close() {
  // Abandon on both failure paths; see FormatWriter::Close in format_writer.h.
  if (auto first_failure = writer_status_.Check(); !first_failure.ok()) {
    Abort();
    return first_failure;
  }
  auto result = CloseImpl();
  if (!result.ok()) {
    auto status = writer_status_.Fail(result.status());
    Abort();
    return status;
  }
  return result;
}

arrow::Result<api::ColumnGroupFile> LanceTableWriter::CloseImpl() {
  assert(!closed_);
  struct ArrowArrayStream array_stream;

  auto batch_iterator = std::make_shared<BatchIterator>(schema_, record_batches_);
  ARROW_RETURN_NOT_OK(ExportRecordBatchReader(batch_iterator, &array_stream));
  // The write calls below consume the stream and null its release; if we bail
  // out before reaching one (open/config failure), release it here instead of
  // leaking the exported reader and every buffered batch.
  auto stream_guard = [](ArrowArrayStream* stream) {
    if (stream->release != nullptr) {
      stream->release(stream);
    }
  };
  std::unique_ptr<ArrowArrayStream, decltype(stream_guard)> release_on_exit(&array_stream, stream_guard);

  // Get storage options from properties for cloud storage support
  ArrowFileSystemConfig fs_config;
  ARROW_RETURN_NOT_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
  auto storage_options = ToStorageOptions(fs_config);

  // Build full Lance URI from relative path
  ARROW_ASSIGN_OR_RAISE(auto lance_uri, BuildLanceBaseUri(fs_config, base_path_));

  if (!dataset_) {
    auto opened = BlockingDataset::OpenUnique(lance_uri, storage_options);
    if (opened.ok()) {
      dataset_ = std::move(*opened);
      ARROW_ASSIGN_OR_RAISE(origin_fids_, dataset_->GetAllFragmentIds());
      ARROW_RETURN_NOT_OK(dataset_->WriteArrowArrayStream(&array_stream));
    } else if (IsMissingDatasetStatus(opened.status())) {
      origin_fids_.clear();
      created_dataset_ = true;
      ARROW_ASSIGN_OR_RAISE(
          dataset_, BlockingDataset::WriteDataset(lance_uri, &array_stream, storage_options, data_storage_format_));
    } else {
      return opened.status();
    }
  } else {
    ARROW_RETURN_NOT_OK(dataset_->WriteArrowArrayStream(&array_stream));
  }
  record_batches_.clear();

  std::vector<uint64_t> append_fids;
  std::vector<uint64_t> current_fids;
  ARROW_ASSIGN_OR_RAISE(current_fids, dataset_->GetAllFragmentIds());

  if (current_fids.size() < origin_fids_.size()) {
    return arrow::Status::Invalid(
        fmt::format("LanceTableWriter: current fragment ids size is less than origin fragment ids size [current "
                    "size={}, origin size={}]",
                    current_fids.size(),  // NOLINT
                    origin_fids_.size()));
  }

  // Store Milvus-format URI (scheme://address/bucket/key) in ColumnGroupFile.path
  // so the reader can resolve the right extfs.<alias>.* by address+bucket. The
  // reader strips address back to standard form before handing to Lance.
  auto milvus_lance_uri = ToMilvusLanceUri(lance_uri, fs_config.address);

  if (current_fids.size() == origin_fids_.size()) {
    return api::ColumnGroupFile{.path = milvus_lance_uri, .start_index = 0, .end_index = written_rows_};
  }

  if (!fids_contains(origin_fids_, current_fids)) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated,
                              "LanceTableWriter: current fragment ids is not a superset of origin fragment ids");
  }

  // Everything below is derived from origin_fids_ being an accurate picture of
  // the dataset before this write. When the dataset was created here, "before"
  // means empty -- and an empty origin makes both checks above vacuous, because
  // every set contains the empty set. So a wrong not-found verdict on an
  // EXISTING dataset would sail through them and hand back the oldest existing
  // fragment as if this writer had just produced it, with this writer's row
  // count attached: the rows written here referenced by nothing, and the entry
  // pointing at somebody else's data. Assert did not cover it -- release builds
  // compile it out, which is exactly where this would have shipped.
  append_fids = fids_diff(origin_fids_, current_fids);
  if (append_fids.size() != 1) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated,
                              "LanceTableWriter: expected exactly one appended fragment, got ", append_fids.size(),
                              created_dataset_ ? " after creating the dataset (it already existed, so the open failure "
                                                 "that led here was not a missing dataset)"
                                               : "");
  }

  dataset_.reset();
  closed_ = true;
  return api::ColumnGroupFile{
      .path = MakeLanceUri(milvus_lance_uri, append_fids[0]),
      .start_index = 0,
      .end_index = written_rows_,
  };
}

}  // namespace milvus_storage::lance

#endif  // BUILD_GTEST
