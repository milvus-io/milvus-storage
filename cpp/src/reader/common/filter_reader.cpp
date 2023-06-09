#include "filter_reader.h"
#include <arrow/type_fwd.h>
#include "arrow/record_batch.h"
#include "arrow/table.h"

#include <memory>
#include <utility>

namespace milvus_storage {
Result<std::shared_ptr<FilterReader>> FilterReader::Make(std::shared_ptr<arrow::RecordBatchReader> reader,
                                                         std::shared_ptr<ReadOptions> option) {
  return std::make_shared<FilterReader>(reader, option);
}

std::shared_ptr<arrow::Schema> FilterReader::schema() const {
  // TODO
}

arrow::Status FilterReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  while (true) {
    if (current_filtered_batch_reader_) {
      std::shared_ptr<arrow::RecordBatch> filtered_batch;
      auto s = current_filtered_batch_reader_->ReadNext(&filtered_batch);
      if (!s.ok()) {
        return s;
      }
      if (!filtered_batch) {
        current_filtered_batch_reader_ = nullptr;
        continue;
      }
      *batch = std::move(filtered_batch);
      return arrow::Status::OK();
    }
    auto s = NextFilteredBatchReader();
    if (!s.ok()) {
      return s;
    }

    if (!current_filtered_batch_reader_) {
      *batch = nullptr;
      return arrow::Status::OK();
    }
  }
}

arrow::RecordBatchVector ApplyFilter(std::shared_ptr<arrow::RecordBatch>& batch, std::vector<Filter*>& filters) {
  filter_mask bitset;
  Filter::ApplyFilter(batch, filters, bitset);
  if (bitset.none()) {
    return {batch};
  }

  arrow::RecordBatchVector filterd_batches;
  int start_idx = 0, end_idx = 0;
  int64_t num_rows = batch->num_rows();
  while (end_idx < num_rows) {
    while (end_idx < num_rows && bitset.test(end_idx)) {
      end_idx++;
      start_idx++;
    }
    while (end_idx < num_rows && !bitset.test(end_idx)) {
      end_idx++;
    }

    if (start_idx >= num_rows) {
      break;
    }
    // zero-copy slice
    filterd_batches.emplace_back(batch->Slice(start_idx, end_idx - start_idx));
    start_idx = end_idx;
  }

  return filterd_batches;
}

arrow::Status FilterReader::NextFilteredBatchReader() {
  arrow::RecordBatchVector filtered_batches;
  do {
    auto r = record_reader_->Next();
    if (!r.ok()) {
      return r.status();
    }
    auto rec_batch = r.ValueUnsafe();
    if (!rec_batch) {
      break;
    }
    filtered_batches = ApplyFilter(rec_batch, option_->filters);
  } while (filtered_batches.empty());

  if (filtered_batches.empty()) {
    return arrow::Status::OK();
  }

  auto r = arrow::RecordBatchReader::Make(filtered_batches);
  if (!r.ok()) {
    return r.status();
  }
  current_filtered_batch_reader_ = r.ValueUnsafe();
  return arrow::Status::OK();
}
}  // namespace milvus_storage
