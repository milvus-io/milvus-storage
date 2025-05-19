

#pragma once

#include <arrow/type_fwd.h>
#include "arrow/record_batch.h"
#include "parquet/arrow/reader.h"
#include "milvus-storage/common/result.h"
#include <utility>
#include "milvus-storage/storage/options.h"

namespace milvus_storage {

// FilterReader filters data by the filters passed by read options.
class FilterReader : public arrow::RecordBatchReader {
  public:
  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  static std::unique_ptr<FilterReader> Make(std::unique_ptr<arrow::RecordBatchReader> reader,
                                            const ReadOptions& option);

  FilterReader(std::unique_ptr<arrow::RecordBatchReader> reader, const ReadOptions& option)
      : record_reader_(std::move(reader)), option_(option) {}

  private:
  arrow::Status NextFilteredBatchReader();

  std::unique_ptr<arrow::RecordBatchReader> record_reader_;
  const ReadOptions& option_;
  std::shared_ptr<arrow::RecordBatchReader> current_filtered_batch_reader_;
};
}  // namespace milvus_storage
