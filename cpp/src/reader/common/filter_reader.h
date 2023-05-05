#pragma once

#include <arrow/type_fwd.h>
#include "arrow/record_batch.h"
#include "parquet/arrow/reader.h"
#include "common/result.h"
#include <utility>
#include "storage/options.h"

namespace milvus_storage {
class FilterReader : public arrow::RecordBatchReader {
  public:
  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  static Result<std::shared_ptr<FilterReader>> Make(std::shared_ptr<arrow::RecordBatchReader> reader,
                                                    std::shared_ptr<ReadOptions> option);

  private:
  FilterReader(std::shared_ptr<arrow::RecordBatchReader> reader, std::shared_ptr<ReadOptions> option)
      : record_reader_(std::move(reader)), option_(std::move(option)) {}

  arrow::Status NextFilteredBatchReader();

  std::shared_ptr<arrow::RecordBatchReader> record_reader_;
  std::shared_ptr<ReadOptions> option_;
  std::shared_ptr<arrow::RecordBatchReader> current_filtered_batch_reader_;
};
}  // namespace milvus_storage