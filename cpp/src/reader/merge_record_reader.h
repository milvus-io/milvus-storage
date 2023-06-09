#pragma once

#include "arrow/record_batch.h"
#include "storage/options.h"
#include "storage/default_space.h"
namespace milvus_storage {

// MergeRecordReader is to scan files to get records and merge them together.
// It organize other readers sequentially.
// ProjectionReader - DeleteReader - CombineReader - FileReader(vector)
//                                                 \ FileReader(scalar)
class MergeRecordReader : public arrow::RecordBatchReader {
  public:
  explicit MergeRecordReader(std::shared_ptr<ReadOptions>& options,
                             const std::vector<std::string>& scalar_files,
                             const std::vector<std::string>& vector_files,
                             const DefaultSpace& space);
  std::shared_ptr<arrow::Schema> schema() const override;
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  const DefaultSpace& space_;
  std::shared_ptr<ReadOptions> options_;
  std::vector<std::string> scalar_fiels_;
  std::vector<std::string> vector_fiels_;

  std::shared_ptr<arrow::RecordBatchReader> curr_reader_ = nullptr;
  size_t curr_idx_ = 0;
};
}  // namespace milvus_storage
