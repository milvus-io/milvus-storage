#pragma once

#include "arrow/record_batch.h"
#include "file/delete_fragment.h"
#include "file/fragment.h"
#include "storage/options.h"
#include "storage/default_space.h"
namespace milvus_storage {

// MergeRecordReader is to scan files to get records and merge them together.
// It organize other readers sequentially.
// ProjectionReader - DeleteReader - CombineReader - FileReader(vector)
//                                                 \ FileReader(scalar)
class MergeRecordReader : public arrow::RecordBatchReader {
  public:
  explicit MergeRecordReader(std::shared_ptr<ReadOptions> options,
                             const FragmentVector& scalar_fragments,
                             const FragmentVector& vector_fragments,
                             const DeleteFragmentVector& delete_fragments,
                             std::shared_ptr<arrow::fs::FileSystem> fs,
                             std::shared_ptr<Schema> schema);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  Result<std::shared_ptr<arrow::RecordBatchReader>> MakeInnerReader();

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<Schema> schema_;
  std::shared_ptr<ReadOptions> options_;

  std::shared_ptr<arrow::RecordBatchReader> scalar_reader_;
  std::shared_ptr<arrow::RecordBatchReader> vector_reader_;
  std::shared_ptr<arrow::RecordBatchReader> curr_reader_;
  const DeleteFragmentVector delete_fragments_;
};
}  // namespace milvus_storage
