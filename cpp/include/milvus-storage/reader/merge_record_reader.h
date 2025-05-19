

#pragma once

#include "arrow/record_batch.h"
#include "milvus-storage/file/delete_fragment.h"
#include "milvus-storage/file/fragment.h"
#include "milvus-storage/storage/options.h"
namespace milvus_storage {

// MergeRecordReader is to scan files to get records and merge them together.
// It organize other readers sequentially.
// ProjectionReader - DeleteReader - CombineReader - FileReader(vector)
//                                                 \ FileReader(scalar)
class MergeRecordReader : public arrow::RecordBatchReader {
  public:
  explicit MergeRecordReader(const ReadOptions& options,
                             const FragmentVector& scalar_fragments,
                             const FragmentVector& vector_fragments,
                             const DeleteFragmentVector& delete_fragments,
                             arrow::fs::FileSystem& fs,
                             std::shared_ptr<Schema> schema);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  Result<std::unique_ptr<arrow::RecordBatchReader>> MakeInnerReader();

  arrow::fs::FileSystem& fs_;
  std::shared_ptr<Schema> schema_;
  const ReadOptions options_;

  std::unique_ptr<arrow::RecordBatchReader> scalar_reader_;
  std::unique_ptr<arrow::RecordBatchReader> vector_reader_;
  std::unique_ptr<arrow::RecordBatchReader> curr_reader_;
  const DeleteFragmentVector delete_fragments_;
};
}  // namespace milvus_storage
