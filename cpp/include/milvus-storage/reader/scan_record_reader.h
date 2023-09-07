#pragma once

#include <arrow/record_batch.h>
#include "file/delete_fragment.h"
#include "file/fragment.h"
#include "reader/multi_files_sequential_reader.h"
namespace milvus_storage {
class ScanRecordReader : public arrow::RecordBatchReader {
  public:
  ScanRecordReader(std::shared_ptr<Schema> schema,
                   std::shared_ptr<ReadOptions> options,
                   std::shared_ptr<arrow::fs::FileSystem> fs,
                   const FragmentVector& fragments,
                   const DeleteFragmentVector& delete_fragments);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  Result<std::shared_ptr<arrow::RecordBatchReader>> MakeInnerReader();

  std::shared_ptr<Schema> schema_;
  std::shared_ptr<ReadOptions> options_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  const FragmentVector fragments_;
  const DeleteFragmentVector delete_fragments_;
  std::shared_ptr<arrow::RecordBatchReader> reader_;
};

}  // namespace milvus_storage
