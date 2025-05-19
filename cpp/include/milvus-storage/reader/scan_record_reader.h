

#pragma once

#include <arrow/record_batch.h>
#include "milvus-storage/file/delete_fragment.h"
#include "milvus-storage/file/fragment.h"
#include "milvus-storage/reader/multi_files_sequential_reader.h"
namespace milvus_storage {
class ScanRecordReader : public arrow::RecordBatchReader {
  public:
  ScanRecordReader(std::shared_ptr<arrow::Schema> schema,
                   const SchemaOptions& schema_options,
                   const ReadOptions& options,
                   arrow::fs::FileSystem& fs,
                   const FragmentVector& fragments,
                   const DeleteFragmentVector& delete_fragments);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  Result<std::unique_ptr<arrow::RecordBatchReader>> MakeInnerReader();

  std::shared_ptr<arrow::Schema> schema_;
  const SchemaOptions schema_options_;
  const ReadOptions options_;
  arrow::fs::FileSystem& fs_;
  const FragmentVector fragments_;
  const DeleteFragmentVector delete_fragments_;
  std::unique_ptr<arrow::RecordBatchReader> reader_;
};

}  // namespace milvus_storage
