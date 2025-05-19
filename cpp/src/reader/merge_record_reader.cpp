

#include "milvus-storage/reader/merge_record_reader.h"
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <parquet/file_reader.h>
#include <memory>
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/status.h"
#include "milvus-storage/reader/common/combine_reader.h"
#include "milvus-storage/common/result.h"
#include "milvus-storage/reader/common/delete_reader.h"
#include "milvus-storage/common/utils.h"
#include "milvus-storage/reader/common/projection_reader.h"
#include "milvus-storage/reader/multi_files_sequential_reader.h"

namespace milvus_storage {

MergeRecordReader::MergeRecordReader(const ReadOptions& options,
                                     const FragmentVector& scalar_fragments,
                                     const FragmentVector& vector_fragments,
                                     const DeleteFragmentVector& delete_fragments,
                                     arrow::fs::FileSystem& fs,
                                     std::shared_ptr<Schema> schema)
    : schema_(schema), fs_(fs), options_(options), delete_fragments_(delete_fragments) {
  scalar_reader_ = std::make_unique<MultiFilesSequentialReader>(fs, scalar_fragments, schema->scalar_schema(),
                                                                schema->options(), options);
  vector_reader_ = std::make_unique<MultiFilesSequentialReader>(fs, vector_fragments, schema->vector_schema(),
                                                                schema->options(), options);
}

std::shared_ptr<arrow::Schema> MergeRecordReader::schema() const {
  auto r = ProjectSchema(schema_->schema(), options_);
  return r.ok() ? r.value() : nullptr;
}

arrow::Status MergeRecordReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  if (!curr_reader_) {
    auto r = MakeInnerReader();
    if (!r.ok()) {
      return arrow::Status::UnknownError(r.status().ToString());
    }
    if (r.value() == nullptr) {
      batch = nullptr;
      return arrow::Status::OK();
    }
    curr_reader_ = std::move(r.value());
  }

  std::shared_ptr<arrow::RecordBatch> tmp_batch;
  auto s = curr_reader_->ReadNext(&tmp_batch);
  if (!s.ok()) {
    return s;
  }
  if (tmp_batch == nullptr) {
    return arrow::Status::OK();
  }

  *batch = tmp_batch;
  return arrow::Status::OK();
}

Result<std::unique_ptr<arrow::RecordBatchReader>> MergeRecordReader::MakeInnerReader() {
  auto combine_reader = CombineReader::Make(std::move(scalar_reader_), std::move(vector_reader_), schema_);
  auto delete_reader =
      DeleteMergeReader::Make(std::move(combine_reader), schema_->options(), delete_fragments_, options_);
  ASSIGN_OR_RETURN_NOT_OK(auto res, ProjectionReader::Make(schema(), std::move(delete_reader), options_));
  return res;
}
}  // namespace milvus_storage
