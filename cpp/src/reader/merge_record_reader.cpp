#include "reader/merge_record_reader.h"
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <parquet/file_reader.h>
#include <memory>
#include "common/arrow_util.h"
#include "common/macro.h"
#include "common/status.h"
#include "reader/common/combine_reader.h"
#include "common/result.h"
#include "reader/common/delete_reader.h"
#include "common/utils.h"
#include "reader/common/projection_reader.h"
#include "reader/multi_files_sequential_reader.h"

namespace milvus_storage {

MergeRecordReader::MergeRecordReader(std::shared_ptr<ReadOptions> options,
                                     const FragmentVector& scalar_fragments,
                                     const FragmentVector& vector_fragments,
                                     const DeleteFragmentVector& delete_fragments,
                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                     std::shared_ptr<Schema> schema)
    : schema_(schema), fs_(fs), options_(options), delete_fragments_(delete_fragments) {
  scalar_reader_ = std::make_shared<MultiFilesSequentialReader>(fs, scalar_fragments, schema->scalar_schema());
  vector_reader_ = std::make_shared<MultiFilesSequentialReader>(fs, vector_fragments, schema->vector_schema());
}

std::shared_ptr<arrow::Schema> MergeRecordReader::schema() const {
  auto r = ProjectSchema(schema_->schema(), options_->output_columns());
  return r.ok() ? r.value() : nullptr;
}

arrow::Status MergeRecordReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  if (!curr_reader_) {
    auto r = MakeInnerReader();
    if (!r.ok()) {
      return arrow::Status::UnknownError(r.status().ToString());
    }
    auto reader = r.value();
    if (reader == nullptr) {
      batch = nullptr;
      return arrow::Status::OK();
    }
    curr_reader_ = reader;
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

Result<std::shared_ptr<arrow::RecordBatchReader>> MergeRecordReader::MakeInnerReader() {
  ASSIGN_OR_RETURN_NOT_OK(auto combine_reader, CombineReader::Make(scalar_reader_, vector_reader_, schema_));
  auto delete_reader = DeleteMergeReader::Make(combine_reader, schema_->options(), delete_fragments_);
  ASSIGN_OR_RETURN_NOT_OK(auto res, ProjectionReader::Make(schema(), delete_reader, options_));
  return res;
}
}  // namespace milvus_storage
