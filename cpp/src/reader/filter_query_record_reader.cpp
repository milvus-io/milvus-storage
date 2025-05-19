

#include "milvus-storage/reader/filter_query_record_reader.h"
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <cassert>
#include <memory>

#include "arrow/array/array_primitive.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/reader/common/combine_offset_reader.h"
#include "milvus-storage/reader/common/delete_reader.h"
#include "milvus-storage/reader/common/filter_reader.h"
#include "milvus-storage/reader/common/projection_reader.h"
#include "milvus-storage/common/utils.h"
namespace milvus_storage {

FilterQueryRecordReader::FilterQueryRecordReader(const ReadOptions& options,
                                                 const FragmentVector& scalar_fragments,
                                                 const FragmentVector& vector_fragments,
                                                 const DeleteFragmentVector& delete_fragments,
                                                 arrow::fs::FileSystem& fs,
                                                 std::shared_ptr<Schema> schema)
    : fs_(fs), schema_(schema), options_(options), delete_fragments_(delete_fragments) {
  // TODO: init schema

  for (const auto& fragment : vector_fragments) {
    vector_files_.insert(vector_files_.end(), fragment.files().begin(), fragment.files().end());
  }
  for (const auto& fragment : scalar_fragments) {
    scalar_files_.insert(scalar_files_.end(), fragment.files().begin(), fragment.files().end());
  }

  assert(scalar_files_.size() == vector_files_.size());
}
std::shared_ptr<arrow::Schema> FilterQueryRecordReader::schema() const {
  auto r = ProjectSchema(schema_->schema(), options_);
  if (!r.ok()) {
    return nullptr;
  }
  return r.value();
}

arrow::Status FilterQueryRecordReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  while (true) {
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
      curr_reader_ = nullptr;
      holding_scalar_file_reader_ = nullptr;
      holding_vector_file_reader_ = nullptr;
      continue;
    }

    *batch = tmp_batch;
    return arrow::Status::OK();
  }
}

Result<std::unique_ptr<arrow::RecordBatchReader>> FilterQueryRecordReader::MakeInnerReader() {
  if (next_pos_ >= scalar_files_.size()) {
    std::unique_ptr<arrow::RecordBatchReader> res = nullptr;
    return res;
  }

  auto scalar_file = scalar_files_[next_pos_], vector_file = vector_files_[next_pos_];
  ASSIGN_OR_RETURN_NOT_OK(holding_scalar_file_reader_, MakeArrowFileReader(fs_, scalar_file));
  ASSIGN_OR_RETURN_NOT_OK(holding_vector_file_reader_, MakeArrowFileReader(fs_, vector_file));
  ASSIGN_OR_RETURN_NOT_OK(
      auto scalar_rec_reader,
      MakeArrowRecordBatchReader(*holding_scalar_file_reader_, schema_->scalar_schema(), schema_->options(), options_));
  auto current_vector_reader = std::make_unique<ParquetFileReader>(std::move(holding_vector_file_reader_));

  auto combine_reader =
      CombineOffsetReader::Make(std::move(scalar_rec_reader), std::move(current_vector_reader), schema_);
  auto filter_reader = FilterReader::Make(std::move(combine_reader), options_);
  auto delete_reader =
      DeleteMergeReader::Make(std::move(filter_reader), schema_->options(), delete_fragments_, options_);
  ASSIGN_OR_RETURN_NOT_OK(auto projection_reader,
                          ProjectionReader::Make(schema_->schema(), std::move(delete_reader), options_));

  next_pos_++;
  return projection_reader;
}
}  // namespace milvus_storage
