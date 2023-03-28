#include "merge_record_reader.h"

#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/type_fwd.h>

#include <memory>

#include "format/parquet/file_reader.h"
#include "scan_record_reader.h"

MergeRecordReader::MergeRecordReader(std::shared_ptr<ReadOption>& options,
                                     const std::vector<std::string>& scalar_files,
                                     const std::vector<std::string>& vector_files,
                                     const DefaultSpace& space)
    : space_(space) {
  scalar_reader_ = std::make_unique<ScanRecordReader>(options, scalar_files, space);
  vector_reader_ = std::make_unique<ScanRecordReader>(options, vector_files, space);
}

std::shared_ptr<arrow::Schema>
MergeRecordReader::schema() const {
  // TODO: projection
  return space_.manifest_->get_schema();
}

arrow::Status
MergeRecordReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  std::shared_ptr<arrow::RecordBatch> scalar_batch;
  std::shared_ptr<arrow::RecordBatch> vector_batch;
  scalar_reader_->ReadNext(&scalar_batch);
  vector_reader_->ReadNext(&vector_batch);
  if (scalar_batch == nullptr || vector_batch == nullptr) {
    return arrow::Status::OK();
  }

  auto vec_column = vector_batch->GetColumnByName(space_.options_->vector_column);
  std::vector<std::shared_ptr<arrow::Array>> columns(scalar_batch->columns().begin(), scalar_batch->columns().end());

  auto vec_column_idx = space_.manifest_->get_schema()->GetFieldIndex(space_.options_->vector_column);
  columns.insert(columns.begin() + vec_column_idx, vec_column);

  *batch = arrow::RecordBatch::Make(schema(), scalar_batch->num_rows(), std::move(columns));
  return arrow::Status::OK();
}
