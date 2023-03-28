#include "filter_query_record_reader.h"

#include <arrow/dataset/scanner.h>
#include <arrow/status.h>

#include <memory>

#include "arrow/array/array_base.h"
#include "arrow/array/array_primitive.h"
#include "common/exception.h"
#include "reader/scan_record_reader.h"

FilterQueryRecordReader::FilterQueryRecordReader(std::shared_ptr<ReadOption>& options,
                                                 const std::vector<std::string>& scalar_files,
                                                 const std::vector<std::string>& vector_files,
                                                 const DefaultSpace& space)
    : space_(space), options_(options), vector_files_(vector_files) {
  if (scalar_files.size() != vector_files.size()) {
    throw StorageException("file num should be same");
  }
  // TODO: init schema
  scalar_reader_ = std::make_unique<ScanRecordReader>(options, scalar_files, space);
}
std::shared_ptr<arrow::Schema>
FilterQueryRecordReader::schema() const {
  return nullptr;
}

arrow::Status
FilterQueryRecordReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  std::shared_ptr<arrow::RecordBatch> tmp_batch;

  auto status = scalar_reader_->ReadNext(&tmp_batch);
  if (!status.ok()) {
    return status;
  }

  if (tmp_batch == nullptr) {
    return arrow::Status::OK();
  }

  // read vector data and merge together
  if (scalar_reader_->next_pos_ > next_pos_) {
    if (current_vector_reader_ != nullptr) {
      current_vector_reader_->Close();
    }
    current_vector_reader_ =
        std::make_unique<ParquetFileReader>(space_.fs_.get(), vector_files_[next_pos_++], options_);
  }

  auto col_arr = tmp_batch->GetColumnByName(kOffsetFieldName);
  if (col_arr == nullptr) {
    throw StorageException("__offset column not found");
  }
  auto offset_arr = std::dynamic_pointer_cast<arrow::Int64Array>(col_arr);
  std::vector<int64_t> offsets;
  for (const auto& v : *offset_arr) {
    offsets.emplace_back(v.value());
  }

  auto table = current_vector_reader_->ReadByOffsets(offsets);
  // maybe copy here
  PARQUET_ASSIGN_OR_THROW(auto table_batch, table->CombineChunksToBatch());

  std::vector<std::shared_ptr<arrow::Array>> columns(tmp_batch->columns().begin(), tmp_batch->columns().end());

  auto vector_col = table_batch->GetColumnByName(space_.options_->vector_column);
  if (vector_col == nullptr) {
    throw StorageException("vector column not found");
  }
  columns.emplace_back(vector_col);

  *batch = arrow::RecordBatch::Make(space_.manifest_->get_schema(), tmp_batch->num_rows(), std::move(columns));
  return arrow::Status::OK();
}
