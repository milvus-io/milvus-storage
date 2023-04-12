#include "reader/filter_query_record_reader.h"

#include "arrow/array/array_primitive.h"
#include "reader/scan_record_reader.h"
namespace milvus_storage {

FilterQueryRecordReader::FilterQueryRecordReader(std::shared_ptr<ReadOptions>& options,
                                                 const std::vector<std::string>& scalar_files,
                                                 const std::vector<std::string>& vector_files,
                                                 const DefaultSpace& space)
    : space_(space), options_(options), vector_files_(vector_files) {
  // TODO: init schema
  scalar_reader_ = std::make_unique<ScanRecordReader>(options, scalar_files, space);
}
std::shared_ptr<arrow::Schema> FilterQueryRecordReader::schema() const { return nullptr; }

arrow::Status FilterQueryRecordReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  std::shared_ptr<arrow::RecordBatch> tmp_batch;

  ARROW_RETURN_NOT_OK(scalar_reader_->ReadNext(&tmp_batch));

  if (tmp_batch == nullptr) {
    return arrow::Status::OK();
  }

  // read vector data and merge together
  if (scalar_reader_->next_pos_ > next_pos_) {
    if (current_vector_reader_ != nullptr) {
      current_vector_reader_->Close();
    }
    current_vector_reader_ = std::make_unique<ParquetFileReader>(space_.fs_, vector_files_[next_pos_++], options_);
    auto s = current_vector_reader_->Init();
    if (!s.ok()) {
      return arrow::Status::UnknownError(s.ToString());
    }
  }

  auto col_arr = tmp_batch->GetColumnByName(kOffsetFieldName);
  if (col_arr == nullptr) {
    return arrow::Status::UnknownError("offset column not found");
  }
  auto offset_arr = std::dynamic_pointer_cast<arrow::Int64Array>(col_arr);
  std::vector<int64_t> offsets;
  for (const auto& v : *offset_arr) {
    offsets.emplace_back(v.value());
  }

  auto table = current_vector_reader_->ReadByOffsets(offsets);
  if (!table.ok()) {
    return arrow::Status::UnknownError(table.status().ToString());
  }
  // maybe copy here
  auto table_batch = table.value()->CombineChunksToBatch();
  if (!table_batch.ok()) {
    return table_batch.status();
  }

  std::vector<std::shared_ptr<arrow::Array>> columns(tmp_batch->columns().begin(), tmp_batch->columns().end());

  auto vector_col = table_batch.ValueOrDie()->GetColumnByName(space_.schema_->options()->vector_column);
  if (vector_col == nullptr) {
    return arrow::Status::UnknownError("vector column not found");
  }
  columns.emplace_back(vector_col);

  *batch = arrow::RecordBatch::Make(space_.schema_->schema(), tmp_batch->num_rows(), std::move(columns));
  return arrow::Status::OK();
}
}  // namespace milvus_storage