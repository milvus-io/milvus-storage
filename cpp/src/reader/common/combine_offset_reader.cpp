#include "reader/common/combine_offset_reader.h"
#include "arrow/array/array_primitive.h"
namespace milvus_storage {
Result<std::shared_ptr<CombineOffsetReader>> CombineOffsetReader::Make(
    std::shared_ptr<arrow::RecordBatchReader> scalar_reader,
    std::shared_ptr<ParquetFileReader> vector_reader,
    std::shared_ptr<Schema> schema) {
  return std::shared_ptr<CombineOffsetReader>(
      new CombineOffsetReader(std::move(scalar_reader), std::move(vector_reader), std::move(schema)));
}

std::shared_ptr<arrow::Schema> CombineOffsetReader::schema() const { return schema_->schema(); }

arrow::Status CombineOffsetReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  std::shared_ptr<arrow::RecordBatch> scalar_batch;
  ARROW_RETURN_NOT_OK(scalar_reader_->ReadNext(&scalar_batch));

  if (!scalar_batch) {
    return arrow::Status::OK();
  }

  auto col_arr = scalar_batch->GetColumnByName(kOffsetFieldName);
  if (!col_arr) {
    return arrow::Status::UnknownError("offset column not found");
  }
  auto offset_arr = std::dynamic_pointer_cast<arrow::Int64Array>(col_arr);
  // TODO: no need to copy
  std::vector<int64_t> offsets;
  for (const auto& v : *offset_arr) {
    offsets.emplace_back(v.value());
  }

  auto table = vector_reader_->ReadByOffsets(offsets);
  if (!table.ok()) {
    return arrow::Status::UnknownError(table.status().ToString());
  }
  // maybe copy here
  auto table_batch = table.value()->CombineChunksToBatch();
  if (!table_batch.ok()) {
    return table_batch.status();
  }

  std::vector<std::shared_ptr<arrow::Array>> columns(scalar_batch->columns().begin(), scalar_batch->columns().end());

  auto vector_col = table_batch.ValueOrDie()->GetColumnByName(schema_->options()->vector_column);
  if (!vector_col) {
    return arrow::Status::UnknownError("vector column not found");
  }
  columns.emplace_back(vector_col);

  *batch = arrow::RecordBatch::Make(schema_->schema(), scalar_batch->num_rows(), std::move(columns));
  return arrow::Status::OK();
}

}  // namespace milvus_storage
