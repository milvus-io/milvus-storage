

#include "milvus-storage/reader/common/combine_reader.h"
#include <memory>
#include "arrow/type.h"
#include "milvus-storage/common/constants.h"
namespace milvus_storage {
std::unique_ptr<CombineReader> CombineReader::Make(std::unique_ptr<arrow::RecordBatchReader> scalar_reader,
                                                   std::unique_ptr<arrow::RecordBatchReader> vector_reader,
                                                   std::shared_ptr<Schema> schema) {
  assert(scalar_reader != nullptr && vector_reader != nullptr);
  return std::make_unique<CombineReader>(std::move(scalar_reader), std::move(vector_reader), schema);
}

std::shared_ptr<arrow::Schema> CombineReader::schema() const { return schema_->schema(); }

arrow::Status CombineReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  std::shared_ptr<arrow::RecordBatch> scalar_batch;
  std::shared_ptr<arrow::RecordBatch> vector_batch;
  ARROW_RETURN_NOT_OK(scalar_reader_->ReadNext(&scalar_batch));
  ARROW_RETURN_NOT_OK(vector_reader_->ReadNext(&vector_batch));
  if (scalar_batch == nullptr || vector_batch == nullptr) {
    batch = nullptr;
    return arrow::Status::OK();
  }

  for (int i = 0; i < scalar_batch->num_columns(); ++i) {
    if (scalar_batch->column_name(i) == kOffsetFieldName) {
      scalar_batch->RemoveColumn(i);
      break;
    }
  }

  assert(scalar_batch->num_rows() == vector_batch->num_rows());

  auto vec_column = vector_batch->GetColumnByName(schema_->options().vector_column);
  std::vector<std::shared_ptr<arrow::Array>> columns(scalar_batch->columns().begin(), scalar_batch->columns().end());

  auto vec_column_idx = schema_->schema()->GetFieldIndex(schema_->options().vector_column);
  columns.insert(columns.begin() + vec_column_idx, vec_column);

  *batch = arrow::RecordBatch::Make(schema(), scalar_batch->num_rows(), std::move(columns));
  return arrow::Status::OK();
}

}  // namespace milvus_storage
