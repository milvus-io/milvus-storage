

#include "test_util.h"
#include <arrow/type_fwd.h>
#include "milvus-storage/format/parquet/file_writer.h"
#include "arrow/array/builder_primitive.h"
#include "milvus-storage/common/config.h"
namespace milvus_storage {
std::shared_ptr<arrow::Schema> CreateArrowSchema(std::vector<std::string> field_names,
                                                 std::vector<std::shared_ptr<arrow::DataType>> field_types) {
  arrow::FieldVector fields;
  for (int i = 0; i < field_names.size(); i++) {
    fields.push_back(arrow::field(field_names[i], field_types[i]));
  }
  return std::make_shared<arrow::Schema>(fields);
}

Status PrepareSimpleParquetFile(std::shared_ptr<arrow::Schema> schema,
                                std::shared_ptr<arrow::fs::FileSystem> fs,
                                const std::string& file_path,
                                int num_rows) {
  // TODO: parse schema and generate data
  auto conf = StorageConfig();
  ParquetFileWriter w(schema, fs, file_path, conf);
  w.Init();
  arrow::Int64Builder builder;
  for (int i = 0; i < num_rows; i++) {
    RETURN_ARROW_NOT_OK(builder.Append(i));
  }
  std::shared_ptr<arrow::Array> array;
  RETURN_ARROW_NOT_OK(builder.Finish(&array));
  auto batch = arrow::RecordBatch::Make(schema, num_rows, {array});
  RETURN_NOT_OK(w.Write(*batch));
  return w.Close();
}
}  // namespace milvus_storage
