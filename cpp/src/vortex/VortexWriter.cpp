#include "milvus-storage/vortex/VortexWriter.h"
#include "vortex.hpp"
#include <arrow/c/bridge.h>
#include <arrow/chunked_array.h>
#include <arrow/array.h>

namespace milvus_storage {

VortexWriter::VortexWriter(std::shared_ptr<arrow::fs::FileSystem> /*fs*/,
                           std::string base_path,
                           std::shared_ptr<arrow::Schema> schema,
                           WriteProperties properties)
    : path_(std::move(base_path)), schema_(std::move(schema)), properties_(std::move(properties)) {}

arrow::Status VortexWriter::write(const std::shared_ptr<arrow::RecordBatch>& batch) {
  ArrowArrayStream stream_reader{};

  const int ncolumns = static_cast<int>(schema_->num_fields());
  if (!batch->schema()->Equals(*schema_, false)) {
    return arrow::Status::Invalid("Schema", " was different: \n", schema_->ToString(), "\nvs\n",
                                  batch->schema()->ToString());
  }

  ::vortex::VortexWriteOptions write_options;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns(ncolumns);
  // must be struct, limit by arrow-ffi(for rust)
  assert(ncolumns == 1);
  for (int i = 0; i < ncolumns; ++i) {
    std::vector<std::shared_ptr<arrow::Array>> column_arrays(1);
    column_arrays[0] = batch->column(i);
    columns[i] = std::make_shared<arrow::ChunkedArray>(column_arrays, schema_->field(i)->type());

    // columns[i] = std::make_shared<arrow::ChunkedArray>(batch->column(i));
    ARROW_RETURN_NOT_OK(ExportChunkedArray(columns[i], &stream_reader));
    std::string copy_path = path_;
    write_options.WriteArrayStream(stream_reader, copy_path);
  }

  return arrow::Status::OK();
}

arrow::Status VortexWriter::flush() {
  // NO implements
  return arrow::Status::OK();
}

void VortexWriter::close() {
  // NO implements
}

}  // namespace milvus_storage