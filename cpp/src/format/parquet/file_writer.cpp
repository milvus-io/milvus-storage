#include "common/macro.h"
#include "format/parquet/file_writer.h"
namespace milvus_storage {

ParquetFileWriter::ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                     std::string& file_path)
    : schema_(std::move(schema)), fs_(std::move(fs)), file_path_(file_path) {}

Status ParquetFileWriter::Init() {
  auto coln = schema_->num_fields();
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto sink, fs_->OpenOutputStream(file_path_));
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto writer,
                                parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(), sink));

  writer_ = std::move(writer);
  return Status::OK();
}

Status ParquetFileWriter::Write(arrow::RecordBatch* record) {
  RETURN_ARROW_NOT_OK(writer_->WriteRecordBatch(*record));
  count_ += record->num_rows();
  return Status::OK();
}

int64_t ParquetFileWriter::count() { return count_; }

Status ParquetFileWriter::Close() {
  RETURN_ARROW_NOT_OK(writer_->Close());
  return Status::OK();
}
}  // namespace milvus_storage
