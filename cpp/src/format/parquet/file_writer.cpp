#include "file_writer.h"

#include <arrow/type_fwd.h>

#include "../../exception.h"
#include "parquet/arrow/writer.h"

ParquetFileWriter::ParquetFileWriter(arrow::Schema *schema,
                                     arrow::fs::FileSystem *fs,
                                     std::string &file_path) {
  auto file = fs->OpenOutputStream(file_path);
  if (!file.ok()) {
    throw StorageException("open file failed");
  }
  auto sink = file.ValueOrDie();
  auto res = parquet::arrow::FileWriter::Open(
      *schema, arrow::default_memory_pool(), sink);
  if (!res.ok()) {
    throw StorageException("open file writer failed");
  }
  writer_ = std::move(res.ValueOrDie());
}

void ParquetFileWriter::Write(arrow::RecordBatch *record) {
  auto res = writer_->WriteRecordBatch(*record);
  if (!res.ok()) {
    throw StorageException("write record failed");
  }
  count_ += record->num_columns();
}

int64_t ParquetFileWriter::count() { return count_; }

void ParquetFileWriter::Close() {
  if (!writer_->Close().ok()) {
    throw StorageException("close file writer failed");
  }
}
