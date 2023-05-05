#pragma once

#include "format/parquet/file_reader.h"
#include "storage/default_space.h"
namespace milvus_storage {

// ScanRecordReader is used to read record batch from parquet files
class ScanRecordReader : public arrow::RecordBatchReader {
  friend FilterQueryRecordReader;

  public:
  ScanRecordReader(std::shared_ptr<ReadOptions>& options,
                   const std::vector<std::string>& files,
                   const DefaultSpace& space);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  const DefaultSpace& space_;
  std::shared_ptr<ReadOptions> options_;
  std::vector<std::string> files_;

  std::shared_ptr<ParquetFileScanner> current_scanner_;
  std::shared_ptr<arrow::TableBatchReader> current_table_reader_;
  std::shared_ptr<RecordBatchWithDeltedOffsets> current_record_batch_;

  int next_pos_ = 0;
};

}  // namespace milvus_storage