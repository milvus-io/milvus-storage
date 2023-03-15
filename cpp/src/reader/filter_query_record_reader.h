#pragma once
#include <memory>

#include "arrow/record_batch.h"
#include "default_space.h"
#include "parquet-format//file_reader.h"
#include "parquet-format/file_scanner.h"
#include "scanner.h"
#include "schema.h"

class FilterQueryRecordReader : public arrow::RecordBatchReader {
 public:
  FilterQueryRecordReader(std::shared_ptr<ReadOption> &options,
                          std::vector<std::string> &scalar_files,
                          std::vector<std::string> &vector_files,
                          const DefaultSpace &space);
  std::shared_ptr<arrow::Schema> schema() const override;
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override;

 private:
  const DefaultSpace &space_;
  std::shared_ptr<ReadOption> options_;
  std::unique_ptr<Schema> schema_;

  std::vector<std::string> vector_files_;

  int next_pos_ = 0;

  std::unique_ptr<ScanRecordReader> scalar_reader_;
  std::unique_ptr<ParquetFileReader> current_vector_reader_;
};