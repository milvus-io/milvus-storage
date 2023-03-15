#pragma once
#include <arrow/record_batch.h>

#include "default_space.h"
#include "options.h"
#include "scanner.h"

class FilterQueryRecordReader;
class ScanRecordReader : public arrow::RecordBatchReader {
  friend FilterQueryRecordReader;

 public:
  ScanRecordReader(std::shared_ptr<ReadOption> &options,
                   const std::vector<std::string> &files,
                   const DefaultSpace &space);
  std::shared_ptr<arrow::Schema> schema() const override;
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override;

 private:
  const DefaultSpace &space_;
  std::shared_ptr<ReadOption> options_;
  std::vector<std::string> files_;

  std::shared_ptr<Scanner> current_scanner_;
  int next_pos_ = 0;
};