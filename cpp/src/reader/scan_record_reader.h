#pragma once
#include <arrow/record_batch.h>

#include "../format/scanner.h"
#include "../options/options.h"
#include "../storage/default_space.h"

class FilterQueryRecordReader;
class ScanRecordReader : public arrow::RecordBatchReader {
  friend FilterQueryRecordReader;

 public:
  ScanRecordReader(std::shared_ptr<ReadOption> &options,
                   std::vector<std::string> &files, const DefaultSpace &space);
  std::shared_ptr<arrow::Schema> schema() const override;
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override;

 private:
  const DefaultSpace &space_;
  std::shared_ptr<ReadOption> options_;
  std::vector<std::string> files_;

  std::unique_ptr<Scanner> current_scanner_;
  int next_pos_ = 0;
};