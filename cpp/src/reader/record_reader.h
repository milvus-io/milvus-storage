#pragma once

#include <memory>

#include "../format/scanner.h"
#include "../storage/default_space.h"
#include "../storage/schema.h"
#include "arrow/record_batch.h"

class RecordReader : public arrow::RecordBatchReader {
 public:
  explicit RecordReader(const DefaultSpace &space,
                        std::shared_ptr<ReadOption> options);
  std::shared_ptr<arrow::Schema> schema() const override;
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override;

  arrow::Status ScanFiles(std::shared_ptr<arrow::RecordBatch> *batch,
                          bool is_scalar);

  arrow::Status ScanAndMerge(std::shared_ptr<arrow::RecordBatch> *batch);

 private:
  const DefaultSpace &space_;
  std::shared_ptr<ReadOption> options_;
  std::unique_ptr<Schema> schema_;
  std::unique_ptr<Scanner> current_scanner_;
  std::vector<std::string> scalar_files_;
  std::vector<std::string> vector_files_;
  int next_pos_;
};