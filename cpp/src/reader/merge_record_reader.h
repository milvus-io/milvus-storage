#pragma once

#include <memory>

#include "arrow/record_batch.h"
#include "storage/default_space.h"
#include "format/scanner.h"

class MergeRecordReader : public arrow::RecordBatchReader {
  public:
  explicit MergeRecordReader(std::shared_ptr<ReadOption>& options,
                             const std::vector<std::string>& scalar_files,
                             const std::vector<std::string>& vector_files,
                             const DefaultSpace& space);
  std::shared_ptr<arrow::Schema>
  schema() const override;
  arrow::Status
  ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  const DefaultSpace& space_;

  std::unique_ptr<ScanRecordReader> scalar_reader_;
  std::unique_ptr<ScanRecordReader> vector_reader_;
};
