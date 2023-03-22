#pragma once
#include <arrow/record_batch.h>
#include <memory>

#include "default_space.h"
#include "deleteset.h"
#include "format/parquet/file_reader.h"
#include "format/scanner.h"
#include "options.h"

class RecordBatchWithDeltedOffsets {
 public:
  RecordBatchWithDeltedOffsets(std::shared_ptr<arrow::RecordBatch> batch, std::vector<int> deleted_offsets)
      : batch_(std::move(batch)), deleted_offsets_(std::move(deleted_offsets)) {}

  std::shared_ptr<arrow::RecordBatch> Next();

 private:
  std::shared_ptr<arrow::RecordBatch> batch_;
  std::vector<int> deleted_offsets_;
  int next_pos_ = 0;
  int start_offset_ = 0;
};

class FilterQueryRecordReader;
class ScanRecordReader : public arrow::RecordBatchReader {
  friend FilterQueryRecordReader;

 public:
  ScanRecordReader(std::shared_ptr<ReadOption> &options, const std::vector<std::string> &files,
                   const DefaultSpace &space);
  std::shared_ptr<arrow::Schema> schema() const override;
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override;

 private:
  const DefaultSpace &space_;
  std::shared_ptr<ReadOption> options_;
  std::vector<std::string> files_;

  std::shared_ptr<ParquetFileReader> current_reader_;
  std::shared_ptr<Scanner> current_scanner_;
  std::shared_ptr<RecordBatchWithDeltedOffsets> current_record_batch_;

  int next_pos_ = 0;
};

class CheckDeleteVisitor : public arrow::ArrayVisitor {
 public:
  CheckDeleteVisitor(std::shared_ptr<arrow::Int64Array> version_col, std::shared_ptr<DeleteSet> delete_set)
      : version_col_(std::move(version_col)), delete_set_(std::move(delete_set)){};

  arrow::Status Visit(const arrow::Int64Array &array) override;
  arrow::Status Visit(const arrow::StringArray &array) override;

 private:
  std::shared_ptr<arrow::Int64Array> version_col_;
  std::shared_ptr<DeleteSet> delete_set_;
  std::vector<int> offsets_;

  friend ScanRecordReader;
};
