#include "record_reader.h"

#include <arrow/record_batch.h>
#include <arrow/status.h>

#include <memory>

#include "../format/parquet/file_reader.h"

RecordReader::RecordReader(const DefaultSpace &space,
                           std::shared_ptr<ReadOption> options)
    : space_(space), options_(options) {
  // TODO: init schema
  scalar_files_ = space_.manifest_->GetScalarFiles();
  vector_files_ = space_.manifest_->GetVectorFiles();
}

std::shared_ptr<arrow::Schema> RecordReader::schema() const {
  return space_.manifest_->get_schema();
  // TODO: projection
}

arrow::Status RecordReader::ReadNext(
    std::shared_ptr<arrow::RecordBatch> *batch) {
  if (schema_->is_scalar_schema(space_.options_.get())) {
    return ScanFiles(batch, true);
  } else if (schema_->is_vector_schema(space_.options_.get())) {
    return ScanFiles(batch, false);
  } else {
    return ScanAndMerge(batch);
  }
}

arrow::Status RecordReader::ScanFiles(
    std::shared_ptr<arrow::RecordBatch> *batch, bool is_scalar) {
  std::vector<std::string> &files = is_scalar ? scalar_files_ : vector_files_;
  while (true) {
    if (current_scanner_ == nullptr) {
      if (next_pos_ >= files.size()) {
        return arrow::Status::OK();
      }

      auto reader = std::make_unique<ParquetFileReader>(
          space_.fs_.get(), files[next_pos_++], options_);

      current_scanner_ = reader->NewScanner();
    }

    auto res_batch = current_scanner_->Read();
    if (res_batch == nullptr) {
      current_scanner_->Close();
      current_scanner_ = nullptr;
      continue;
    }

    *batch = res_batch;
    return arrow::Status::OK();
  }
}

arrow::Status RecordReader::ScanAndMerge(
    std::shared_ptr<arrow::RecordBatch> *batch) {
  std::shared_ptr<arrow::RecordBatch> temp_batch;
  auto status = ScanFiles(&temp_batch, true);
  if (!status.ok()) {
    return status;
  }

  int file_idx = next_pos_ - 1;
  
}