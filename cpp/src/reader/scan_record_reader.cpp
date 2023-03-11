#include "scan_record_reader.h"

#include "../format/parquet/file_reader.h"
ScanRecordReader::ScanRecordReader(std::shared_ptr<ReadOption> &options,
                                   std::vector<std::string> &files,
                                   const DefaultSpace &space)
    : space_(space), options_(options), files_(files) {
  // projection schema
}

std::shared_ptr<arrow::Schema> ScanRecordReader::schema() const {
  // TODO
  return nullptr;
}

arrow::Status ScanRecordReader::ReadNext(
    std::shared_ptr<arrow::RecordBatch> *batch) {
  while (true) {
    if (current_scanner_ == nullptr) {
      if (next_pos_ >= files_.size()) {
        return arrow::Status::OK();
      }

      auto reader = std::make_unique<ParquetFileReader>(
          space_.fs_.get(), files_[next_pos_++], options_);

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