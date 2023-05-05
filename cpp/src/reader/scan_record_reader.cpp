#include "arrow/array/array_primitive.h"
#include "reader/scan_record_reader.h"
#include "storage/deleteset.h"
#include "arrow/array/array_binary.h"
namespace milvus_storage {


ScanRecordReader::ScanRecordReader(std::shared_ptr<ReadOptions>& options,
                                   const std::vector<std::string>& files,
                                   const DefaultSpace& space)
    : space_(space), options_(options), files_(files) {
  // projection schema
}

std::shared_ptr<arrow::Schema> ScanRecordReader::schema() const {
  // TODO
  return nullptr;
}

arrow::Status ScanRecordReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  while (true) {
    if (current_record_batch_) {
      auto rec_batch = current_record_batch_->Next();
      if (!rec_batch) {
        current_record_batch_ = nullptr;
        continue;
      }
      *batch = rec_batch;
      return arrow::Status::OK();
    }

    // create new record batch
    if (current_table_reader_) {
      std::shared_ptr<arrow::RecordBatch> table_batch;
      ARROW_RETURN_NOT_OK(current_table_reader_->ReadNext(&table_batch));
      if (!table_batch) {
        ARROW_RETURN_NOT_OK(current_table_reader_->Close());
        current_table_reader_ = nullptr;
        continue;
      }

      auto pk_col = table_batch->GetColumnByName(space_.schema_->options()->primary_column);
      auto version_col = table_batch->GetColumnByName(space_.schema_->options()->version_column);
      CheckDeleteVisitor visitor(std::static_pointer_cast<arrow::Int64Array>(version_col), space_.delete_set_);
      ARROW_RETURN_NOT_OK(pk_col->Accept(&visitor));
      current_record_batch_ = std::make_shared<RecordBatchWithDeltedOffsets>(table_batch, visitor.offsets_);
      continue;
    }

    // create new table reader
    if (current_scanner_) {
      auto table = current_scanner_->Read();
      if (!table.ok()) {
        return arrow::Status::UnknownError(table.status().ToString());
      }
      if (!table.value()) {
        current_scanner_->Close();
        current_scanner_ = nullptr;
        continue;
      }

      current_table_reader_ = std::make_shared<arrow::TableBatchReader>(table.value());
      continue;
    }

    // create new scanner
    if (next_pos_ >= files_.size()) {
      return arrow::Status::OK();
    }
    auto file_reader = std::make_unique<ParquetFileReader>(space_.fs_, files_[next_pos_++], options_);
    auto s = file_reader->Init();
    if (!s.ok()) {
      return arrow::Status::UnknownError(s.ToString());
    }
    current_scanner_ = file_reader->NewScanner();
    s = current_scanner_->Init();
    if (!s.ok()) {
      return arrow::Status::UnknownError(s.ToString());
    }
  }
}
}  // namespace milvus_storage