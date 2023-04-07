#include <arrow/type_fwd.h>
#include <parquet/exception.h>
#include <variant>

#include <iostream>
#include "arrow/array/array_primitive.h"
#include "format/parquet/file_reader.h"
#include "reader/scan_record_reader.h"
#include "storage/default_space.h"
#include "storage/deleteset.h"

std::shared_ptr<arrow::RecordBatch>
RecordBatchWithDeltedOffsets::Next() {
  while (next_pos_ < deleted_offsets_.size() && deleted_offsets_[next_pos_] == start_offset_) {
    next_pos_++;
    start_offset_++;
  }

  if (next_pos_ >= deleted_offsets_.size()) {
    std::shared_ptr<arrow::RecordBatch> res;
    if (start_offset_ != -1 && start_offset_ < batch_->num_rows()) {
      res = batch_->Slice(start_offset_);
    } else {
      res = nullptr;
    }
    start_offset_ = -1;
    return res;
  }

  auto res = batch_->Slice(start_offset_, deleted_offsets_[next_pos_] - start_offset_ - 1);
  start_offset_ = deleted_offsets_[next_pos_] + 1;
  next_pos_++;
  return res;
}

arrow::Status
CheckDeleteVisitor::Visit(const arrow::Int64Array& array) {
  for (int i = 0; i < array.length(); i++) {
    pk_type pk = array.Value(i);
    auto versions = delete_set_->GetVersionByPk(pk);
    auto version = version_col_->Value(i);
    for (auto& v : versions) {
      if (v >= version) {
        offsets_.emplace_back(i);
        break;
      }
    }
  }

  return arrow::Status::OK();
}

arrow::Status
CheckDeleteVisitor::Visit(const arrow::StringArray& array) {
  // FIXME: duplicated codes
  for (int i = 0; i < array.length(); i++) {
    pk_type pk = array.Value(i);
    auto versions = delete_set_->GetVersionByPk(pk);
    auto version = version_col_->Value(i);
    for (auto& v : versions) {
      if (v >= version) {
        offsets_.emplace_back(i);
        break;
      }
    }
  }

  return arrow::Status::OK();
}

ScanRecordReader::ScanRecordReader(std::shared_ptr<ReadOptions>& options,
                                   const std::vector<std::string>& files,
                                   const DefaultSpace& space)
    : space_(space), options_(options), files_(files) {
  // projection schema
}

std::shared_ptr<arrow::Schema>
ScanRecordReader::schema() const {
  // TODO
  return nullptr;
}

arrow::Status
ScanRecordReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
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
      PARQUET_THROW_NOT_OK(current_table_reader_->ReadNext(&table_batch));
      if (!table_batch) {
        PARQUET_THROW_NOT_OK(current_table_reader_->Close());
        current_table_reader_ = nullptr;
        continue;
      }

      auto pk_col = table_batch->GetColumnByName(space_.schema_->options()->primary_column);
      auto version_col = table_batch->GetColumnByName(space_.schema_->options()->version_column);
      CheckDeleteVisitor visitor(std::static_pointer_cast<arrow::Int64Array>(version_col), space_.delete_set_);
      PARQUET_THROW_NOT_OK(pk_col->Accept(&visitor));
      current_record_batch_ = std::make_shared<RecordBatchWithDeltedOffsets>(table_batch, visitor.offsets_);
      continue;
    }

    // create new table reader
    if (current_scanner_) {
      auto table = current_scanner_->Read();
      if (!table) {
        current_scanner_->Close();
        current_scanner_ = nullptr;
        continue;
      }

      current_table_reader_ = std::make_shared<arrow::TableBatchReader>(table);
      continue;
    }

    // create new scanner
    if (next_pos_ >= files_.size()) {
      return arrow::Status::OK();
    }
    auto file_reader = std::make_unique<ParquetFileReader>(space_.fs_.get(), files_[next_pos_++], options_);
    current_scanner_ = file_reader->NewScanner();
  }
}
