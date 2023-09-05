#include "multi_files_sequential_reader.h"
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <algorithm>
#include "common/arrow_util.h"
#include "common/macro.h"

namespace milvus_storage {

MultiFilesSequentialReader::MultiFilesSequentialReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                       const FragmentVector& fragments,
                                                       std::shared_ptr<arrow::Schema> schema,
                                                       std::shared_ptr<ReadOptions> options)
    : fs_(fs), schema_(std::move(schema)), options_(options) {
  for (const auto& fragment : fragments) {
    files_.insert(files_.end(), fragment.files().begin(), fragment.files().end());
  }
}

std::shared_ptr<arrow::Schema> MultiFilesSequentialReader::schema() const { return schema_; }

arrow::Status MultiFilesSequentialReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  while (true) {
    if (!curr_reader_) {
      if (next_pos_ >= files_.size()) {
        batch = nullptr;
        return arrow::Status::OK();
      }

      auto s = MakeArrowFileReader(fs_, files_[next_pos_++]);
      if (!s.ok()) {
        return arrow::Status::UnknownError(s.status().ToString());
      }
      holding_file_reader_ = s.value();

      auto s2 = MakeArrowRecordBatchReader(holding_file_reader_, options_);
      if (!s2.ok()) {
        return arrow::Status::UnknownError(s2.status().ToString());
      }
      curr_reader_ = s2.value();
    }

    std::shared_ptr<arrow::RecordBatch> tmp_batch;
    auto s = curr_reader_->ReadNext(&tmp_batch);
    if (!s.ok()) {
      return s;
    }

    if (tmp_batch == nullptr) {
      curr_reader_ = nullptr;
      holding_file_reader_ = nullptr;
      continue;
    }

    *batch = tmp_batch;
    return arrow::Status::OK();
  }
}

}  // namespace milvus_storage
