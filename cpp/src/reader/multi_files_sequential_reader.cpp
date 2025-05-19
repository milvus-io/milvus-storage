

#include "milvus-storage/reader/multi_files_sequential_reader.h"
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include "milvus-storage/common/arrow_util.h"

namespace milvus_storage {

MultiFilesSequentialReader::MultiFilesSequentialReader(arrow::fs::FileSystem& fs,
                                                       const FragmentVector& fragments,
                                                       std::shared_ptr<arrow::Schema> schema,
                                                       const SchemaOptions& schema_options,
                                                       const ReadOptions& options)
    : fs_(fs), schema_(std::move(schema)), schema_options_(schema_options), options_(options) {
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
      holding_file_reader_ = std::move(s.value());

      auto s2 = MakeArrowRecordBatchReader(*holding_file_reader_, schema_, schema_options_, options_);
      if (!s2.ok()) {
        return arrow::Status::UnknownError(s2.status().ToString());
      }
      curr_reader_ = std::move(s2.value());
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
