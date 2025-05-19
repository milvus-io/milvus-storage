

#include "milvus-storage/reader/common/delete_reader.h"

namespace milvus_storage {
std::unique_ptr<DeleteMergeReader> DeleteMergeReader::Make(std::unique_ptr<arrow::RecordBatchReader> reader,
                                                           const SchemaOptions& schema_options,
                                                           const DeleteFragmentVector& delete_fragments,
                                                           const ReadOptions& options) {
  // DeleteFragmentVector filtered_delete_fragments;
  // for (auto& delete_fragment : delete_fragments) {
  //   if (schema_options->has_version_column() || delete_fragment.id() > fragment_id) {
  //     // If user declares the version column, we have to compare the version column to decide if the pk is deleted.
  //     // Or the fragment id can be used as the version column to filter previous delete fragments.
  //     filtered_delete_fragments.push_back(delete_fragment);
  //   }
  // }
  return std::make_unique<DeleteMergeReader>(std::move(reader), delete_fragments, schema_options, options);
}

std::shared_ptr<arrow::Schema> DeleteMergeReader::schema() const { return reader_->schema(); }

arrow::Status DeleteMergeReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  while (true) {
    if (filtered_batch_reader_) {
      auto b = filtered_batch_reader_->Next();
      if (b) {
        *batch = b;
        return arrow::Status::OK();
      }
      filtered_batch_reader_ = nullptr;
    }

    std::shared_ptr<arrow::RecordBatch> record_batch;
    RETURN_NOT_OK(reader_->ReadNext(&record_batch));
    if (!record_batch) {
      *batch = nullptr;
      return arrow::Status::OK();
    }

    if (schema_options_.has_version_column()) {
      auto version_col = record_batch->GetColumnByName(schema_options_.version_column);
      if (version_col == nullptr) {
        return arrow::Status::Invalid("Version column not found");
      }
      auto visitor = DeleteFilterVisitor(delete_fragments_, std::static_pointer_cast<arrow::Int64Array>(version_col),
                                         options_.version);

      auto pk_col = record_batch->GetColumnByName(schema_options_.primary_column);
      if (pk_col == nullptr) {
        return arrow::Status::Invalid("Primary column not found");
      }
      ARROW_RETURN_NOT_OK(pk_col->Accept(&visitor));
      filtered_batch_reader_ = std::make_shared<RecordBatchWithDeltedOffsets>(record_batch, visitor.GetOffsets());
    }
  }
}

std::shared_ptr<arrow::RecordBatch> DeleteMergeReader::RecordBatchWithDeltedOffsets::Next() {
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

  // zero-copy slice
  auto res = batch_->Slice(start_offset_, deleted_offsets_[next_pos_] - start_offset_ - 1);
  start_offset_ = deleted_offsets_[next_pos_] + 1;
  next_pos_++;
  return res;
}

arrow::Status DeleteMergeReader::DeleteFilterVisitor::Visit(const arrow::Int64Array& array) {
  return VisitTemplate(array);
}

arrow::Status DeleteMergeReader::DeleteFilterVisitor::Visit(const arrow::StringArray& array) {
  return VisitTemplate(array);
}
}  // namespace milvus_storage
