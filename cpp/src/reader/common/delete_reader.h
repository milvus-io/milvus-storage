#pragma once

#include <utility>
#include "arrow/record_batch.h"
#include "arrow/array/array_primitive.h"
#include "file/delete_fragment.h"
#include "storage/options.h"
#include "arrow/visitor.h"

namespace milvus_storage {

// DeleteMergeReader filters the deleted record.
class DeleteMergeReader : public arrow::RecordBatchReader {
  public:
  class RecordBatchWithDeltedOffsets;
  class DeleteFilterVisitor;

  static std::shared_ptr<DeleteMergeReader> Make(std::shared_ptr<arrow::RecordBatchReader> reader,
                                                 std::shared_ptr<SchemaOptions> schema_options,
                                                 const DeleteFragmentVector& delete_fragments);
  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  DeleteMergeReader(std::shared_ptr<arrow::RecordBatchReader> reader,
                    DeleteFragmentVector delete_fragments,
                    std::shared_ptr<SchemaOptions> schema_options)
      : reader_(std::move(reader)),
        delete_fragments_(std::move(delete_fragments)),
        schema_options_(std::move(schema_options)) {}

  private:
  std::shared_ptr<arrow::RecordBatchReader> reader_;
  std::shared_ptr<RecordBatchWithDeltedOffsets> filtered_batch_reader_;
  DeleteFragmentVector delete_fragments_;
  std::shared_ptr<SchemaOptions> schema_options_;
};

// RecordBatchWithDeltedOffsets is reader helper to fetch records not deleted without copy
class DeleteMergeReader::RecordBatchWithDeltedOffsets {
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

class DeleteMergeReader::DeleteFilterVisitor : public arrow::ArrayVisitor {
  public:
  explicit DeleteFilterVisitor(DeleteFragmentVector delete_fragments,
                               std::shared_ptr<arrow::Int64Array> version_col = nullptr)
      : version_col_(std::move(version_col)), delete_fragments_(std::move(delete_fragments)){};

  arrow::Status Visit(const arrow::Int64Array& array) override;
  arrow::Status Visit(const arrow::StringArray& array) override;

  std::vector<int> GetOffsets() { return offsets_; }

  private:
  template <typename T>
  arrow::Status VisitTemplate(const T& array) {
    for (int i = 0; i < array.length(); i++) {
      pk_type pk = array.Value(i);
      for (auto& delete_fragment : delete_fragments_) {
        if (version_col_ != nullptr) {
          if (delete_fragment.Filter(pk, version_col_->Value(i))) {
            offsets_.push_back(i);
            break;
          }
        } else {
          if (delete_fragment.Filter(pk)) {
            offsets_.push_back(i);
            break;
          }
        }
      }
    }

    return arrow::Status::OK();
  }

  std::shared_ptr<arrow::Int64Array> version_col_;
  DeleteFragmentVector delete_fragments_;
  std::vector<int> offsets_;
};
}  // namespace milvus_storage
