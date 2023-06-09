#include "reader/merge_record_reader.h"

namespace milvus_storage {

MergeRecordReader::MergeRecordReader(std::shared_ptr<ReadOptions>& options,
                                     const std::vector<std::string>& scalar_files,
                                     const std::vector<std::string>& vector_files,
                                     const DefaultSpace& space)
    : space_(space) {}

std::shared_ptr<arrow::Schema> MergeRecordReader::schema() const {
  // TODO: projection
  return space_.schema_->schema();
}

arrow::Status MergeRecordReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  while (true) {
    if (!curr_reader_) {
    }
  }
}
}  // namespace milvus_storage
