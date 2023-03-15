#include "merge_record_reader.h"

#include <arrow/record_batch.h>
#include <arrow/status.h>

#include <memory>

#include "format/parquet/file_reader.h"

MergeRecordReader::MergeRecordReader(std::shared_ptr<ReadOption> &options,
                                     std::vector<std::string> &scalar_files,
                                     std::vector<std::string> &vector_files,
                                     const DefaultSpace &space) {
  // TODO: init schema
}

std::shared_ptr<arrow::Schema> MergeRecordReader::schema() const {
  // TODO: projection
}

arrow::Status MergeRecordReader::ReadNext(
    std::shared_ptr<arrow::RecordBatch> *batch) {}
