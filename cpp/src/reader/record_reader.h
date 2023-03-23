#pragma once

#include <memory>
#include <arrow/type_fwd.h>

#include "reader/scan_record_reader.h"
#include "storage/default_space.h"

struct RecordReader {
  static std::unique_ptr<arrow::RecordBatchReader>
  GetRecordReader(const DefaultSpace& space, std::shared_ptr<ReadOption>& options) {
    return std::unique_ptr<arrow::RecordBatchReader>(
        new ScanRecordReader(options, space.manifest_->GetScalarFiles(), space));
  }
};
