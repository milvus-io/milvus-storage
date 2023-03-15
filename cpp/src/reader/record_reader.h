#pragma once

#include <arrow/type_fwd.h>

#include <memory>

#include "default_space.h"
#include "scan_record_reader.h"
struct RecordReader {
  static std::unique_ptr<arrow::RecordBatchReader> GetRecordReader(
      const DefaultSpace &space, std::shared_ptr<ReadOption> &options) {
    return std::unique_ptr<arrow::RecordBatchReader>(new ScanRecordReader(
        options, space.manifest_->GetScalarFiles(), space));
  }
};