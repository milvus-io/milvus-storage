#pragma once

#include <arrow/type_fwd.h>

#include "../storage/default_space.h"
struct RecordReader {
  static std::unique_ptr<arrow::RecordBatchReader> GetRecordReader(
      DefaultSpace *space, SpaceOption *options) {
    // 1. check if filters and projection only relate to scalar or vector
    // data.
    // then use ScanRecordReader to scan only scalar/vector files.
    // 2. check if filters are empty or always true, and use MergeRecordReader
    // to scan and merge.
    // 3. use FilterQueryRecordReader to scan scalar data and read by offsets
    // to
    // get vector data.
  }
};