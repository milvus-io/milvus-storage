

#pragma once

#include <sstream>
#include <parquet/properties.h>

using namespace std;

namespace milvus_storage {

static constexpr int64_t DEFAULT_MAX_ROW_GROUP_SIZE = 1024 * 1024;  // 1 MB

// https://github.com/apache/arrow/blob/6b268f62a8a172249ef35f093009c740c32e1f36/cpp/src/arrow/filesystem/s3fs.cc#L1596
static constexpr int64_t ARROW_PART_UPLOAD_SIZE = 10 * 1024 * 1024;  // 10 MB

static constexpr int64_t MIN_BUFFER_SIZE_PER_FILE = DEFAULT_MAX_ROW_GROUP_SIZE + ARROW_PART_UPLOAD_SIZE;

// Default number of rows to read when using ::arrow::RecordBatchReader
static constexpr int64_t DEFAULT_READ_BATCH_SIZE = 1024;
static constexpr int64_t DEFAULT_READ_BUFFER_SIZE = 16 * 1024 * 1024;   // 16 MB
static constexpr int64_t DEFAULT_WRITE_BUFFER_SIZE = 16 * 1024 * 1024;  // 16 MB

struct StorageConfig {
  int64_t part_size = 0;
};

}  // namespace milvus_storage