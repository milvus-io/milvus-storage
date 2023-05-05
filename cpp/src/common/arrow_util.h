#pragma once
#include "parquet/arrow/reader.h"
#include "arrow/filesystem/filesystem.h"
#include "common/result.h"
#include "storage/options.h"

namespace milvus_storage {
Result<std::shared_ptr<parquet::arrow::FileReader>> MakeArrowFileReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                        std::string& file_path);

Result<std::shared_ptr<arrow::RecordBatchReader>> MakeArrowRecordBatchReader(
    std::shared_ptr<parquet::arrow::FileReader> reader, ReadOptions options);
}  // namespace milvus_storage