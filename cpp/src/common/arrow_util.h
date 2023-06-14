#pragma once
#include "parquet/arrow/reader.h"
#include "arrow/filesystem/filesystem.h"
#include "common/result.h"
#include "storage/options.h"

namespace milvus_storage {
Result<std::shared_ptr<parquet::arrow::FileReader>> MakeArrowFileReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                        std::string& file_path);

Result<std::shared_ptr<arrow::RecordBatchReader>> MakeArrowRecordBatchReader(
    std::shared_ptr<parquet::arrow::FileReader> reader,
    const ReadOptions& options = ReadOptions::default_read_options());
}  // namespace milvus_storage
