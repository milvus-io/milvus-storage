

#pragma once
#include <memory>
#include "parquet/arrow/reader.h"
#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/common/result.h"
#include "milvus-storage/storage/options.h"

namespace milvus_storage {
Result<std::unique_ptr<parquet::arrow::FileReader>> MakeArrowFileReader(arrow::fs::FileSystem& fs,
                                                                        const std::string& file_path);

Result<std::unique_ptr<parquet::arrow::FileReader>> MakeArrowFileReader(
    arrow::fs::FileSystem& fs, const std::string& file_path, const parquet::ReaderProperties& read_properties);

Result<std::unique_ptr<arrow::RecordBatchReader>> MakeArrowRecordBatchReader(parquet::arrow::FileReader& reader,
                                                                             std::shared_ptr<arrow::Schema> schema,
                                                                             const SchemaOptions& schema_options,
                                                                             const ReadOptions& options = {});

size_t GetRecordBatchMemorySize(const std::shared_ptr<arrow::RecordBatch>& record_batch);

size_t GetArrowArrayMemorySize(const std::shared_ptr<arrow::Array>& array);

size_t GetTableMemorySize(const std::shared_ptr<arrow::Table>& table);

}  // namespace milvus_storage
