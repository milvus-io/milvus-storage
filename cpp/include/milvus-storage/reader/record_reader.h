

#pragma once

#include "milvus-storage/file/delete_fragment.h"
#include "milvus-storage/storage/manifest.h"
namespace milvus_storage {

namespace internal {
std::unique_ptr<arrow::RecordBatchReader> MakeRecordReader(std::shared_ptr<Manifest> manifest,
                                                           std::shared_ptr<Schema> schema,
                                                           arrow::fs::FileSystem& fs,
                                                           const DeleteFragmentVector& delete_fragments,
                                                           const ReadOptions& options);

bool only_contain_scalar_columns(std::shared_ptr<Schema> schema, const std::set<std::string>& related_columns);

bool only_contain_vector_columns(std::shared_ptr<Schema> schema, const std::set<std::string>& related_columns);

bool filters_only_contain_pk_and_version(std::shared_ptr<Schema> schema, const Filter::FilterSet& filters);

std::unique_ptr<arrow::RecordBatchReader> MakeScanDataReader(std::shared_ptr<Manifest> manifest,
                                                             arrow::fs::FileSystem& fs,
                                                             const ReadOptions& options = {});

std::unique_ptr<arrow::RecordBatchReader> MakeScanDeleteReader(std::shared_ptr<Manifest> manifest,
                                                               arrow::fs::FileSystem& fs);
}  // namespace internal
}  // namespace milvus_storage
