#pragma once

#include "reader/filter_query_record_reader.h"
namespace milvus_storage {

struct RecordReader {
  static std::unique_ptr<arrow::RecordBatchReader> GetRecordReader(const DefaultSpace& space,
                                                                   std::shared_ptr<ReadOptions>& options);

  static bool only_contain_scalar_columns(const DefaultSpace& space, const std::set<std::string>& related_columns);

  static bool only_contain_vector_columns(const DefaultSpace& space, const std::set<std::string>& related_columns);

  static bool filters_only_contain_pk_and_version(const DefaultSpace& space, const std::vector<Filter*>& filters);
};
}  // namespace milvus_storage