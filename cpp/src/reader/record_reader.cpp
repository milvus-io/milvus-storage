#include "reader/record_reader.h"
#include <arrow/filesystem/filesystem.h>
#include <cstdint>
#include <memory>
#include "common/macro.h"
#include "file/delete_fragment.h"
#include "file/fragment.h"
#include "reader/common/combine_reader.h"
#include "reader/filter_query_record_reader.h"
#include "reader/merge_record_reader.h"
#include "reader/scan_record_reader.h"
#include "common/utils.h"

namespace milvus_storage {
DeleteFragmentVector FilterDeleteFragments(FragmentVector& data_fragments, DeleteFragmentVector& delete_fragments) {
  int64_t minid = INT64_MAX;
  for (const auto& fragment : data_fragments) {
    if (fragment.id() < minid) {
      minid = fragment.id();
    }
  }

  DeleteFragmentVector res;
  for (const auto& fragment : delete_fragments) {
    if (fragment.id() >= minid) {
      res.push_back(fragment);
    }
  }
  return res;
}

std::unique_ptr<arrow::RecordBatchReader> RecordReader::MakeRecordReader(std::shared_ptr<Manifest> manifest,
                                                                         std::shared_ptr<Schema> schema,
                                                                         std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                         DeleteFragmentVector delete_fragments,
                                                                         std::shared_ptr<ReadOptions>& options) {
  // TODO: Implement a common optimization method. For now we just enumerate few plans.
  std::set<std::string> related_columns;
  for (auto& column : options->columns) {
    related_columns.insert(column);
  }
  for (auto& filter : options->filters) {
    related_columns.insert(filter->get_column_name());
  }

  auto scalar_data = manifest->scalar_fragments(), vector_data = manifest->vector_fragments();
  if (bool only_scalar, only_vector; (only_scalar = only_contain_scalar_columns(schema, related_columns)) ||
                                     (only_vector = only_contain_vector_columns(schema, related_columns))) {
    auto data_fragments = only_scalar ? scalar_data : vector_data;
    return std::make_unique<ScanRecordReader>(schema, options, fs, data_fragments, delete_fragments);
  }

  if (filters_only_contain_pk_and_version(schema, options->filters)) {
    return std::make_unique<MergeRecordReader>(options, scalar_data, vector_data, delete_fragments, fs, schema);
  }
  return std::make_unique<FilterQueryRecordReader>(options, scalar_data, vector_data, delete_fragments, fs, schema);
}

bool RecordReader::only_contain_scalar_columns(const std::shared_ptr<Schema> schema,
                                               const std::set<std::string>& related_columns) {
  for (auto& column : related_columns) {
    if (schema->options()->vector_column == column) {
      return false;
    }
  }
  return true;
}

bool RecordReader::only_contain_vector_columns(const std::shared_ptr<Schema> schema,
                                               const std::set<std::string>& related_columns) {
  for (auto& column : related_columns) {
    if (schema->options()->vector_column != column && schema->options()->primary_column != column &&
        schema->options()->version_column != column) {
      return false;
    }
  }
  return true;
}

bool RecordReader::filters_only_contain_pk_and_version(std::shared_ptr<Schema> schema,
                                                       const std::vector<std::unique_ptr<Filter>>& filters) {
  for (auto& filter : filters) {
    if (filter->get_column_name() != schema->options()->primary_column &&
        filter->get_column_name() != schema->options()->version_column) {
      return false;
    }
  }
  return true;
}

Result<std::shared_ptr<arrow::RecordBatchReader>> RecordReader::MakeScanDataReader(
    std::shared_ptr<Manifest> manifest, std::shared_ptr<arrow::fs::FileSystem> fs) {
  auto scalar_reader = std::make_shared<MultiFilesSequentialReader>(
      fs, manifest->scalar_fragments(), manifest->schema()->scalar_schema(), ReadOptions::default_read_options());
  auto vector_reader = std::make_shared<MultiFilesSequentialReader>(
      fs, manifest->vector_fragments(), manifest->schema()->vector_schema(), ReadOptions::default_read_options());

  ASSIGN_OR_RETURN_NOT_OK(auto combine_reader, CombineReader::Make(scalar_reader, vector_reader, manifest->schema()));
  return std::static_pointer_cast<arrow::RecordBatchReader>(combine_reader);
}

std::shared_ptr<arrow::RecordBatchReader> RecordReader::MakeScanDeleteReader(
    std::shared_ptr<Manifest> manifest, std::shared_ptr<arrow::fs::FileSystem> fs) {
  auto reader = std::make_shared<MultiFilesSequentialReader>(
      fs, manifest->delete_fragments(), manifest->schema()->delete_schema(), ReadOptions::default_read_options());
  return reader;
}
}  // namespace milvus_storage
