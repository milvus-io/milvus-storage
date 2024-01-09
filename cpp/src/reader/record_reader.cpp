// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "reader/record_reader.h"
#include <arrow/filesystem/filesystem.h>
#include <cstdint>
#include <memory>
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

namespace internal {

std::unique_ptr<arrow::RecordBatchReader> MakeRecordReader(std::shared_ptr<Manifest> manifest,
                                                           std::shared_ptr<Schema> schema,
                                                           arrow::fs::FileSystem& fs,
                                                           const DeleteFragmentVector& delete_fragments,
                                                           const ReadOptions& options) {
  // TODO: Implement a common optimization method. For now we just enumerate few plans.
  std::set<std::string> related_columns;
  for (auto& column : options.columns) {
    related_columns.insert(column);
  }
  for (auto& filter : options.filters) {
    related_columns.insert(filter->get_column_name());
  }

  auto scalar_data = manifest->scalar_fragments(), vector_data = manifest->vector_fragments();
  if (bool only_scalar, only_vector; (only_scalar = only_contain_scalar_columns(schema, related_columns)) ||
                                     (only_vector = only_contain_vector_columns(schema, related_columns))) {
    auto data_fragments = only_scalar ? scalar_data : vector_data;
    auto scan_schema = only_scalar ? schema->scalar_schema() : schema->vector_schema();
    return std::make_unique<ScanRecordReader>(scan_schema, schema->options(), options, fs, data_fragments,
                                              delete_fragments);
  }

  if (filters_only_contain_pk_and_version(schema, options.filters)) {
    return std::make_unique<MergeRecordReader>(options, scalar_data, vector_data, delete_fragments, fs, schema);
  }
  return std::make_unique<FilterQueryRecordReader>(options, scalar_data, vector_data, delete_fragments, fs, schema);
}

bool only_contain_scalar_columns(const std::shared_ptr<Schema> schema, const std::set<std::string>& related_columns) {
  for (auto& column : related_columns) {
    if (schema->options().vector_column == column) {
      return false;
    }
  }
  return true;
}

bool only_contain_vector_columns(const std::shared_ptr<Schema> schema, const std::set<std::string>& related_columns) {
  for (auto& column : related_columns) {
    if (schema->options().vector_column != column && schema->options().primary_column != column &&
        schema->options().version_column != column) {
      return false;
    }
  }
  return true;
}

bool filters_only_contain_pk_and_version(std::shared_ptr<Schema> schema, const Filter::FilterSet& filters) {
  for (auto& filter : filters) {
    if (filter->get_column_name() != schema->options().primary_column &&
        filter->get_column_name() != schema->options().version_column) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<arrow::RecordBatchReader> MakeScanDataReader(std::shared_ptr<Manifest> manifest,
                                                             arrow::fs::FileSystem& fs,
                                                             const ReadOptions& options) {
  auto scalar_reader = std::make_unique<MultiFilesSequentialReader>(fs, manifest->scalar_fragments(),
                                                                    manifest->schema()->scalar_schema(),
                                                                    manifest->schema()->options(), ReadOptions());
  auto vector_reader = std::make_unique<MultiFilesSequentialReader>(fs, manifest->vector_fragments(),
                                                                    manifest->schema()->vector_schema(),
                                                                    manifest->schema()->options(), ReadOptions());

  return CombineReader::Make(std::move(scalar_reader), std::move(vector_reader), manifest->schema());
}

std::unique_ptr<arrow::RecordBatchReader> MakeScanDeleteReader(std::shared_ptr<Manifest> manifest,
                                                               arrow::fs::FileSystem& fs) {
  return std::make_unique<MultiFilesSequentialReader>(fs, manifest->delete_fragments(),
                                                      manifest->schema()->delete_schema(),
                                                      manifest->schema()->options(), ReadOptions());
}
}  // namespace internal
}  // namespace milvus_storage
