#pragma once

#include <memory>

#include "reader/scan_record_reader.h"
#include "reader/merge_record_reader.h"
#include "reader/filter_query_record_reader.h"
#include "storage/default_space.h"

struct RecordReader {
  static std::unique_ptr<arrow::RecordBatchReader>
  GetRecordReader(const DefaultSpace& space, std::shared_ptr<ReadOption>& options) {
    std::set<std::string> related_columns;
    for (auto& column : options->columns) {
      related_columns.insert(column);
    }
    for (auto& filter : options->filters) {
      related_columns.insert(filter->get_column_name());
    }

    if (only_contain_scalar_columns(space, related_columns)) {
      return std::unique_ptr<arrow::RecordBatchReader>(
          new ScanRecordReader(options, space.manifest_->GetScalarFiles(), space));
    }

    if (only_contain_vector_columns(space, related_columns)) {
      return std::unique_ptr<arrow::RecordBatchReader>(
          new ScanRecordReader(options, space.manifest_->GetVectorFiles(), space));
    }

    if (filters_only_contain_pk_and_version(space, options->filters)) {
      return std::unique_ptr<arrow::RecordBatchReader>(
          new MergeRecordReader(options, space.manifest_->GetScalarFiles(), space.manifest_->GetVectorFiles(), space));
    } else {
      return std::unique_ptr<arrow::RecordBatchReader>(new FilterQueryRecordReader(
          options, space.manifest_->GetScalarFiles(), space.manifest_->GetVectorFiles(), space));
    }
  }

  static bool
  only_contain_scalar_columns(const DefaultSpace& space, const std::set<std::string>& related_columns) {
    for (auto& column : related_columns) {
      if (space.options_->vector_column == column) {
        return false;
      }
    }
    return true;
  }

  static bool
  only_contain_vector_columns(const DefaultSpace& space, const std::set<std::string>& related_columns) {
    for (auto& column : related_columns) {
      if (space.options_->vector_column != column && space.options_->primary_column != column &&
          space.options_->version_column != column) {
        return false;
      }
    }
    return true;
  }

  static bool
  filters_only_contain_pk_and_version(const DefaultSpace& space, const std::vector<Filter*>& filters) {
    for (auto& filter : filters) {
      if (filter->get_column_name() != space.options_->primary_column &&
          filter->get_column_name() != space.options_->version_column) {
        return false;
      }
    }
    return true;
  }
};
