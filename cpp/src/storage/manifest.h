#pragma once

#include <arrow/type_fwd.h>

#include <iostream>
#include <memory>

#include "arrow/type.h"
#include "options.h"

class Manifest {
  public:
  Manifest(std::shared_ptr<SpaceOption>& option,
           std::shared_ptr<arrow::Schema>& schema,
           std::shared_ptr<arrow::Schema>&& scalar_schema,
           std::shared_ptr<arrow::Schema>&& vector_schema,
           std::shared_ptr<arrow::Schema>&& delete_schema)
      : option_(option),
        schema_(schema),
        scalar_schama_(scalar_schema),
        vector_schema_(vector_schema),
        delete_schema_(delete_schema) {
  }

  const std::shared_ptr<arrow::Schema>&
  get_schema() {
    return schema_;
  }

  const std::shared_ptr<arrow::Schema>&
  get_scalar_schema() {
    return scalar_schama_;
  }

  const std::shared_ptr<arrow::Schema>&
  get_vector_schema() {
    return vector_schema_;
  }

  const std::shared_ptr<arrow::Schema>&
  get_delete_schema() {
    return delete_schema_;
  }

  const std::shared_ptr<SpaceOption>&
  get_option() {
    return option_;
  }

  void
  AddDataFiles(std::vector<std::string>& scalar_files, std::vector<std::string>& vector_files) {
    scalar_files_.insert(scalar_files_.end(), scalar_files.begin(), scalar_files.end());
    vector_files_.insert(vector_files_.end(), vector_files.begin(), vector_files.end());
  }

  void
  AddDeleteFile(std::string& delete_file) {
    delete_files_.emplace_back(delete_file);
  }

  const std::vector<std::string>&
  GetScalarFiles() const {
    return scalar_files_;
  }
  const std::vector<std::string>&
  GetVectorFiles() const {
    return vector_files_;
  }
  const std::vector<std::string>&
  GetDeleteFiles() const {
    return delete_files_;
  }

  private:
  std::shared_ptr<SpaceOption> option_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> scalar_schama_;
  std::shared_ptr<arrow::Schema> vector_schema_;
  std::shared_ptr<arrow::Schema> delete_schema_;
  std::vector<std::string> scalar_files_;
  std::vector<std::string> vector_files_;
  std::vector<std::string> delete_files_;
};