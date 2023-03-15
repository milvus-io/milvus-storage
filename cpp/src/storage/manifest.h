#pragma once

#include <memory>

#include "arrow/type.h"
#include "options.h"

class Manifest {
 public:
  Manifest(std::shared_ptr<arrow::Schema> &schema,
           std::shared_ptr<arrow::Schema> &&scalar_schema,
           std::shared_ptr<arrow::Schema> &&vector_schema)
      : schema_(schema),
        scalar_schama_(scalar_schema),
        vector_schema_(vector_schema) {}

  const std::shared_ptr<arrow::Schema> &get_schema() { return schema_; }

  const std::shared_ptr<arrow::Schema> &get_scalar_schema() {
    return scalar_schama_;
  }

  const std::shared_ptr<arrow::Schema> &get_vector_schema() {
    return vector_schema_;
  }

  void AddDataFiles(std::vector<std::string> &scalar_files,
                    std::vector<std::string> &vector_files) {
    scalar_files_.insert(scalar_files_.end(), scalar_files.begin(),
                         scalar_files.end());
    vector_files_.insert(vector_files_.end(), vector_files.begin(),
                         vector_files.end());
  }

  const std::vector<std::string> &GetScalarFiles() { return scalar_files_; }
  const std::vector<std::string> &GetVectorFiles() { return vector_files_; }

 private:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> scalar_schama_;
  std::shared_ptr<arrow::Schema> vector_schema_;
  std::vector<std::string> scalar_files_;
  std::vector<std::string> vector_files_;
};