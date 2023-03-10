#pragma once

#include <arrow/type_fwd.h>

#include <memory>

#include "../options/options.h"

class Manifest {
 public:
  explicit Manifest(std::shared_ptr<arrow::Schema> schema,
                    std::shared_ptr<arrow::Schema> scalar_schema,
                    std::shared_ptr<arrow::Schema> vector_schema);
  std::shared_ptr<arrow::Schema> get_schema();
  std::shared_ptr<arrow::Schema> get_scalar_schema();
  std::shared_ptr<arrow::Schema> get_vector_schema();
  void AddDataFiles(std::vector<std::string> scalar_files,
                    std::vector<std::string> vector_files);

  std::vector<std::string> GetScalarFiles();
  std::vector<std::string> GetVectorFiles();

 private:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> scalar_schama_;
  std::shared_ptr<arrow::Schema> vector_schema_;
};