#pragma once

#include <arrow/io/interfaces.h>
#include <arrow/type_fwd.h>

#include <iostream>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "arrow/type.h"
#include "common/exception.h"
#include "options.h"
#include "proto/manifest.pb.h"
#include "common/utils.h"
#include <iostream>
#include "parquet/exception.h"

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

  manifest::Manifest
  ToProtobufManifest() const {
    manifest::Manifest manifest;
    manifest.mutable_options()->set_path(option_->path);
    manifest.mutable_options()->set_primary_column(option_->primary_column);
    manifest.mutable_options()->set_version_column(option_->version_column);
    manifest.mutable_options()->set_vector_column(option_->vector_column);

    manifest.add_vector_files()->append(vector_files_.begin(), vector_files_.end());
    manifest.add_scalar_files()->append(scalar_files_.begin(), scalar_files_.end());
    manifest.add_delete_files()->append(delete_files_.begin(), delete_files_.end());

    manifest.set_allocated_schema(ToProtobufSchema(schema_.get()).release());
    return manifest;
  }

  static void
  WriteManifestFile(const Manifest* manifest, arrow::io::OutputStream* output) {
    auto proto_manifest = manifest->ToProtobufManifest();
    auto size = proto_manifest.ByteSizeLong();
    uint8_t manifest_bytes[size];
    proto_manifest.SerializeToArray(manifest_bytes, size);
    PARQUET_THROW_NOT_OK(output->Write(manifest_bytes, size));
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