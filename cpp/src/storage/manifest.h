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
#include "schema.h"
#include "parquet/exception.h"
namespace milvus_storage {

class Manifest {
  public:
  Manifest() = default;
  explicit Manifest(std::shared_ptr<SpaceOptions> options, std::shared_ptr<Schema> schema);

  const std::shared_ptr<Schema>
  schema();

  void
  add_scalar_files(const std::vector<std::string>& scalar_files);

  void
  add_vector_files(const std::vector<std::string>& vector_files);

  void
  add_delete_file(std::string& delete_file);

  const std::vector<std::string>&
  scalar_files() const;

  const std::vector<std::string>&
  vector_files() const;

  const std::vector<std::string>&
  delete_files() const;

  const std::shared_ptr<SpaceOptions>
  space_options();

  manifest_proto::Manifest
  ToProtobuf() const;

  void
  FromProtobuf(const manifest_proto::Manifest& manifest);

  static void
  WriteManifestFile(const Manifest* manifest, arrow::io::OutputStream* output);

  private:
  std::shared_ptr<SpaceOptions> options_;
  std::shared_ptr<Schema> schema_;
  std::vector<std::string> scalar_files_;
  std::vector<std::string> vector_files_;
  std::vector<std::string> delete_files_;
};
}  // namespace milvus_storage