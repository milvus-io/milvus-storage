#pragma once

#include "storage/schema.h"
#include "file/fragment.h"
namespace milvus_storage {

class Manifest {
  public:
  Manifest() = default;
  explicit Manifest(std::shared_ptr<SpaceOptions> options, std::shared_ptr<Schema> schema);

  const std::shared_ptr<Schema> schema();

  void add_scalar_fragment(Fragment&& fragment);

  void add_vector_fragment(Fragment&& fragment);

  void add_delete_fragment(Fragment&& fragment);

  const FragmentVector& scalar_fragments() const;

  const FragmentVector& vector_fragments() const;

  const FragmentVector& delete_fragments() const;

  int64_t version() const;

  void set_version(int64_t version);

  const std::shared_ptr<SpaceOptions> space_options() const;

  Result<manifest_proto::Manifest> ToProtobuf() const;

  void FromProtobuf(const manifest_proto::Manifest& manifest);

  static Status WriteManifestFile(const Manifest* manifest, arrow::io::OutputStream* output);

  private:
  std::shared_ptr<SpaceOptions> options_;
  std::shared_ptr<Schema> schema_;
  FragmentVector scalar_fragments_;
  FragmentVector vector_fragments_;
  FragmentVector delete_fragments_;

  int64_t version_;
};
}  // namespace milvus_storage
