#include "storage/manifest.h"
#include <algorithm>

namespace milvus_storage {

Manifest::Manifest(std::shared_ptr<SpaceOptions> options, std::shared_ptr<Schema> schema)
    : options_(std::move(options)), schema_(std::move(schema)) {}

const std::shared_ptr<Schema> Manifest::schema() { return schema_; }

void Manifest::add_scalar_fragment(Fragment&& fragment) { scalar_fragments_.push_back(fragment); }

void Manifest::add_vector_fragment(Fragment&& fragment) { vector_fragments_.push_back(fragment); }

void Manifest::add_delete_fragment(Fragment&& fragment) { delete_fragments_.push_back(fragment); }

const FragmentVector& Manifest::scalar_fragments() const { return scalar_fragments_; }

const FragmentVector& Manifest::vector_fragments() const { return vector_fragments_; }

const FragmentVector& Manifest::delete_fragments() const { return delete_fragments_; }

int64_t Manifest::version() const { return version_; }

void Manifest::set_version(int64_t version) { version_ = version; }

const std::shared_ptr<SpaceOptions> Manifest::space_options() const { return options_; }

Result<manifest_proto::Manifest> Manifest::ToProtobuf() const {
  manifest_proto::Manifest manifest;
  manifest.set_version(version_);
  manifest.set_allocated_options(options_->ToProtobuf().release());
  for (auto& fragment : vector_fragments_) {
    manifest.mutable_scalar_fragments()->AddAllocated(fragment.ToProtobuf().release());
  }
  for (auto& fragment : scalar_fragments_) {
    manifest.mutable_vector_fragments()->AddAllocated(fragment.ToProtobuf().release());
  }
  for (auto& fragment : delete_fragments_) {
    manifest.mutable_delete_fragments()->AddAllocated(fragment.ToProtobuf().release());
  }

  ASSIGN_OR_RETURN_NOT_OK(auto schema_proto, schema_->ToProtobuf());
  manifest.set_allocated_schema(schema_proto.release());
  return manifest;
}

void Manifest::FromProtobuf(const manifest_proto::Manifest& manifest) {
  options_ = std::make_shared<SpaceOptions>();
  options_->FromProtobuf(manifest.options());

  schema_ = std::make_shared<Schema>();
  schema_->FromProtobuf(manifest.schema());

  for (auto& fragment : manifest.vector_fragments()) {
    vector_fragments_.emplace_back(*Fragment::FromProtobuf(fragment).release());
  }

  for (auto& fragment : manifest.scalar_fragments()) {
    scalar_fragments_.emplace_back(*Fragment::FromProtobuf(fragment).release());
  }

  for (auto& fragment : manifest.delete_fragments()) {
    delete_fragments_.emplace_back(*Fragment::FromProtobuf(fragment).release());
  }
  version_ = manifest.version();
}

Status Manifest::WriteManifestFile(const Manifest* manifest, arrow::io::OutputStream* output) {
  ASSIGN_OR_RETURN_NOT_OK(auto proto_manifest, manifest->ToProtobuf());
  auto size = proto_manifest.ByteSizeLong();
  char* buffer = new char[size];
  proto_manifest.SerializeToArray(buffer, static_cast<int>(size));
  RETURN_ARROW_NOT_OK(output->Write(buffer, size));
  delete[] buffer;
  return Status::OK();
}
}  // namespace milvus_storage
