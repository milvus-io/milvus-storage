#include "storage/manifest.h"

namespace milvus_storage {

Manifest::Manifest(std::shared_ptr<SpaceOptions> options, std::shared_ptr<Schema> schema)
    : options_(std::move(options)), schema_(std::move(schema)) {}

const std::shared_ptr<Schema> Manifest::schema() { return schema_; }

void Manifest::add_scalar_files(const std::vector<std::string>& scalar_files) {
  scalar_files_.insert(scalar_files_.end(), scalar_files.begin(), scalar_files.end());
}

void Manifest::add_vector_files(const std::vector<std::string>& vector_files) {
  vector_files_.insert(vector_files_.end(), vector_files.begin(), vector_files.end());
}

void Manifest ::add_delete_file(const std::string& delete_file) { delete_files_.emplace_back(delete_file); }

const std::vector<std::string>& Manifest::scalar_files() const { return scalar_files_; }

const std::vector<std::string>& Manifest::vector_files() const { return vector_files_; }

const std::vector<std::string>& Manifest::delete_files() const { return delete_files_; }

const std::shared_ptr<SpaceOptions> Manifest::space_options() { return options_; }

Result<manifest_proto::Manifest> Manifest::ToProtobuf() const {
  manifest_proto::Manifest manifest;
  manifest.set_allocated_options(options_->ToProtobuf().release());
  for (const auto& file : vector_files_) {
    manifest.add_vector_files(file);
  }
  for (const auto& file : scalar_files_) {
    manifest.add_scalar_files(file);
  }
  for (const auto& file : delete_files_) {
    manifest.add_delete_files(file);
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

  for (auto& file : manifest.vector_files()) {
    vector_files_.emplace_back(file);
  }

  for (auto& file : manifest.scalar_files()) {
    scalar_files_.emplace_back(file);
  }

  for (auto& file : manifest.delete_files()) {
    delete_files_.emplace_back(file);
  }
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