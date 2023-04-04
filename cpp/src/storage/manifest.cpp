#include "manifest.h"
#include "common/schema_util.h"

Manifest::Manifest(std::shared_ptr<SpaceOption>& option, std::shared_ptr<arrow::Schema>& schema)
    : option_(option), schema_(schema) {
  scalar_schema_ = BuildScalarSchema(schema.get(), option.get());
  vector_schema_ = BuildVectorSchema(schema.get(), option.get());
  delete_schema_ = BuildDeleteSchema(schema.get(), option.get());
}

manifest::Manifest
Manifest::ToProtobufManifest() const {
  manifest::Manifest manifest;
  manifest.mutable_options()->set_path(option_->uri);
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
  char* buffer = new char[size];
  proto_manifest.SerializeToArray(buffer, static_cast<int>(size));
  PARQUET_THROW_NOT_OK(output->Write(buffer, size));
  delete[] buffer;
}
