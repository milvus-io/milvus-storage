// Copyright 2023 Zilliz
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "storage/manifest.h"
#include <algorithm>
#include <memory>
#include "arrow/filesystem/filesystem.h"

namespace milvus_storage {

Manifest::Manifest(std::shared_ptr<Schema> schema) : schema_(std::move(schema)) {}

const std::shared_ptr<Schema> Manifest::schema() { return schema_; }

void Manifest::add_scalar_fragment(Fragment&& fragment) { scalar_fragments_.push_back(fragment); }

void Manifest::add_vector_fragment(Fragment&& fragment) { vector_fragments_.push_back(fragment); }

void Manifest::add_delete_fragment(Fragment&& fragment) { delete_fragments_.push_back(fragment); }

void Manifest::add_blob(Blob&& blob) { blobs_.emplace_back(blob); }

const FragmentVector& Manifest::scalar_fragments() const { return scalar_fragments_; }

const FragmentVector& Manifest::vector_fragments() const { return vector_fragments_; }

const FragmentVector& Manifest::delete_fragments() const { return delete_fragments_; }

bool Manifest::has_blob(const std::string& name) {
  auto iter = std::find_if(blobs_.begin(), blobs_.end(), [&](Blob& blob) { return blob.name == name; });
  return iter != blobs_.end();
}

void Manifest::remove_blob_if_exist(const std::string& name) {
  std::remove_if(blobs_.begin(), blobs_.end(), [&](Blob& blob) { return blob.name == name; });
}

Result<Blob> Manifest::get_blob(const std::string& name) {
  auto iter = std::find_if(blobs_.begin(), blobs_.end(), [&](Blob& blob) { return blob.name == name; });
  if (iter == blobs_.end()) {
    return Status::FileNotFound("blob not found");
  }
  return *iter;
}

const std::vector<Blob>& Manifest::blobs() const { return blobs_; }

int64_t Manifest::version() const { return version_; }

void Manifest::set_version(int64_t version) { version_ = version; }

Result<manifest_proto::Manifest> Manifest::ToProtobuf() const {
  manifest_proto::Manifest manifest;
  manifest.set_version(version_);
  for (auto& fragment : vector_fragments_) {
    manifest.mutable_vector_fragments()->AddAllocated(fragment.ToProtobuf().release());
  }
  for (auto& fragment : scalar_fragments_) {
    manifest.mutable_scalar_fragments()->AddAllocated(fragment.ToProtobuf().release());
  }
  for (auto& fragment : delete_fragments_) {
    manifest.mutable_delete_fragments()->AddAllocated(fragment.ToProtobuf().release());
  }
  for (auto& blob : blobs_) {
    manifest.mutable_blobs()->AddAllocated(blob.ToProtobuf().release());
  }

  ASSIGN_OR_RETURN_NOT_OK(auto schema_proto, schema_->ToProtobuf());
  manifest.set_allocated_schema(schema_proto.release());
  return manifest;
}

void Manifest::FromProtobuf(const manifest_proto::Manifest& manifest) {
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

  for (auto& blob : manifest.blobs()) {
    blobs_.emplace_back(Blob::FromProtobuf(blob));
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

Result<std::shared_ptr<Manifest>> Manifest::ParseFromFile(std::shared_ptr<arrow::io::InputStream> istream,
                                                          arrow::fs::FileInfo& file_info) {
  auto size = file_info.size();
  char* buffer = new char[size];
  auto res = istream->Read(size, buffer);
  if (!res.ok()) {
    delete[] buffer;
    return Status::ArrowError(res.status().ToString());
  }

  manifest_proto::Manifest proto_manifest;
  proto_manifest.ParseFromArray(buffer, size);
  auto manifest = std::make_shared<Manifest>();
  manifest->FromProtobuf(proto_manifest);
  delete[] buffer;
  return manifest;
}
}  // namespace milvus_storage
