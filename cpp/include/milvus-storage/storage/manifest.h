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

#pragma once

#include "milvus-storage/storage/schema.h"
#include "milvus-storage/file/fragment.h"
#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/file/blob.h"
namespace milvus_storage {

class Manifest {
  public:
  Manifest() = default;
  explicit Manifest(std::shared_ptr<Schema> schema);

  std::shared_ptr<Schema> schema();

  void add_scalar_fragment(Fragment&& fragment);

  void add_vector_fragment(Fragment&& fragment);

  void add_delete_fragment(Fragment&& fragment);

  void add_blob(Blob&& blob);

  [[nodiscard]] const FragmentVector& scalar_fragments() const;

  [[nodiscard]] const FragmentVector& vector_fragments() const;

  [[nodiscard]] const FragmentVector& delete_fragments() const;

  bool has_blob(const std::string& name);

  void remove_blob_if_exist(const std::string& name);

  Result<Blob> get_blob(const std::string& name);

  [[nodiscard]] const std::vector<Blob>& blobs() const;

  [[nodiscard]] int64_t version() const;

  void set_version(int64_t version);

  [[nodiscard]] Result<manifest_proto::Manifest> ToProtobuf() const;

  void FromProtobuf(const manifest_proto::Manifest& manifest);

  static Status WriteManifestFile(const Manifest& manifest, arrow::io::OutputStream& output);

  static Result<std::shared_ptr<Manifest>> ParseFromFile(std::shared_ptr<arrow::io::InputStream> istream,
                                                         arrow::fs::FileInfo& file_info);

  private:
  // arrow's RecordBatchReader have a schema method which returns an shared ptr wrapped schema
  // we store schema_ as shared_ptr here to avoid copy
  std::shared_ptr<Schema> schema_;
  FragmentVector scalar_fragments_;
  FragmentVector vector_fragments_;
  FragmentVector delete_fragments_;
  BlobVector blobs_;

  int64_t version_ = 0;
};
}  // namespace milvus_storage
