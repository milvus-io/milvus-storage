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
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <mutex>

#include "storage/manifest.h"
#include "storage/schema.h"
#include "file/delete_fragment.h"
namespace milvus_storage {

class FilterQueryRecordReader;
class RecordReader;

class Space {
  public:
  Status Write(arrow::RecordBatchReader& reader, const WriteOption& option);

  std::unique_ptr<arrow::RecordBatchReader> Read(const ReadOptions& option) const;

  // Scan delete files
  std::unique_ptr<arrow::RecordBatchReader> ScanDelete() const;

  // Scan data files without filtering deleted data
  std::unique_ptr<arrow::RecordBatchReader> ScanData() const;

  Status Delete(arrow::RecordBatchReader& reader);

  // Open opened a space or create if the space does not exist.
  // If space does not exist. schema should not be nullptr, or an error will be returned.
  // If space exists and version is specified, it will restore to the state at this version,
  // or it will choose the latest version.
  static Result<std::unique_ptr<Space>> Open(const std::string& uri, const Options& options);

  // Write a blob to space. Will return a error if replace is false and a blob with the same name exists.
  Status WriteBlob(const std::string& name, const void* blob, int64_t length, bool replace = false);

  // Read a blob from space, the target must have enough size to hold this blob.
  Status ReadBlob(const std::string& name, void* target) const;

  // Get the byte size of a blob.
  Result<int64_t> GetBlobByteSize(const std::string& name) const;

  const std::vector<Blob>& StatisticsBlobs() const;

  std::shared_ptr<Schema> schema() const;

  int64_t GetCurrentVersion() const;

  private:
  Status Init();

  static Status SafeSaveManifest(std::shared_ptr<arrow::fs::FileSystem> fs,
                                 const std::string& path,
                                 const Manifest* manifest);

  static Result<arrow::fs::FileInfoVector> FindAllManifest(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                           const std::string& path);

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<Manifest> manifest_;
  std::string path_;

  DeleteFragmentVector delete_fragments_;

  int64_t next_manifest_version_ = 0;
  std::mutex mutex_;

  friend FilterQueryRecordReader;
  friend RecordReader;
};
}  // namespace milvus_storage
