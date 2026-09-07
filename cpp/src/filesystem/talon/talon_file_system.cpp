// Copyright 2025 Zilliz
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

#include "milvus-storage/filesystem/talon/talon_file_system.h"

#include <memory>
#include <string>
#include <utility>

#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/talon/talon_input_file.h"

namespace milvus_storage::talon {

namespace {

// Talon addresses objects by origin URI. Only S3-compatible backends are
// supported today; they all share the `s3` scheme (the endpoint/region a
// non-AWS backend needs is Talon worker configuration, not part of the URI).
arrow::Result<std::string> TalonSchemeForProvider(const std::string& cloud_provider) {
  if (cloud_provider == kCloudProviderAWS || cloud_provider == kCloudProviderAliyun ||
      cloud_provider == kCloudProviderTencent || cloud_provider == kCloudProviderHuawei) {
    return std::string("s3");
  }
  return arrow::Status::NotImplemented("Talon read caching does not support cloud provider '", cloud_provider,
                                       "' yet (only S3-compatible backends)");
}

}  // namespace

std::string TalonFileSystemProxy::BuildUri(const std::string& path) const {
  std::string key = path;
  while (!key.empty() && key.front() == '/') {
    key.erase(key.begin());
  }
  return scheme_ + "://" + bucket_ + "/" + key;
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> TalonFileSystemProxy::OpenTalon(const std::string& path,
                                                                                            int64_t size) const {
  return OpenTalonInputFile(client_, BuildUri(path), size);
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> TalonFileSystemProxy::OpenInputFile(
    const std::string& path) {
  return OpenTalon(path, arrow::fs::kNoSize);
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> TalonFileSystemProxy::OpenInputFile(
    const arrow::fs::FileInfo& info) {
  // A listing already knows the length; hand it to the reader so GetSize()
  // needs no Talon stat.
  return OpenTalon(info.path(), info.size());
}

arrow::Result<std::shared_ptr<arrow::io::InputStream>> TalonFileSystemProxy::OpenInputStream(const std::string& path) {
  ARROW_ASSIGN_OR_RAISE(auto file, OpenTalon(path, arrow::fs::kNoSize));
  return std::static_pointer_cast<arrow::io::InputStream>(file);
}

arrow::Result<std::shared_ptr<arrow::io::InputStream>> TalonFileSystemProxy::OpenInputStream(
    const arrow::fs::FileInfo& info) {
  ARROW_ASSIGN_OR_RAISE(auto file, OpenTalon(info.path(), info.size()));
  return std::static_pointer_cast<arrow::io::InputStream>(file);
}

arrow::Result<ArrowFileSystemPtr> WrapWithTalon(const ArrowFileSystemConfig& config, ArrowFileSystemPtr base_fs) {
  if (config.talon_coordinator.empty()) {
    return arrow::Status::Invalid("fs.talon.enabled=true requires fs.talon.coordinator to be set");
  }
  ARROW_ASSIGN_OR_RAISE(auto scheme, TalonSchemeForProvider(config.cloud_provider));

  // Producers return a bucket-rooted FileSystemProxy; reuse its base path and
  // origin backend so non-read operations are unchanged.
  auto proxy = std::dynamic_pointer_cast<FileSystemProxy>(base_fs);
  if (proxy == nullptr) {
    return arrow::Status::Invalid("Talon read caching requires a remote filesystem backend");
  }

  ARROW_ASSIGN_OR_RAISE(auto client, TalonClient::Make(config.talon_coordinator));
  LOG_STORAGE_INFO_ << "Talon read caching enabled for bucket '" << config.bucket_name << "' via coordinator '"
                    << config.talon_coordinator << "'";
  return std::make_shared<TalonFileSystemProxy>(proxy->base_path(), proxy->base_fs(), std::move(client),
                                                std::move(scheme), config.bucket_name);
}

}  // namespace milvus_storage::talon
