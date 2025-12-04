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

#include "milvus-storage/transaction/transhandler.h"

#include <cassert>
#include <charconv>
#include <string>
#include <string_view>
#include <memory>
#include <algorithm>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/buffer.h>
#include <arrow/filesystem/filesystem.h>
#include <mutex>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/path_util.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"

namespace milvus_storage::api::transaction {

#define MANIFEST_VERSION_MINIMAL 0

arrow::Status unsafe_write(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                           const std::string& path,
                           std::string_view data) {
  static std::mutex write_mutex;
  std::scoped_lock lock(write_mutex);

  ARROW_ASSIGN_OR_RAISE(auto file_info, fs->GetFileInfo(path));
  if (file_info.type() != arrow::fs::FileType::NotFound) {
    return arrow::Status::AlreadyExists("File already exists: ", path);
  }

  auto [parent, _] = milvus_storage::GetAbstractPathParent(path);
  if (!parent.empty()) {
    ARROW_RETURN_NOT_OK(fs->CreateDir(parent));
  }

  ARROW_ASSIGN_OR_RAISE(auto output_stream, fs->OpenOutputStream(path));
  ARROW_RETURN_NOT_OK(output_stream->Write(data.data(), data.size()));
  ARROW_RETURN_NOT_OK(output_stream->Close());

  return arrow::Status::OK();
}

arrow::Status conditional_write(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                const std::string& path,
                                std::string_view data) {
  static std::mutex write_mutex;
  std::scoped_lock lock(write_mutex);

  // check if fs support conditional write
  if (!milvus_storage::ExtendFileSystem::IsExtendFileSystem(fs)) {
    return arrow::Status::Invalid("File system can't support conditional write.");
  }

  // do the conditional write
  auto fs_ext = std::dynamic_pointer_cast<milvus_storage::ExtendFileSystem>(fs);
  ARROW_ASSIGN_OR_RAISE(auto output_stream, fs_ext->OpenConditionalOutputStream(path));
  ARROW_RETURN_NOT_OK(output_stream->Write(data.data(), data.size()));
  auto result = output_stream->Close();
  if (!result.ok()) {
    // already exist then return AlreadyExists
    if (result.code() == arrow::StatusCode::IOError) {
      return arrow::Status::AlreadyExists("File already exists: ", path);
    }
    // others return the error
    return result;
  }

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> direct_read(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                          const std::string& path) {
  std::shared_ptr<arrow::Buffer> buffer;
  // Open input file and get size
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(path));
  ARROW_ASSIGN_OR_RAISE(int64_t file_size, input_file->GetSize());

  // Read into an Arrow Buffer (makes memory management automatic)
  ARROW_ASSIGN_OR_RAISE(buffer, input_file->Read(file_size));

  // Ensure we read the expected size
  if (buffer->size() != file_size) {
    return arrow::Status::IOError("Failed to read the complete file, expected size =", file_size,
                                  ", actual size =", static_cast<int64_t>(buffer->size()));
  }

  ARROW_RETURN_NOT_OK(input_file->Close());
  return buffer;
}

arrow::Status lazy_load_file_system(milvus_storage::ArrowFileSystemPtr& file_system,
                                    const api::Properties& properties) {
  if (file_system == nullptr) {
    auto& fs_cache = milvus_storage::LRUCache<milvus_storage::ArrowFileSystemConfig,
                                              milvus_storage::ArrowFileSystemPtr>::getInstance();
    milvus_storage::ArrowFileSystemConfig fs_config;
    ARROW_RETURN_NOT_OK(milvus_storage::ArrowFileSystemConfig::create_file_system_config(properties, fs_config));
    ARROW_ASSIGN_OR_RAISE(file_system, fs_cache.get(fs_config, milvus_storage::CreateArrowFileSystem));
  }

  return arrow::Status::OK();
}

/**
 * UnsafeTransHandler implements a simple transaction handler with thread isolation.
 * It assumes that there are no concurrent transactions modifying the same manifest
 * in process-level.
 * Should NOT use this handler in multi-process concurrent scenarios.
 * Only used to test the transaction.
 */
template <typename T>
class UnsafeTransHandler : public TransactionHandler<T> {
  public:
  UnsafeTransHandler(const std::string& base_path, const api::Properties& properties);

  arrow::Result<int64_t> get_latest_version() override;

  arrow::Result<std::shared_ptr<T>> get_current_manifest(int64_t version) override;

  arrow::Result<CommitResult> commit(std::shared_ptr<T>& manifest, int64_t old_version, int64_t new_version) override;

  protected:
  virtual arrow::Status direct_write(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                     const std::string& path,
                                     std::string_view data);

  protected:
  std::string base_path_;
  api::Properties properties_;

  milvus_storage::ArrowFileSystemPtr fs_;
};

template <typename T>
UnsafeTransHandler<T>::UnsafeTransHandler(const std::string& base_path, const api::Properties& properties)
    : base_path_(base_path), properties_(properties), fs_(nullptr) {}

template <typename T>
arrow::Result<int64_t> UnsafeTransHandler<T>::get_latest_version() {
  ARROW_RETURN_NOT_OK(lazy_load_file_system(fs_, properties_));

  // Check if metadata directory exists
  std::string metadata_dir = base_path_ + kSep + kMetadataDir;
  ARROW_ASSIGN_OR_RAISE(auto dir_info, fs_->GetFileInfo(metadata_dir));
  if (dir_info.type() == arrow::fs::FileType::NotFound) {
    return MANIFEST_VERSION_MINIMAL;  // No manifests yet
  }

  arrow::fs::FileSelector selector;
  selector.base_dir = metadata_dir;
  selector.allow_not_found = false;
  selector.recursive = false;
  selector.max_recursion = 0;

  // list the objects in metadata directory and get the latest manifest file
  ARROW_ASSIGN_OR_RAISE(auto file_infos, fs_->GetFileInfo(selector));

  int64_t latest_version = MANIFEST_VERSION_MINIMAL;
  for (const auto& file_info : file_infos) {
    std::string file_name = file_info.base_name();
    // filter manifest files with prefix and suffix
    if (file_name.find(kManifestFileNamePrefix) != 0) {
      continue;  // must start with prefix
    }
    if (file_name.size() <= kManifestFileNamePrefix.length() + kManifestFileNameSuffix.length()) {
      continue;  // too short to contain version number
    }
    // extract version number (between prefix and suffix)
    std::string version_str =
        file_name.substr(kManifestFileNamePrefix.length(),
                         file_name.length() - kManifestFileNamePrefix.length() - kManifestFileNameSuffix.length());
    int64_t version = 0;
    auto [ptr, ec] = std::from_chars(version_str.data(), version_str.data() + version_str.size(), version);
    if (ec != std::errc() || ptr != version_str.data() + version_str.size()) {
      continue;  // not a valid version number
    }
    latest_version = std::max<int64_t>(latest_version, version);
  }

  return latest_version;
}

template <typename T>
arrow::Result<std::shared_ptr<T>> UnsafeTransHandler<T>::get_current_manifest(int64_t version) {
  auto manifest = std::make_shared<T>();
  // not exist any manifest in local
  if (version <= MANIFEST_VERSION_MINIMAL) {
    return manifest;
  }
  ARROW_RETURN_NOT_OK(lazy_load_file_system(fs_, properties_));

  ARROW_ASSIGN_OR_RAISE(auto manifest_buffer, direct_read(fs_, base_path_ + kSep + kManifestFilePrefix +
                                                                   std::to_string(version) + kManifestFileNameSuffix));
  ARROW_RETURN_NOT_OK(manifest->deserialize(
      std::string_view(reinterpret_cast<const char*>(manifest_buffer->data()), manifest_buffer->size())));

  auto all_cgs = manifest->get_all();
  for (auto& cg : all_cgs) {
    for (auto& file : cg->files) {
      if (!file.path.empty() && file.path[0] == '_') {
        file.path = base_path_ + kSep + file.path;
      }
    }
  }
  return manifest;
}

template <typename T>
arrow::Result<CommitResult> UnsafeTransHandler<T>::commit(std::shared_ptr<T>& manifest,
                                                          int64_t old_version,
                                                          int64_t new_version) {
  ARROW_RETURN_NOT_OK(lazy_load_file_system(fs_, properties_));

  // Serialize new column groups to Avro
  ARROW_ASSIGN_OR_RAISE(auto manifest_bytes, manifest->serialize());

  arrow::Status result =
      direct_write(fs_, base_path_ + kSep + kManifestFilePrefix + std::to_string(new_version) + kManifestFileNameSuffix,
                   manifest_bytes);

  return CommitResult{.success = result.ok(),
                      .committed_version = result.ok() ? new_version : MANIFEST_VERSION_INVALID,
                      .failed_message = result.ok() ? "" : result.message()};
}

template <typename T>
arrow::Status UnsafeTransHandler<T>::direct_write(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                  const std::string& path,
                                                  std::string_view data) {
  return unsafe_write(fs, path, data);
}

template <typename T>
class ConditionalTransHandler final : public UnsafeTransHandler<T> {
  public:
  ConditionalTransHandler(const std::string& base_path, const api::Properties& properties);

  private:
  arrow::Status direct_write(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                             const std::string& path,
                             std::string_view data) override;
};

template <typename T>
ConditionalTransHandler<T>::ConditionalTransHandler(const std::string& base_path, const api::Properties& properties)
    : UnsafeTransHandler<T>(base_path, properties) {}

template <typename T>
arrow::Status ConditionalTransHandler<T>::direct_write(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                       const std::string& path,
                                                       std::string_view data) {
  return conditional_write(fs, path, data);
}

template <typename T>
std::shared_ptr<TransactionHandler<T>> TransactionHandler<T>::CreateTransactionHandler(
    const std::string& handler_type, const std::string& base_path, const api::Properties& properties) {
  if (handler_type == TRANSACTION_HANDLER_TYPE_UNSAFE) {
    return std::make_shared<UnsafeTransHandler<T>>(base_path, properties);
  } else if (handler_type == TRANSACTION_HANDLER_TYPE_CONDITIONAL) {
    return std::make_shared<ConditionalTransHandler<T>>(base_path, properties);
  } else {
    assert(false);
  }

  return nullptr;
}

template class TransactionHandler<Manifest>;
}  // namespace milvus_storage::api::transaction