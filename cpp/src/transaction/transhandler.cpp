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

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/buffer.h>
#include <mutex>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"

namespace milvus_storage::api::transaction {

#define VERSION_HIT_FILE_NAME "/version-hint.txt"
#define MANIFEST_FILE_NAME_PREFIX "/manifest-"
#define MANIFEST_VERSION_MINIMAL 0

arrow::Result<bool> direct_write(std::shared_ptr<arrow::fs::FileSystem> fs,
                                 const std::string& path,
                                 std::shared_ptr<arrow::Buffer> buffer,
                                 bool overwrite = false) {
  static std::mutex write_mutex;
  std::scoped_lock lock(write_mutex);

  if (!overwrite) {
    ARROW_ASSIGN_OR_RAISE(auto file_info, fs->GetFileInfo(path));
    if (file_info.type() != arrow::fs::FileType::NotFound) {
      return false;
    }
  }

  ARROW_ASSIGN_OR_RAISE(auto output_stream, fs->OpenOutputStream(path));
  ARROW_RETURN_NOT_OK(output_stream->Write(buffer));
  ARROW_RETURN_NOT_OK(output_stream->Close());

  return true;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> direct_read(std::shared_ptr<arrow::fs::FileSystem> fs,
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

arrow::Result<int64_t> buffer_to_int64(std::shared_ptr<arrow::Buffer> buffer) {
  assert(buffer);
  const char* str = reinterpret_cast<const char*>(buffer->data());
  size_t str_len = buffer->size();

  int64_t result = 0;
  const char* end = str + str_len;
  auto [ptr, ec] = std::from_chars(str, end, result);
  if (ec != std::errc() || ptr != end) {
    return arrow::Status::Invalid("Failed to convert string to int64_t");
  }

  return result;
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

  arrow::Result<bool> commit(std::shared_ptr<T>& manifest, int64_t old_version, int64_t new_version) override;

  private:
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

  // Read version hint into Arrow buffer
  auto version_buf_result = direct_read(fs_, base_path_ + VERSION_HIT_FILE_NAME);
  if (!version_buf_result.ok()) {
    // If version hint file does not exist, assume version 0
    if (version_buf_result.status().code() == arrow::StatusCode::IOError) {
      // direct return version 0
      return MANIFEST_VERSION_MINIMAL;
    }

    return version_buf_result.status();
  }

  // Interpret first 8 bytes as int64_t (assuming file was written in native endianness)
  ARROW_ASSIGN_OR_RAISE(auto current_version, buffer_to_int64(version_buf_result.ValueOrDie()));
  return current_version;
}

template <typename T>
arrow::Result<std::shared_ptr<T>> UnsafeTransHandler<T>::get_current_manifest(int64_t version) {
  auto manifest = std::make_shared<T>();
  // not exist any manifest in local
  if (version <= MANIFEST_VERSION_MINIMAL) {
    return manifest;
  }
  ARROW_RETURN_NOT_OK(lazy_load_file_system(fs_, properties_));

  ARROW_ASSIGN_OR_RAISE(auto manifest_buffer,
                        direct_read(fs_, base_path_ + MANIFEST_FILE_NAME_PREFIX + std::to_string(version)));
  ARROW_RETURN_NOT_OK(
      manifest->deserialize(std::string_view((const char*)manifest_buffer->data(), manifest_buffer->size())));
  return manifest;
}

template <typename T>
arrow::Result<bool> UnsafeTransHandler<T>::commit(std::shared_ptr<T>& manifest,
                                                  int64_t old_version,
                                                  int64_t new_version) {
  // for now, we only support old_version +1 == new_version
  assert(old_version == new_version - 1);

  ARROW_RETURN_NOT_OK(lazy_load_file_system(fs_, properties_));

  // Serialize new cloumn groups to JSON
  ARROW_ASSIGN_OR_RAISE(auto manifest_json, manifest->serialize());

  // Read current manifest to ensure consistency
  ARROW_ASSIGN_OR_RAISE(auto current_version, get_latest_version());

  if (current_version != old_version) {
    return false;
  }

  auto manifest_buffer = std::make_shared<arrow::Buffer>((const uint8_t*)manifest_json.c_str(), manifest_json.size());

  // current manifest version already exist
  ARROW_ASSIGN_OR_RAISE(
      auto write_ok,
      direct_write(fs_, base_path_ + MANIFEST_FILE_NAME_PREFIX + std::to_string(new_version), manifest_buffer));
  if (!write_ok) {
    return false;
  }

  // Update version hint
  auto version_str = std::to_string(new_version);
  auto version_buffer = std::make_shared<arrow::Buffer>((const uint8_t*)version_str.c_str(), version_str.size());

  // overwrite won't failed
  ARROW_RETURN_NOT_OK(direct_write(fs_, base_path_ + VERSION_HIT_FILE_NAME, version_buffer, true /* overwrite */));

  return true;
}

template <typename T>
std::shared_ptr<TransactionHandler<T>> TransactionHandler<T>::CreateTransactionHandler(
    const std::string& handler_type, const std::string& base_path, const api::Properties& properties) {
  if (handler_type == TRANSACTION_HANDLER_TYPE_UNSAFE) {
    return std::make_shared<UnsafeTransHandler<T>>(base_path, properties);
  } else {
    assert(false);
    return nullptr;
  }

  // unreachable
}

template class TransactionHandler<Manifest>;

}  // namespace milvus_storage::api::transaction