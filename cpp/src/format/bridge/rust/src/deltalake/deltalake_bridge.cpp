// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#include "deltalake/deltalake_bridge.h"

#include <string_view>

#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "runtime/bridge_util.h"

namespace milvus_storage::deltalake {
namespace {

arrow::Status DeltaErrorStatus(std::string_view message) {
  if (message.find(kFFIErrorCodeMarker) != std::string_view::npos) {
    return MakeBridgeErrorStatus(message);
  }
  if (message.find("[delta:error=invalid]") != std::string_view::npos) {
    return arrow::Status::Invalid(message);
  }
  if (message.find("[delta:error=unsupported]") != std::string_view::npos) {
    return arrow::Status::NotImplemented(message);
  }
  return MakeBridgeErrorStatus(message);
}

}  // namespace

arrow::Result<std::shared_ptr<BlockingDeltaSnapshot>> BlockingDeltaSnapshot::Open(
    const std::string& table_url,
    int64_t requested_version,
    const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
    const FilesystemBinding& binding) {
  if (!filesystem) {
    return arrow::Status::Invalid("Delta snapshot requires a filesystem");
  }
  return CatchRustResult<std::shared_ptr<BlockingDeltaSnapshot>>(DeltaErrorStatus, [&]() {
    ffi::DeltaFilesystemBinding options{binding.is_local ? ffi::DeltaStorageKind::Local : ffi::DeltaStorageKind::Remote,
                                        rust::String(binding.scheme), rust::String(binding.bucket),
                                        rust::String(binding.local_root)};
    auto lease = std::make_shared<FileSystemWrapper>(filesystem);
    auto snapshot =
        ffi::delta_open_snapshot(std::move(lease), rust::Str(table_url), requested_version, std::move(options));
    return std::make_shared<BlockingDeltaSnapshot>(std::move(snapshot));
  });
}

uint64_t BlockingDeltaSnapshot::Version() const { return impl_->delta_snapshot_version(); }

std::vector<FileInfo> BlockingDeltaSnapshot::Files() const {
  auto planned = impl_->delta_snapshot_files();
  std::vector<FileInfo> result;
  result.reserve(planned.size());
  for (const auto& file : planned) {
    result.push_back(FileInfo{std::string(file.path), file.file_size, file.row_count});
  }
  return result;
}

}  // namespace milvus_storage::deltalake
