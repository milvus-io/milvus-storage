// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#include "deltalake/deltalake_bridge.h"

#include <string_view>
#include <arrow/c/bridge.h>

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

template <typename Loader>
arrow::Result<std::shared_ptr<BlockingDeltaSnapshot>> LoadSnapshot(
    const std::string& table_url,
    int64_t version,
    const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
    const FilesystemBinding& binding,
    Loader&& load) {
  if (!filesystem) {
    return arrow::Status::Invalid("Delta snapshot requires a filesystem");
  }
  return CatchRustResult<std::shared_ptr<BlockingDeltaSnapshot>>(DeltaErrorStatus, [&]() {
    ffi::DeltaFilesystemBinding options{binding.is_local ? ffi::DeltaStorageKind::Local : ffi::DeltaStorageKind::Remote,
                                        rust::String(binding.scheme), rust::String(binding.bucket),
                                        rust::String(binding.local_root)};
    auto lease = std::make_shared<FileSystemWrapper>(filesystem);
    auto snapshot = load(std::move(lease), rust::Str(table_url), version, std::move(options));
    return std::make_shared<BlockingDeltaSnapshot>(std::move(snapshot));
  });
}

}  // namespace

arrow::Result<std::shared_ptr<BlockingDeltaSnapshot>> BlockingDeltaSnapshot::Open(
    const std::string& table_url,
    int64_t requested_version,
    const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
    const FilesystemBinding& binding) {
  return LoadSnapshot(table_url, requested_version, filesystem, binding, ffi::delta_open_snapshot);
}

uint64_t BlockingDeltaSnapshot::Version() const { return impl_->delta_snapshot_version(); }

arrow::Result<std::shared_ptr<BlockingDeltaSnapshot>> BlockingDeltaSnapshot::LoadForReader(
    const std::string& table_url,
    int64_t version,
    const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
    const FilesystemBinding& binding) {
  return LoadSnapshot(table_url, version, filesystem, binding, ffi::delta_load_snapshot);
}

arrow::Result<std::shared_ptr<arrow::Schema>> BlockingDeltaSnapshot::Schema() const {
  return CatchRustResult<std::shared_ptr<arrow::Schema>>(
      DeltaErrorStatus, [&]() -> arrow::Result<std::shared_ptr<arrow::Schema>> {
        ArrowSchema schema{};
        ArrowCDataReleaseGuard guard(&schema);
        impl_->delta_snapshot_export_schema(reinterpret_cast<uintptr_t>(&schema));
        return arrow::ImportSchema(&schema);
      });
}

uint64_t BlockingDeltaSnapshot::MemorySize() const { return impl_->delta_snapshot_memory_size(); }

bool BlockingDeltaSnapshot::HasPartitionColumns() const { return impl_->delta_snapshot_has_partition_columns(); }

arrow::Result<std::shared_ptr<BlockingDeltaFile>> BlockingDeltaSnapshot::LoadFile(const std::string& location,
                                                                                  const std::string& footer,
                                                                                  uint64_t physical_rows,
                                                                                  uint64_t live_rows,
                                                                                  uint64_t size) const {
  return CatchRustResult<std::shared_ptr<BlockingDeltaFile>>(
      DeltaErrorStatus, [&]() -> arrow::Result<std::shared_ptr<BlockingDeltaFile>> {
        auto file = impl_->delta_load_file(
            location, rust::Slice<const uint8_t>(reinterpret_cast<const uint8_t*>(footer.data()), footer.size()),
            physical_rows, live_rows, size);
        return std::make_shared<BlockingDeltaFile>(std::move(file));
      });
}

std::span<const uint64_t> BlockingDeltaFile::Deletions() const {
  auto positions = impl_->delta_file_deletions();
  return {positions.data(), positions.size()};
}
uint64_t BlockingDeltaFile::PhysicalRows() const { return impl_->delta_file_rows(); }
uint64_t BlockingDeltaFile::LiveRows() const { return impl_->delta_file_live_rows(); }

arrow::Result<std::shared_ptr<BlockingDeltaReadPlan>> BlockingDeltaFile::Plan(
    const BlockingDeltaSnapshot& snapshot, const std::vector<std::string>& names) const {
  return CatchRustResult<std::shared_ptr<BlockingDeltaReadPlan>>(DeltaErrorStatus, [&]() {
    rust::Vec<rust::String> columns;
    for (const auto& name : names) {
      columns.push_back(rust::String(name));
    }
    return std::make_shared<BlockingDeltaReadPlan>(impl_->delta_file_plan(*snapshot.impl_, std::move(columns)));
  });
}

std::vector<int> BlockingDeltaReadPlan::Leaves() const {
  auto leaves = impl_->delta_plan_leaves();
  return {leaves.begin(), leaves.end()};
}

arrow::Result<std::shared_ptr<arrow::Schema>> BlockingDeltaReadPlan::Schema() const {
  return CatchRustResult<std::shared_ptr<arrow::Schema>>(
      DeltaErrorStatus, [&]() -> arrow::Result<std::shared_ptr<arrow::Schema>> {
        ArrowSchema schema{};
        ArrowCDataReleaseGuard guard(&schema);
        impl_->delta_plan_export_schema(reinterpret_cast<uintptr_t>(&schema));
        return arrow::ImportSchema(&schema);
      });
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> BlockingDeltaReadPlan::Apply(
    const std::shared_ptr<arrow::RecordBatch>& batch) const {
  return CatchRustResult<std::shared_ptr<arrow::RecordBatch>>(
      DeltaErrorStatus, [&]() -> arrow::Result<std::shared_ptr<arrow::RecordBatch>> {
        ArrowArray input{}, output{};
        ArrowSchema input_schema{}, output_schema{};
        ArrowCDataReleaseGuard in_array(&input), out_array(&output);
        ArrowCDataReleaseGuard in_schema(&input_schema), out_schema(&output_schema);
        ARROW_RETURN_NOT_OK(arrow::ExportRecordBatch(*batch, &input, &input_schema));
        impl_->delta_plan_apply(reinterpret_cast<uintptr_t>(&input), reinterpret_cast<uintptr_t>(&input_schema),
                                reinterpret_cast<uintptr_t>(&output), reinterpret_cast<uintptr_t>(&output_schema));
        return arrow::ImportRecordBatch(&output, &output_schema);
      });
}

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
