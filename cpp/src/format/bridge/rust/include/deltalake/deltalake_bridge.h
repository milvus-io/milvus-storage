// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <span>

#include <arrow/filesystem/filesystem.h>
#include <arrow/result.h>
#include <arrow/record_batch.h>

#include "rust/cxx.h"
#include "rust-bridge/lib.h"

namespace milvus_storage::deltalake {

class BlockingDeltaFile;
class BlockingDeltaReadPlan;

struct FilesystemBinding {
  bool is_local = false;
  std::string scheme;
  std::string bucket;
  std::string local_root;
};

struct FileInfo {
  std::string path;
  uint64_t file_size;
  uint64_t row_count;
};

// Reader-owned metadata retains this handle. Rust owns the engine, pinned
// snapshot and cached scan batches; no Kernel state is persisted.
class BlockingDeltaSnapshot {
  public:
  static arrow::Result<std::shared_ptr<BlockingDeltaSnapshot>> Open(
      const std::string& table_url,
      int64_t requested_version,
      const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
      const FilesystemBinding& binding);
  static arrow::Result<std::shared_ptr<BlockingDeltaSnapshot>> LoadForReader(
      const std::string& table_url,
      int64_t version,
      const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
      const FilesystemBinding& binding);

  explicit BlockingDeltaSnapshot(rust::Box<ffi::DeltaSnapshotContext> impl) : impl_(std::move(impl)) {}

  uint64_t Version() const;
  std::vector<FileInfo> Files() const;
  arrow::Result<std::shared_ptr<arrow::Schema>> Schema() const;
  uint64_t MemorySize() const;
  bool HasPartitionColumns() const;
  arrow::Result<std::shared_ptr<BlockingDeltaFile>> LoadFile(const std::string& location,
                                                             const std::shared_ptr<arrow::Schema>& file_schema,
                                                             uint64_t physical_rows,
                                                             uint64_t live_rows,
                                                             uint64_t size) const;
  friend class BlockingDeltaFile;

  private:
  rust::Box<ffi::DeltaSnapshotContext> impl_;
};

class BlockingDeltaFile {
  public:
  explicit BlockingDeltaFile(rust::Box<ffi::DeltaFileContext> impl) : impl_(std::move(impl)) {}
  std::span<const uint64_t> Deletions() const;
  uint64_t PhysicalRows() const;
  uint64_t LiveRows() const;
  uint64_t MemorySize() const;
  arrow::Result<std::shared_ptr<BlockingDeltaReadPlan>> Plan(const BlockingDeltaSnapshot& snapshot,
                                                             const std::vector<std::string>& names) const;

  private:
  rust::Box<ffi::DeltaFileContext> impl_;
};

class BlockingDeltaReadPlan {
  public:
  explicit BlockingDeltaReadPlan(rust::Box<ffi::DeltaReadPlan> impl) : impl_(std::move(impl)) {}
  std::vector<std::string> Columns() const;
  arrow::Result<std::shared_ptr<arrow::Schema>> Schema() const;
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Apply(const std::shared_ptr<arrow::RecordBatch>& batch) const;

  private:
  rust::Box<ffi::DeltaReadPlan> impl_;
};

}  // namespace milvus_storage::deltalake
