// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <arrow/filesystem/filesystem.h>
#include <arrow/result.h>

#include "rust/cxx.h"
#include "rust-bridge/lib.h"

namespace milvus_storage::deltalake {

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

// Reader-owned shared metadata will retain this handle in Stage 2. Rust owns
// the engine, pinned snapshot and cached scan batches; no Kernel state is persisted.
class BlockingDeltaSnapshot {
  public:
  static arrow::Result<std::shared_ptr<BlockingDeltaSnapshot>> Open(
      const std::string& table_url,
      int64_t requested_version,
      const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
      const FilesystemBinding& binding);

  explicit BlockingDeltaSnapshot(rust::Box<ffi::DeltaSnapshotContext> impl) : impl_(std::move(impl)) {}

  uint64_t Version() const;
  std::vector<FileInfo> Files() const;

  private:
  rust::Box<ffi::DeltaSnapshotContext> impl_;
};

}  // namespace milvus_storage::deltalake
