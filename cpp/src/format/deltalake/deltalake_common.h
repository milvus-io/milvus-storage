// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz
#pragma once

#include "deltalake/deltalake_bridge.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::deltalake {

struct ResolvedLocation {
  ArrowFileSystemConfig config;
  std::shared_ptr<arrow::fs::FileSystem> filesystem;
  std::string url;
  // Local URLs use the configured namespace; config.root_path is physical.
  FilesystemBinding binding;
  std::string filesystem_key;
};

struct ResolvedDataFile {
  std::string url;
  std::string path;
};

arrow::Result<ResolvedLocation> ResolveLocation(const std::string& location,
                                                const std::shared_ptr<const api::Properties>& properties);
arrow::Result<std::string> ToExternalUri(const std::string& uri, const ArrowFileSystemConfig& config);
arrow::Result<ResolvedDataFile> ResolveDataFile(const std::string& path,
                                                const ResolvedLocation& table,
                                                const std::shared_ptr<const api::Properties>& properties);

uint64_t LogicalToPhysical(uint64_t logical, std::span<const uint64_t> deleted);

}  // namespace milvus_storage::deltalake
