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

#pragma once

#include <memory>
#include <string>

#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/talon/talon_client.h"

namespace milvus_storage::talon {

/// A FileSystemProxy whose read openers serve from the Talon object cache while
/// every other operation (writes, listing, deletes, metrics, conditional and
/// sized uploads) is inherited unchanged and keeps going to the origin backend.
///
/// It subclasses FileSystemProxy rather than wrapping it so the milvus-storage
/// interfaces the proxy already exposes — UploadConditional, Observable,
/// UploadSizable, and the SubTree bucket rooting — continue to work with no
/// forwarding boilerplate. Only OpenInputFile / OpenInputStream are overridden.
///
/// Paths reaching these overrides are bucket-relative (the SubTree contract), so
/// the Talon object URI is `<scheme>://<bucket>/<path>`.
class TalonFileSystemProxy : public FileSystemProxy {
  public:
  TalonFileSystemProxy(const std::string& base_path,
                       std::shared_ptr<arrow::fs::FileSystem> base_fs,
                       std::shared_ptr<TalonClient> client,
                       std::string scheme,
                       std::string bucket)
      : FileSystemProxy(base_path, std::move(base_fs)),
        client_(std::move(client)),
        scheme_(std::move(scheme)),
        bucket_(std::move(bucket)) {}

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override;
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override;
  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const std::string& path) override;
  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const arrow::fs::FileInfo& info) override;

  private:
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenTalon(const std::string& path, int64_t size) const;
  std::string BuildUri(const std::string& path) const;

  std::shared_ptr<TalonClient> client_;
  std::string scheme_;
  std::string bucket_;
};

/// Wrap a producer-built remote filesystem so its reads go through Talon. The
/// input must be the FileSystemProxy a remote producer returns (bucket-rooted);
/// returns an error for local or otherwise unsupported backends, or when the
/// coordinator address is missing. Creates the Talon client bound to the
/// filesystem's lifetime.
arrow::Result<ArrowFileSystemPtr> WrapWithTalon(const ArrowFileSystemConfig& config, ArrowFileSystemPtr base_fs);

}  // namespace milvus_storage::talon
