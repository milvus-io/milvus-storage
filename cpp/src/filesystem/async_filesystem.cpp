// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#include "milvus-storage/filesystem/async_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include <arrow/filesystem/path_util.h>

namespace milvus_storage {
namespace {
class AsyncSubTree final : public AsyncFileSystem {
  public:
  AsyncSubTree(std::string base, std::shared_ptr<AsyncFileSystem> underlying)
      : base_(std::move(base)), underlying_(std::move(underlying)) {}
  arrow::Future<arrow::fs::FileInfo> GetFileInfoAsync(const std::string& path) override {
    return underlying_->GetFileInfoAsync(Full(path)).Then([path](arrow::fs::FileInfo info) {
      info.set_path(path);
      return info;
    });
  }
  arrow::fs::FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& selector) override {
    auto full = selector;
    full.base_dir = Full(selector.base_dir);
    auto generator = underlying_->GetFileInfoGenerator(full);
    return [base = base_, generator = std::move(generator)]() mutable {
      return generator().Then([base](arrow::fs::FileInfoVector infos) -> arrow::Result<arrow::fs::FileInfoVector> {
        for (auto& info : infos) {
          if (info.path().compare(0, base.size(), base) != 0)
            return arrow::Status::IOError("S3 listing escaped subtree");
          info.set_path(info.path().substr(base.size()));
        }
        return infos;
      });
    };
  }
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(const std::string& path) override {
    return underlying_->ReadMetadataAsync(Full(path));
  }
  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const std::string& path,
                                                          int64_t offset,
                                                          int64_t nbytes) override {
    return underlying_->ReadAsync(Full(path), offset, nbytes);
  }

  private:
  std::string Full(const std::string& path) const { return base_ + path; }
  std::string base_;
  std::shared_ptr<AsyncFileSystem> underlying_;
};
}  // namespace

arrow::Result<std::shared_ptr<AsyncFileSystem>> MakeAsyncFileSystem(std::shared_ptr<arrow::fs::FileSystem> filesystem,
                                                                    const arrow::io::IOContext& io_context) {
  if (!filesystem || !io_context.executor())
    return arrow::Status::Invalid("Filesystem and executor are required");
  if (auto subtree = std::dynamic_pointer_cast<arrow::fs::SubTreeFileSystem>(filesystem)) {
    ARROW_ASSIGN_OR_RAISE(auto underlying, MakeAsyncFileSystem(subtree->base_fs(), io_context));
    return std::make_shared<AsyncSubTree>(subtree->base_path(), std::move(underlying));
  }
  if (auto s3 = std::dynamic_pointer_cast<S3FileSystem>(filesystem))
    return s3->MakeAsync(io_context);
  return arrow::Status::NotImplemented("Filesystem has no native asynchronous object transport");
}
}  // namespace milvus_storage
