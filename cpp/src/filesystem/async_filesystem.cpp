// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"

namespace milvus_storage {
namespace {
// Borrow the subtree chain only during synchronous dispatch. Async operations
// retain their own state; the traversal does not need shared ownership.
template <typename Call>
auto WithNativeS3(const std::shared_ptr<arrow::fs::FileSystem>& fs, std::string path, Call call)
    -> decltype(call(std::declval<S3FileSystem&>(), path)) {
  auto* current = fs.get();
  while (auto* subtree = dynamic_cast<arrow::fs::SubTreeFileSystem*>(current)) {
    if (!path.empty() && path.front() == '/')
      return arrow::Status::Invalid("Expected a relative subtree path");
    path = subtree->base_path() + path;
    current = subtree->base_fs().get();
  }
  if (auto* s3 = dynamic_cast<S3FileSystem*>(current))
    return call(*s3, path);
  return arrow::Status::NotImplemented("Filesystem has no native asynchronous S3 transport");
}
}  // namespace

arrow::Future<arrow::fs::FileInfo> FileSystemProxy::GetFileInfoAsync(const std::string& path) {
  ARROW_ASSIGN_OR_RAISE(auto full, PrependBase(path));
  return WithNativeS3(base_fs(), full, [](S3FileSystem& fs, const std::string& p) { return fs.GetFileInfoAsync(p); })
      .Then([path](arrow::fs::FileInfo info) {
        info.set_path(path);
        return info;
      });
}

arrow::Future<arrow::fs::FileInfoVector> FileSystemProxy::GetFileInfoAsync(const std::vector<std::string>& paths) {
  std::vector<std::string> full_paths;
  full_paths.reserve(paths.size());
  for (const auto& path : paths) {
    ARROW_ASSIGN_OR_RAISE(auto full, PrependBase(path));
    full_paths.push_back(std::move(full));
  }
  auto* fs = base_fs().get();
  while (auto* subtree = dynamic_cast<arrow::fs::SubTreeFileSystem*>(fs)) {
    for (auto& path : full_paths) path = subtree->base_path() + path;
    fs = subtree->base_fs().get();
  }
  return fs->GetFileInfoAsync(full_paths)
      .Then([paths](arrow::fs::FileInfoVector infos) -> arrow::Result<arrow::fs::FileInfoVector> {
        if (infos.size() != paths.size())
          return arrow::Status::IOError("Invalid batch stat result size");
        for (size_t i = 0; i < paths.size(); ++i) infos[i].set_path(paths[i]);
        return infos;
      });
}
arrow::Future<std::shared_ptr<arrow::io::OutputStream>> FileSystemProxy::OpenOutputStreamAsync(
    const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) {
  FIU_RETURN_ON(FIUKEY_FS_OPEN_OUTPUT_FAIL,
                arrow::Status::IOError("Injected output-stream open failure"));
  ARROW_ASSIGN_OR_RAISE(auto full, PrependBaseNonEmpty(path));
  return WithNativeS3(base_fs(), full, [&metadata](S3FileSystem& fs, const std::string& p) {
    return fs.OpenOutputStreamAsync(p, metadata);
  });
}
}  // namespace milvus_storage
