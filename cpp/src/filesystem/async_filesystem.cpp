// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage {
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
}  // namespace milvus_storage
