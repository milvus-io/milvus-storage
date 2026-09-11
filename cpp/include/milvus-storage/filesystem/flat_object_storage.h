// Copyright 2024 Zilliz
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
#include <vector>

#include <arrow/filesystem/filesystem.h>
#include <arrow/result.h>
#include <arrow/status.h>

namespace milvus_storage {

/// \brief Object-key-native operations for filesystems backed by a flat object
/// store (AWS S3, S3-compatible services, and GCS via the S3 interoperability
/// layer).
///
/// Arrow's FileSystem models a hierarchical namespace, which forces the S3
/// backend to emulate directories:
///   - GetFileInfo(FileSelector) enumerates a whole directory subtree;
///   - DeleteFile is directory-aware (HeadObject, then DeleteObject, then a
///     PutObject to recreate the parent "dir/" marker);
///   - GetFileInfo(path) reports a key that only has descendants (or a "key/"
///     marker) as a Directory rather than NotFound.
///
/// On a real object store those behaviours cost extra requests, mutate marker
/// objects, and break exact-object semantics. A caller that treats the store as
/// a flat key/value namespace (e.g. Milvus's ChunkManager) wants the raw
/// equivalents. Filesystems that can serve them natively implement this
/// interface; the free helpers below dispatch to it and otherwise fall back to
/// Arrow's directory semantics, so callers stay backend-agnostic.
class FlatObjectStorage {
  public:
  virtual ~FlatObjectStorage() = default;

  /// \brief List every object whose key begins with @prefix.
  ///
  /// The match is a raw object-key prefix (it may end mid-segment) evaluated
  /// server-side (S3 ListObjectsV2 with Prefix=). Unlike GetFileInfo(selector),
  /// it does NOT enumerate the prefix's parent directory subtree, so cost is
  /// proportional to the number of matching keys rather than to the parent's
  /// size. Every returned FileInfo is FileType::File; paths are in this
  /// filesystem's own namespace. Results are paginated internally.
  [[nodiscard]] virtual arrow::Result<std::vector<arrow::fs::FileInfo>> ListObjectsByPrefix(
      const std::string& prefix) = 0;

  /// \brief Delete exactly one object by key.
  ///
  /// Idempotent: deleting a key that does not exist is success. Issues a single
  /// DeleteObject with no preceding HeadObject and no parent directory-marker
  /// maintenance, unlike DeleteFile.
  [[nodiscard]] virtual arrow::Status DeleteObject(const std::string& path) = 0;

  /// \brief Whether @path names an existing object (exact-key HeadObject
  /// semantics).
  ///
  /// Only a real object stored at the exact key returns true. A key that merely
  /// has descendants, or a zero-byte "key/" directory marker, is NOT an object
  /// and returns false.
  [[nodiscard]] virtual arrow::Result<bool> ObjectExists(const std::string& path) = 0;
};

// ---------------------------------------------------------------------------
// Free helpers: dispatch to FlatObjectStorage when the filesystem implements it
// (AWS/S3-compatible/GCS), otherwise fall back to Arrow's directory semantics
// (local, Azure). Callers use these and stay backend-agnostic.
// ---------------------------------------------------------------------------

/// See FlatObjectStorage::ListObjectsByPrefix. Fallback lists the prefix's
/// parent directory recursively and filters file entries on the raw prefix.
[[nodiscard]] arrow::Result<std::vector<arrow::fs::FileInfo>> ListObjectsByPrefix(
    const std::shared_ptr<arrow::fs::FileSystem>& fs, const std::string& prefix);

/// See FlatObjectStorage::DeleteObject. Fallback uses DeleteFile and swallows a
/// not-found target so removal stays idempotent.
[[nodiscard]] arrow::Status DeleteObject(const std::shared_ptr<arrow::fs::FileSystem>& fs, const std::string& path);

/// See FlatObjectStorage::ObjectExists. Fallback uses GetFileInfo and treats
/// only FileType::File as an existing object.
[[nodiscard]] arrow::Result<bool> ObjectExists(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                               const std::string& path);

}  // namespace milvus_storage
