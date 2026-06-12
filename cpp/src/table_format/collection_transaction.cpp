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

#include "milvus-storage/table_format/collection_transaction.h"

#include <sstream>

#include <arrow/buffer.h>
#include <arrow/filesystem/filesystem.h>
#include <fmt/format.h>

#include "milvus-storage/common/path_util.h"
#include "milvus-storage/filesystem/upload_conditional.h"
#include "milvus-storage/table_format/layout.h"
#include "milvus-storage/table_format/manifest_list.h"
#include "milvus-storage/table_format/metadata.h"

namespace milvus_storage::api::table_format {

static bool IsRetriable(const arrow::Status& s) { return s.IsIOError(); }

// ---- Filesystem helpers (private to this TU) ----

static arrow::Status WriteFileCAS(const milvus_storage::ArrowFileSystemPtr& fs,
                                  const std::string& path,
                                  const std::string& data) {
  auto conditional_fs = std::dynamic_pointer_cast<milvus_storage::UploadConditional>(fs);
  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> res;
  if (conditional_fs) {
    res = conditional_fs->OpenConditionalOutputStream(path, nullptr);
  } else {
    res = arrow::Status::NotImplemented("Filesystem does not support conditional writes");
  }

  if (res.ok()) {
    auto output_stream = res.ValueOrDie();
    ARROW_RETURN_NOT_OK(output_stream->Write(data.data(), static_cast<int64_t>(data.size())));
    auto result = output_stream->Close();
    if (!result.ok()) {
      if (result.code() == arrow::StatusCode::IOError) {
        return arrow::Status::AlreadyExists("File already exists: ", path);
      }
      return result;
    }
    return arrow::Status::OK();
  }

  // Fallback: check existence, then write
  ARROW_ASSIGN_OR_RAISE(auto file_info, fs->GetFileInfo(path));
  if (file_info.type() != arrow::fs::FileType::NotFound) {
    return arrow::Status::AlreadyExists("File already exists: ", path);
  }

  auto [parent, _] = milvus_storage::GetAbstractPathParent(path);
  if (!parent.empty()) {
    ARROW_RETURN_NOT_OK(fs->CreateDir(parent));
  }

  ARROW_ASSIGN_OR_RAISE(auto output_stream, fs->OpenOutputStream(path));
  ARROW_RETURN_NOT_OK(output_stream->Write(data.data(), static_cast<int64_t>(data.size())));
  ARROW_RETURN_NOT_OK(output_stream->Close());
  return arrow::Status::OK();
}

static arrow::Result<std::shared_ptr<arrow::Buffer>> ReadFileBuffer(const milvus_storage::ArrowFileSystemPtr& fs,
                                                                    const std::string& path) {
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(path));
  ARROW_ASSIGN_OR_RAISE(int64_t file_size, input_file->GetSize());
  ARROW_ASSIGN_OR_RAISE(auto buffer, input_file->Read(file_size));
  if (buffer->size() != file_size) {
    return arrow::Status::IOError(
        fmt::format("Failed to read complete file, expected={}, actual={}", file_size, buffer->size()));
  }
  ARROW_RETURN_NOT_OK(input_file->Close());
  return buffer;
}

static arrow::Status WriteMetadata(const milvus_storage::ArrowFileSystemPtr& fs,
                                   const std::string& base_path,
                                   int64_t version,
                                   const Metadata& metadata) {
  std::ostringstream oss;
  ARROW_RETURN_NOT_OK(metadata.serialize(oss));
  std::string data = oss.str();

  std::string filepath = GetCollMetadataFilepath(base_path, version);
  return WriteFileCAS(fs, filepath, data);
}

static arrow::Result<Metadata> ReadMetadata(const milvus_storage::ArrowFileSystemPtr& fs,
                                            const std::string& base_path,
                                            int64_t version) {
  std::string filepath = GetCollMetadataFilepath(base_path, version);
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadFileBuffer(fs, filepath));

  std::string data(reinterpret_cast<const char*>(buffer->data()), buffer->size());
  std::istringstream iss(data);

  Metadata metadata;
  ARROW_RETURN_NOT_OK(metadata.deserialize(iss));
  return metadata;
}

static arrow::Result<std::pair<Metadata, int64_t>> ReadLatestMetadata(const milvus_storage::ArrowFileSystemPtr& fs,
                                                                       const std::string& base_path) {
  ARROW_ASSIGN_OR_RAISE(auto version, GetLatestMetadataVersion(fs, base_path));
  if (version == 0) {
    return arrow::Status::IOError("No metadata files found");
  }
  ARROW_ASSIGN_OR_RAISE(auto metadata, ReadMetadata(fs, base_path, version));
  return std::make_pair(std::move(metadata), version);
}

// ---- CollectionTransaction ----

CollectionTransaction::CollectionTransaction(const milvus_storage::ArrowFileSystemPtr& fs,
                                             const std::string& base_path,
                                             int64_t read_version,
                                             Metadata metadata,
                                             uint32_t retry_limit)
    : fs_(fs),
      base_path_(base_path),
      read_version_(read_version),
      metadata_(std::move(metadata)),
      retry_limit_(retry_limit) {}

arrow::Result<std::unique_ptr<CollectionTransaction>> CollectionTransaction::Open(
    const milvus_storage::ArrowFileSystemPtr& fs,
    const std::string& base_path,
    int64_t version,
    uint32_t retry_limit) {
  Metadata metadata;
  int64_t actual_version = 0;

  if (version == LATEST_VERSION) {
    auto result = ReadLatestMetadata(fs, base_path);
    if (result.ok()) {
      auto [md, ver] = std::move(result).ValueOrDie();
      metadata = std::move(md);
      actual_version = ver;
    }
    // If no metadata, start with empty metadata and version 0
  } else {
    auto result = ReadMetadata(fs, base_path, version);
    if (!result.ok()) {
      return result.status();
    }
    metadata = std::move(result).ValueOrDie();
    actual_version = version;
  }

  return std::unique_ptr<CollectionTransaction>(
      new CollectionTransaction(fs, base_path, actual_version, std::move(metadata), retry_limit));
}

Metadata& CollectionTransaction::GetSnapshot() { return metadata_; }

const Metadata& CollectionTransaction::GetMetadata() const { return metadata_; }

int64_t CollectionTransaction::GetReadVersion() const { return read_version_; }

arrow::Result<const SnapshotEntry*> CollectionTransaction::GetCurrentSnapshot() const {
  for (const auto& snap : metadata_.snapshots()) {
    if (snap.snapshot_id == metadata_.current_snapshot_id()) {
      return &snap;
    }
  }
  return arrow::Status::Invalid("Current snapshot not found");
}

arrow::Result<const SnapshotEntry*> CollectionTransaction::GetSnapshot(int64_t snapshot_id) const {
  for (const auto& snap : metadata_.snapshots()) {
    if (snap.snapshot_id == snapshot_id) {
      return &snap;
    }
  }
  return arrow::Status::Invalid(fmt::format("Snapshot {} not found", snapshot_id));
}

arrow::Result<const SnapshotEntry*> CollectionTransaction::GetSnapshotAtTime(int64_t timestamp_ms) const {
  const SnapshotEntry* best = nullptr;
  for (const auto& snap : metadata_.snapshots()) {
    if (snap.timestamp_ms <= timestamp_ms) {
      if (!best || snap.timestamp_ms > best->timestamp_ms ||
          (snap.timestamp_ms == best->timestamp_ms && snap.snapshot_id > best->snapshot_id)) {
        best = &snap;
      }
    }
  }
  if (!best) {
    return arrow::Status::Invalid(fmt::format("No snapshot found at or before timestamp {}", timestamp_ms));
  }
  return best;
}

arrow::Result<const SchemaInfo*> CollectionTransaction::GetSchema(int32_t schema_id) const {
  for (const auto& schema : metadata_.schemas()) {
    if (schema.schema_id == schema_id) {
      return &schema;
    }
  }
  return arrow::Status::Invalid(fmt::format("Schema {} not found", schema_id));
}

arrow::Result<const IndexSpec*> CollectionTransaction::GetIndexSpec(int32_t spec_id) const {
  for (const auto& spec : metadata_.index_specs()) {
    if (spec.spec_id == spec_id) {
      return &spec;
    }
  }
  return arrow::Status::Invalid(fmt::format("IndexSpec {} not found", spec_id));
}

arrow::Result<std::vector<ManifestListEntry>> CollectionTransaction::LoadManifestEntries(
    const std::vector<ManifestListInfo>& manifest_lists) const {
  std::vector<ManifestListEntry> all_entries;

  for (const auto& ml_ref : manifest_lists) {
    ARROW_ASSIGN_OR_RAISE(auto ml, ReadManifestListFromFile(fs_, ml_ref.manifest_list));
    for (auto& entry : ml.entries()) {
      all_entries.push_back(std::move(entry));
    }
  }

  return all_entries;
}

arrow::Result<std::vector<ManifestListEntry>> CollectionTransaction::ListSegments(
    const SnapshotEntry& snapshot) const {
  return LoadManifestEntries(snapshot.manifest_lists);
}

arrow::Result<std::vector<SegmentInfo>> CollectionTransaction::ListSegments(const SnapshotEntry& snapshot,
                                                                               int64_t partition_id) const {
  ARROW_ASSIGN_OR_RAISE(auto all_entries, ListSegments(snapshot));

  std::vector<SegmentInfo> result;
  for (const auto& entry : all_entries) {
    if (entry.partition_id == partition_id) {
      for (const auto& seg : entry.segments) {
        result.push_back(seg);
      }
    }
  }
  return result;
}

arrow::Result<int64_t> CollectionTransaction::Commit(std::shared_ptr<Action> action) {
  uint32_t retry_count = 0;
  while (retry_count <= retry_limit_) {
    ARROW_ASSIGN_OR_RAISE(auto latest_version, GetLatestMetadataVersion(fs_, base_path_));

    // Start from the appropriate base metadata
    Metadata commit_md;
    if (latest_version != read_version_ && latest_version != 0) {
      ARROW_ASSIGN_OR_RAISE(commit_md, ReadMetadata(fs_, base_path_, latest_version));
    } else {
      commit_md = metadata_;
    }
    size_t base_snap_count = commit_md.snapshots().size();

    // Apply the action
    auto status = action->Apply(commit_md);
    if (!status.ok()) {
      if (!IsRetriable(status)) {
        return status;  // Unretriable (validation), abort immediately
      }
      retry_count++;
      if (retry_count > retry_limit_) {
        return arrow::Status::Invalid(
            fmt::format("Commit failed: exceeded retry limit of {} attempts", retry_limit_));
      }
      continue;
    }

    // The action must produce at least one new snapshot
    if (commit_md.snapshots().size() <= base_snap_count) {
      return arrow::Status::Invalid("Commit failed: action did not produce a new snapshot");
    }

    // Squash intermediate snapshots: keep base snapshots + one final snapshot
    auto& snaps = commit_md.mutable_snapshots();
    if (snaps.size() > base_snap_count + 1) {
      auto final_snap = std::move(snaps.back());
      snaps.resize(base_snap_count);
      if (base_snap_count > 0) {
        final_snap.parent_snapshot_id = snaps.back().snapshot_id;
      }
      snaps.push_back(std::move(final_snap));
    }
    commit_md.set_current_snapshot_id(snaps.back().snapshot_id);

    // Write metadata file (CAS)
    int64_t new_version = latest_version + 1;
    auto write_status = WriteMetadata(fs_, base_path_, new_version, commit_md);

    if (write_status.ok()) {
      return new_version;
    }

    if (write_status.IsAlreadyExists()) {
      retry_count++;
      if (retry_count > retry_limit_) {
        return arrow::Status::Invalid(fmt::format("Commit failed: exceeded retry limit of {} attempts", retry_limit_));
      }
      continue;
    }

    return write_status;
  }

  return arrow::Status::Invalid("Commit failed: unexpected retry loop exit");
}

}  // namespace milvus_storage::api::table_format
