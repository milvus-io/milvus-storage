#pragma once
#include <atomic>
#include <mutex>
#include "arrow/filesystem/filesystem.h"

#include "storage/manifest.h"
#include "storage/space.h"
#include "file/delete_fragment.h"
namespace milvus_storage {

class FilterQueryRecordReader;
class RecordReader;

class Space {
  public:
  Status Write(arrow::RecordBatchReader* reader, WriteOption* option);

  std::unique_ptr<arrow::RecordBatchReader> Read(std::shared_ptr<ReadOptions> option);

  Status Delete(arrow::RecordBatchReader* reader);

  // Open opened a space or create if the space does not exist.
  // If space does not exist. schema should not be nullptr, or an error will be returned.
  // If space exists and version is specified, it will restore to the state at this version,
  // or it will choose the latest version.
  static Result<std::unique_ptr<Space>> Open(const std::string& uri, Options options);

  private:
  Status Init();

  static Status SafeSaveManifest(std::shared_ptr<arrow::fs::FileSystem> fs,
                                 const std::string& path,
                                 const Manifest* manifest);

  static Result<arrow::fs::FileInfoVector> FindAllManifest(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                           const std::string& path);

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<Manifest> manifest_;
  std::string path_;

  DeleteFragmentVector delete_fragments_;

  int64_t next_manifest_version_ = 0;
  std::mutex mutex_;

  friend FilterQueryRecordReader;
  friend RecordReader;
};
}  // namespace milvus_storage
