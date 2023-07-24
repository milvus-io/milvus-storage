#pragma once
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
  Space(std::shared_ptr<Schema> schema, std::shared_ptr<Options>& options);

  Status Init();

  Status Write(arrow::RecordBatchReader* reader, WriteOption* option);

  std::unique_ptr<arrow::RecordBatchReader> Read(std::shared_ptr<ReadOptions> option);

  Status Delete(arrow::RecordBatchReader* reader);

  // Open opened a existed space. It will return a error in status if a space is not existed in path. If version is
  // specified, it will restore to the state at this version. If not, it will choose the latest version.
  static std::unique_ptr<Space> Open(std::shared_ptr<arrow::fs::FileSystem> fs, std::string path, int64_t version = -1);

  // Create created a new space. If there is already a space, a error will be returned.
  static std::unique_ptr<Space> Create();

  private:
  Status SafeSaveManifest(const Manifest* manifest);

  std::string base_path_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<Schema> schema_;

  DeleteFragmentVector delete_fragments_;
  std::shared_ptr<Manifest> manifest_;
  std::shared_ptr<Options> options_;

  std::mutex mutex_;

  friend FilterQueryRecordReader;
  friend RecordReader;
};
}  // namespace milvus_storage
