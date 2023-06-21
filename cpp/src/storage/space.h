#pragma once
#include "arrow/filesystem/filesystem.h"

#include "storage/manifest.h"
#include "storage/space.h"
#include "file/delete_fragment.h"
namespace milvus_storage {

class FilterQueryRecordReader;
class RecordReader;

// some comment
// comment
class Space {
  public:
  Status Init();

  Status Write(arrow::RecordBatchReader* reader, WriteOption* option);

  std::unique_ptr<arrow::RecordBatchReader> Read(std::shared_ptr<ReadOptions> option);

  Status Delete(arrow::RecordBatchReader* reader);

  static std::unique_ptr<Space> Open(std::shared_ptr<arrow::fs::FileSystem> fs, std::string path);

  static std::unique_ptr<Space> Open();

  static std::unique_ptr<Space> Create();

  private:
  Space(std::shared_ptr<Schema> schema, std::shared_ptr<Options>& options);

  Status SafeSaveManifest(const Manifest* manifest);

  std::string base_path_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<Schema> schema_;

  DeleteFragmentVector delete_fragments_;
  std::shared_ptr<Manifest> manifest_;
  std::shared_ptr<Options> options_;

  friend FilterQueryRecordReader;
  friend RecordReader;
};
}  // namespace milvus_storage
