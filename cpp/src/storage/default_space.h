#pragma once
#include "arrow/filesystem/filesystem.h"

#include "storage/manifest.h"
#include "storage/space.h"
#include "file/delete_fragment.h"
namespace milvus_storage {

class FilterQueryRecordReader;
class RecordReader;

class DefaultSpace : public Space {
  public:
  Status Init() override;

  Status Write(arrow::RecordBatchReader* reader, WriteOption* option) override;

  std::unique_ptr<arrow::RecordBatchReader> Read(std::shared_ptr<ReadOptions> option) override;

  Status Delete(arrow::RecordBatchReader* reader) override;

  static std::unique_ptr<DefaultSpace> Open(std::shared_ptr<arrow::fs::FileSystem> fs, std::string path);

  static std::unique_ptr<DefaultSpace> Open();

  static std::unique_ptr<DefaultSpace> Create();

  private:
  DefaultSpace(std::shared_ptr<Schema> schema, std::shared_ptr<SpaceOptions>& options);

  Status SafeSaveManifest(const Manifest* manifest);

  std::string base_path_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<Schema> schema_;

  DeleteFragmentVector delete_fragments_;
  std::shared_ptr<Manifest> manifest_;

  friend FilterQueryRecordReader;
  friend RecordReader;
};
}  // namespace milvus_storage
