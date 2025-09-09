
#include "milvus-storage/vortex/VortexReader.h"
#include "vortex.hpp"
#include <arrow/c/bridge.h>
#include <arrow/chunked_array.h>
#include <arrow/array.h>

namespace milvus_storage {

VortexReader::VortexReader(std::shared_ptr<arrow::fs::FileSystem> /*fs*/, std::string path) : path_(std::move(path)) {}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> VortexReader::TakeToRecordBatchs(
    const uint64_t* indices, std::size_t size) {
  auto file = vortex::VortexFile::Open(path_);
  auto scan_builder = file.CreateScanBuilder().WithIncludeByIndex(indices, size).IntoStream();

  auto rb_reader = *arrow::ImportRecordBatchReader(&scan_builder);
  auto rbs = rb_reader->ToRecordBatches();
  rb_reader->Close();
  return rbs;
}

// There still remain lots of interface have not export in C++ implments
arrow::Result<std::shared_ptr<arrow::Table>> VortexReader::TakeToTable(const uint64_t* indices, std::size_t size) {
  auto file = vortex::VortexFile::Open(path_);
  auto scan_builder = file.CreateScanBuilder().WithIncludeByIndex(indices, size).IntoStream();

  auto rb_reader = *arrow::ImportRecordBatchReader(&scan_builder);
  auto tb = rb_reader->ToTable();
  rb_reader->Close();
  return tb;
}

void VortexReader::close() {
  // nothing todo
}

}  // namespace milvus_storage