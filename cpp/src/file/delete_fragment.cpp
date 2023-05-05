#include "file/delete_fragment.h"
#include "common/status.h"
#include "common/arrow_util.h"
#include "common/macro.h"
#include "storage/options.h"

namespace milvus_storage {
DeleteFragment::DeleteFragment(Fragment& fragment) : fragment_(fragment) {}

Result<std::shared_ptr<DeleteFragment>> DeleteFragment::Make(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                             Fragment& fragment) {
  auto delete_files = fragment.get_delete_files();
  if (delete_files.empty()) {
    return Status::InternalStateError("No delete files found");
  }

  auto delete_fragment = std::make_shared<DeleteFragment>(fragment);
  for (auto& delete_file : delete_files) {
    ASSIGN_OR_RETURN_NOT_OK(auto reader, MakeArrowFileReader(fs, delete_file.path()));
    ASSIGN_OR_RETURN_NOT_OK(auto record_batch_reader,
                            MakeArrowRecordBatchReader(reader, ReadOptions::default_read_options()));
  }
}
}  // namespace milvus_storage