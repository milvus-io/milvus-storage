#pragma once
#include <memory>
#include <unordered_map>
#include <variant>
#include "file/fragment.h"
#include "common/result.h"
#include "storage/schema.h"
#include "arrow/filesystem/filesystem.h"

namespace milvus_storage {

using pk_type = std::variant<std::string, std::int64_t>;

// DeleteFragment is a set of deleted records
class DeleteFragment {
  public:
  bool id() { return fragment_.id(); }

  // Return true if this pk at this version have been deleted
  bool Filter(pk_type& pk, std::int64_t version);

  // Return true if this pk have been deleted
  bool Filter(pk_type& pk);

  // Make an instance of DeleteFragment of the given fragment whose type is kDelete
  static Result<std::shared_ptr<DeleteFragment>> Make(std::shared_ptr<arrow::fs::FileSystem> fs, Fragment& fragment);

  private:
  DeleteFragment(Fragment& fragment);

  Fragment& fragment_;

  // the deleted data parsed from the files of fragment_
  std::unordered_map<pk_type, std::vector<int64_t>> data_;  // pk to versions(if exists)
};

using DeleteFragmentVector = std::vector<DeleteFragment>;

}  // namespace milvus_storage