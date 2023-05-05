#pragma once

#include <cstdint>
#include <vector>
#include <file/file.h>

namespace milvus_storage {
enum class FragmentType {
  kUnknown,
  kData,
  kDelete,
};

// Fragment is a block of data, which contains multiple files.
// For data fragment type, it contains vector files and scalar files.
// For delete fragment type, it contains delete files.
class Fragment {
  public:
  Fragment(std::int64_t fragment_id, FragmentType fragment_type);

  void add_file(File& file);

  bool is_data() const;

  bool is_delete() const;

  std::vector<File> get_vector_files();

  std::vector<File> get_scalar_files();

  std::vector<File> get_delete_files();

  std::int64_t id();

  private:
  std::int64_t fragment_id_;
  FragmentType fragment_type_;
  std::vector<File> files_;
};
}  // namespace milvus_storage