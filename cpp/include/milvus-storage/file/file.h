

#pragma once

#include <cstdint>
#include <string>

namespace milvus_storage {
enum class FileType {
  kUnknown,
  kVector,
  kScalar,
  kDelete,
};

class File {
  public:
  File(std::string& file_path, FileType file_type);

  bool is_vector() const;

  bool is_scalar() const;

  bool is_delete() const;

  std::string& path();

  private:
  std::string file_path_;
  FileType file_type_;
};

}  // namespace milvus_storage