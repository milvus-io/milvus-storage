#include "file/file.h"

namespace milvus_storage {

File::File(std::string& file_path, FileType file_type) : file_path_(std::move(file_path)), file_type_(file_type) {}

bool File::is_vector() const { return file_type_ == FileType::kVector; }

bool File::is_scalar() const { return file_type_ == FileType::kScalar; }

bool File::is_delete() const { return file_type_ == FileType::kDelete; }

std::string& File::path() { return file_path_; }
}  // namespace milvus_storage