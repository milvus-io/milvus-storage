#include "file/fragment.h"
#include "assert.h"

namespace milvus_storage {
Fragment::Fragment(std::int64_t fragment_id, FragmentType fragment_type)
    : fragment_id_(fragment_id), fragment_type_(fragment_type) {}

void Fragment::add_file(File& file) { files_.push_back(file); }

bool Fragment::is_data() const { return fragment_type_ == FragmentType::kData; }

bool Fragment::is_delete() const { return fragment_type_ == FragmentType::kDelete; }

std::vector<File> Fragment::get_vector_files() {
  assert(is_data());
  std::vector<File> vector_files;
  for (auto& file : files_) {
    if (file.is_vector()) {
      vector_files.push_back(file);
    }
  }
  return vector_files;
}

std::vector<File> Fragment::get_scalar_files() {
  assert(is_data());
  std::vector<File> scalar_files;
  for (auto& file : files_) {
    if (file.is_scalar()) {
      scalar_files.push_back(file);
    }
  }
  return scalar_files;
}

std::vector<File> Fragment::get_delete_files() {
  assert(is_delete());
  std::vector<File> delete_files;
  for (auto& file : files_) {
    if (file.is_delete()) {
      delete_files.push_back(file);
    }
  }
  return delete_files;
}

std::int64_t Fragment::id() { return fragment_id_; }

}  // namespace milvus_storage
