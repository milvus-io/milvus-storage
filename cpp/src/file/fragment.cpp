

#include "milvus-storage/file/fragment.h"
#include <memory>
#include <vector>
#include "assert.h"
#include "proto/manifest.pb.h"

namespace milvus_storage {
Fragment::Fragment(std::int64_t fragment_id) : fragment_id_(fragment_id) {}

void Fragment::add_file(const std::string& file) { files_.push_back(file); }

const std::vector<std::string>& Fragment::files() const { return files_; }

std::int64_t Fragment::id() const { return fragment_id_; }

void Fragment::set_id(int64_t id) { fragment_id_ = id; }

std::unique_ptr<manifest_proto::Fragment> Fragment::ToProtobuf() const {
  auto fragment_proto = std::make_unique<manifest_proto::Fragment>();
  for (const auto& file : files_) {
    fragment_proto->add_files(file);
  }
  fragment_proto->set_id(fragment_id_);
  return fragment_proto;
}

std::unique_ptr<Fragment> Fragment::FromProtobuf(const manifest_proto::Fragment& fragment) {
  auto res = std::make_unique<Fragment>(fragment.id());
  for (const auto& file : fragment.files()) {
    res->add_file(file);
  }
  return res;
}
}  // namespace milvus_storage
