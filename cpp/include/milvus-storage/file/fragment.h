#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <file/file.h>
#include "proto/manifest.pb.h"

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
  Fragment() = default;
  explicit Fragment(std::int64_t fragment_id);

  void add_file(const std::string& file);

  const std::vector<std::string>& files() const;

  std::int64_t id() const;

  std::unique_ptr<manifest_proto::Fragment> ToProtobuf() const;

  static std::unique_ptr<Fragment> FromProtobuf(const manifest_proto::Fragment& fragment);

  void set_id(int64_t id);

  bool operator==(const Fragment& other) const {
    return (fragment_id_ == other.fragment_id_ && files_ == other.files_);
  }

  private:
  std::int64_t fragment_id_;
  std::vector<std::string> files_;
};

using FragmentVector = std::vector<Fragment>;
}  // namespace milvus_storage
