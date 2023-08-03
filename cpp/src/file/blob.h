#pragma once

#include <string>
#include <vector>
#include "proto/manifest.pb.h"
#include "common/result.h"

namespace milvus_storage {
struct Blob {
  std::string name;
  int64_t size;
  std::string file;

  [[nodiscard]] std::unique_ptr<manifest_proto::Blob> ToProtobuf() const;
  static Blob FromProtobuf(const manifest_proto::Blob blob);
};

using BlobVector = std::vector<Blob>;
}  // namespace milvus_storage
