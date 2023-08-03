#include "blob.h"
#include <memory>
#include "proto/manifest.pb.h"

namespace milvus_storage {

std::unique_ptr<manifest_proto::Blob> Blob::ToProtobuf() const {
  auto blob = std::make_unique<manifest_proto::Blob>();
  blob->set_name(name);
  blob->set_size(size);
  blob->set_file(file);
  return blob;
}

Blob Blob::FromProtobuf(const manifest_proto::Blob blob) {
  Blob ret;
  ret.name = blob.name();
  ret.size = blob.size();
  ret.file = blob.file();
  return ret;
}

}  // namespace milvus_storage
