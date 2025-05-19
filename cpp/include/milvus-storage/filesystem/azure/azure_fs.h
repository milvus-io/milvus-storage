

#pragma once
#ifdef MILVUS_AZURE_FS

#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage {

class AzureFileSystemProducer : public FileSystemProducer {
  public:
  AzureFileSystemProducer(const ArrowFileSystemConfig& config) : config_(config) {}

  Result<ArrowFileSystemPtr> Make() override;

  private:
  const ArrowFileSystemConfig config_;
};

}  // namespace milvus_storage
#endif
