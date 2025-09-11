
#include <string>
#include "milvus-storage/writer.h"
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/result.h>
#include <parquet/properties.h>

namespace milvus_storage {
using namespace milvus_storage::api;
class VortexWriter {
  public:
  VortexWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
               std::string base_path,
               std::shared_ptr<arrow::Schema> schema,
               WriteProperties properties = default_write_properties);

  ~VortexWriter() = default;

  arrow::Status write(const std::shared_ptr<arrow::RecordBatch>& batch);

  arrow::Status flush();

  void close();

  private:
  std::string path_;
  std::shared_ptr<arrow::Schema> schema_;
  WriteProperties properties_;

  // ::vortex::VortexFile vortex_file_;
};

}  // namespace milvus_storage