#include <string>

#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/result.h>
#include <parquet/properties.h>

namespace milvus_storage {
class VortexReader {
  public:
  VortexReader(std::shared_ptr<arrow::fs::FileSystem> fs, std::string path);

  arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> TakeToRecordBatchs(const uint64_t* indices,
                                                                                     std::size_t size);

  arrow::Result<std::shared_ptr<arrow::Table>> TakeToTable(const uint64_t* indices, std::size_t size);

  void close();

  private:
  std::string path_;
};

}  // namespace milvus_storage