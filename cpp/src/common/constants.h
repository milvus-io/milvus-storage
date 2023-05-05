#include <string>

namespace milvus_storage {

const int kReadBatchSize = 1024;

const std::string kManifestTempFileName = "manifest.tmp";
const std::string kManifestFileName = "manifest";
const std::string kParquetDataFileSuffix = ".parquet";
const std::string kOffsetFieldName = "__offset";

}  // namespace milvus_storage