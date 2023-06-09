#include "arrow/type.h"
#include <memory>
#include <string>
#include <unordered_map>
#include "proto/schema.pb.h"
#include "result.h"

namespace milvus_storage {

Result<std::unique_ptr<schema_proto::ArrowSchema>> ToProtobufSchema(const arrow::Schema* schema);

Result<std::shared_ptr<arrow::Schema>> FromProtobufSchema(const schema_proto::ArrowSchema& schema);

std::string GetNewParquetFilePath(std::string& path);

std::string GetManifestFilePath(std::string& path);

std::string GetManifestTmpFilePath(std::string& path);

Result<std::shared_ptr<arrow::Schema>> ProjectSchema(std::shared_ptr<arrow::Schema> schema,
                                                     std::vector<std::string> columns);
}  // namespace milvus_storage
