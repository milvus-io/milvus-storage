#include "arrow/type.h"
#include <memory>
#include <string>
#include <unordered_map>
#include "proto/schema.pb.h"

std::unique_ptr<schema::Schema>
ToProtobufSchema(arrow::Schema* schema);

std::shared_ptr<arrow::Schema>
FromProtobufSchema(schema::Schema schema);

std::string
GetNewParquetFile(std::string& path);

std::string
GetManifestFile(std::string& path);

std::string
GetManifestTmpFile(std::string& path);