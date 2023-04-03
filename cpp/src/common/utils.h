#include "arrow/type.h"
#include <memory>
#include <unordered_map>
#include "proto/schema.pb.h"

std::unique_ptr<schema::Schema>
ToProtobufSchema(arrow::Schema* schema);

std::shared_ptr<arrow::Schema>
FromProtobufSchema(schema::Schema schema);
