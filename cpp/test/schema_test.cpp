#include "storage/schema.h"
#include "gtest/gtest.h"
#include "storage/options.h"
#include "storage/test_util.h"

namespace milvus_storage {

TEST(SchemaValidateTest, SchemaBuildSchemaTest) {
  std::vector<std::string> field_names = {"field1", "field2"};
  std::vector<std::shared_ptr<arrow::DataType>> field_types = {arrow::int64(), arrow::float32()};
  std::shared_ptr<arrow::Schema> schema = CreateArrowSchema(field_names, field_types);
  SchemaOptions options;
  options.primary_column = "field1";

  Schema schema_obj(schema, options);
  Status status = schema_obj.Validate();
  ASSERT_TRUE(status.ok());
}

}  // namespace milvus_storage
