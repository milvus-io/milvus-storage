#include "gtest/gtest.h"
#include "storage/options.h"
#include "arrow/type.h"
#include "storage/test_util.h"

namespace milvus_storage {
TEST(SchemaOptionsTest, EmptyPrimaryColumnTest) {
  SchemaOptions schema_options;
  auto status = schema_options.Validate(nullptr);
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(status.IsInvalidArgument());
  ASSERT_EQ("InvalidArgument: primary column is empty or not exist", status.ToString());
}

TEST(SchemaOptionsTest, PrimaryColTypeTest) {
  SchemaOptions schema_options;
  schema_options.primary_column = "field1";
  auto schema = CreateArrowSchema({"field1"}, {arrow::float32()});
  auto status = schema_options.Validate(schema.get());
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(status.IsInvalidArgument());
  ASSERT_EQ("InvalidArgument: primary column is not int64 or string", status.ToString());
}
}  // namespace milvus_storage