#include "storage/schema.h"
#include "gtest/gtest.h"
#include "storage/options.h"
#include "storage/test_util.h"

namespace milvus_storage {

// TEST(SchemaValidateTest, SchemaValidateNoVersionColTest) {
//   std::vector<std::string> field_names = {"field1", "field2"};
//   std::vector<std::shared_ptr<arrow::DataType>> field_types = {
//       arrow::int64(),
//       arrow::fixed_size_binary(8),
//   };
//   std::shared_ptr<arrow::Schema> schema = CreateArrowSchema(field_names, field_types);
//   SchemaOptions options;
//   options.primary_column = "field1";
//   options.vector_column = "field2";
//
//   Schema schema_obj(schema, options);
//   Status status = schema_obj.Validate();
//   ASSERT_TRUE(status.ok());
//
//   auto scalar_schema = schema_obj.scalar_schema();
//   ASSERT_EQ(scalar_schema->num_fields(), 1);
//   ASSERT_EQ(scalar_schema->field(0)->name(), options.primary_column);
//
//   auto vector_schema = schema_obj.vector_schema();
//   ASSERT_EQ(vector_schema->num_fields(), 2);
//   ASSERT_EQ(vector_schema->field(0)->name(), options.primary_column);
//   ASSERT_EQ(vector_schema->field(1)->name(), options.vector_column);
//
//   auto delete_schema = schema_obj.delete_schema();
//   ASSERT_EQ(delete_schema->num_fields(), 1);
//   ASSERT_EQ(delete_schema->field(0)->name(), options.primary_column);
// }
//
// TEST(SchemaValidateTest, SchemaValidateVersionColTest) {
//   std::vector<std::string> field_names = {"field1", "field2", "field3"};
//   std::vector<std::shared_ptr<arrow::DataType>> field_types = {arrow::int64(), arrow::fixed_size_binary(8),
//                                                                arrow::int64()};
//   std::shared_ptr<arrow::Schema> schema = CreateArrowSchema(field_names, field_types);
//   SchemaOptions options;
//   options.primary_column = "field1";
//   options.vector_column = "field2";
//   options.version_column = "field3";
//
//   Schema schema_obj(schema, options);
//   auto status = schema_obj.Validate();
//   ASSERT_TRUE(status.ok());
//
//   auto scalar_schema = schema_obj.scalar_schema();
//   ASSERT_EQ(scalar_schema->num_fields(), 2);
//   ASSERT_EQ(scalar_schema->field(0)->name(), options.primary_column);
//   ASSERT_EQ(scalar_schema->field(1)->name(), options.version_column);
//
//   auto vector_schema = schema_obj.vector_schema();
//   ASSERT_EQ(vector_schema->num_fields(), 3);
//   auto n = vector_schema->num_fields();
//   ASSERT_EQ(vector_schema->field(0)->name(), options.primary_column);
//   ASSERT_EQ(vector_schema->field(1)->name(), options.vector_column);
//   ASSERT_EQ(vector_schema->field(2)->name(), options.version_column);
//
//   auto delete_schema = schema_obj.delete_schema();
//   ASSERT_EQ(delete_schema->num_fields(), 2);
//   ASSERT_EQ(delete_schema->field(0)->name(), options.primary_column);
//   ASSERT_EQ(delete_schema->field(1)->name(), options.version_column);
// }
//
}  // namespace milvus_storage
