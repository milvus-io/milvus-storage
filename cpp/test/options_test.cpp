#include <google/protobuf/message.h>
#include "gtest/gtest.h"
#include "storage/options.h"
#include "arrow/type.h"
#include "test_util.h"
#include "google/protobuf/util/message_differencer.h"

namespace milvus_storage {
TEST(SchemaOptionsTest, PrimaryColumnExistTest) {
  SchemaOptions schema_options;
  // primary column is not set in options
  schema_options.vector_column = "field2";
  auto schema = CreateArrowSchema({"field1", "field2"}, {arrow::int64(), arrow::fixed_size_binary(8)});
  auto status = schema_options.Validate(schema.get());
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(status.IsInvalidArgument());
  ASSERT_EQ("InvalidArgument: primary column is empty", status.ToString());

  // primary column does not exist in schema
  schema_options.primary_column = "field0";
  status = schema_options.Validate(schema.get());
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(status.IsInvalidArgument());
  ASSERT_EQ("InvalidArgument: primary column is not exist", status.ToString());

  // primary column exists in schema
  schema_options.primary_column = "field1";
  status = schema_options.Validate(schema.get());
  ASSERT_TRUE(status.ok());
}

TEST(SchemaOptionsTest, PrimaryColTypeTest) {
  SchemaOptions schema_options;
  schema_options.primary_column = "field1";
  schema_options.vector_column = "field2";
  auto schema = CreateArrowSchema({"field1", "field2"}, {arrow::float32(), arrow::fixed_size_binary(8)});
  auto status = schema_options.Validate(schema.get());
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(status.IsInvalidArgument());
  ASSERT_EQ("InvalidArgument: primary column is not int64 or string", status.ToString());

  schema = CreateArrowSchema({"field1", "field2"}, {arrow::int64(), arrow::fixed_size_binary(8)});
  status = schema_options.Validate(schema.get());
  ASSERT_TRUE(status.ok());

  schema = CreateArrowSchema({"field1", "field2"}, {arrow::utf8(), arrow::fixed_size_binary(8)});
  status = schema_options.Validate(schema.get());
  ASSERT_TRUE(status.ok());
}

TEST(SchemaOptionsTest, VersionColTypeTest) {
  SchemaOptions schema_options;
  schema_options.primary_column = "field1";
  schema_options.version_column = "field2";
  schema_options.vector_column = "field3";
  auto schema = CreateArrowSchema({"field1", "field2", "field3"},
                                  {arrow::int64(), arrow::float32(), arrow::fixed_size_binary(8)});
  auto status = schema_options.Validate(schema.get());
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(status.IsInvalidArgument());
  ASSERT_EQ("InvalidArgument: version column is not int64", status.ToString());

  schema =
      CreateArrowSchema({"field1", "field2", "field3"}, {arrow::int64(), arrow::int64(), arrow::fixed_size_binary(8)});
  status = schema_options.Validate(schema.get());
  ASSERT_TRUE(status.ok());
}

TEST(SchemaOptionsTest, VectorColExistTest) {
  SchemaOptions schema_options;
  // vector column is not set in options
  schema_options.primary_column = "field1";
  auto schema = CreateArrowSchema({"field1", "field2"}, {arrow::int64(), arrow::fixed_size_binary(8)});
  auto status = schema_options.Validate(schema.get());
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(status.IsInvalidArgument());
  ASSERT_EQ("InvalidArgument: vector column is empty", status.ToString());

  // vector column does not exist in schema
  schema_options.vector_column = "field0";
  status = schema_options.Validate(schema.get());
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(status.IsInvalidArgument());
  ASSERT_EQ("InvalidArgument: vector column is not exist", status.ToString());

  // vector column exists in schema
  schema_options.vector_column = "field2";
  status = schema_options.Validate(schema.get());
  ASSERT_TRUE(status.ok());
}

TEST(SchemaOptionsTest, VectorColTypeTest) {
  SchemaOptions schema_options;
  schema_options.primary_column = "field1";
  schema_options.vector_column = "field2";
  auto schema = CreateArrowSchema({"field1", "field2"}, {arrow::int64(), arrow::int64()});
  auto status = schema_options.Validate(schema.get());
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(status.IsInvalidArgument());
  ASSERT_EQ("InvalidArgument: vector column is not fixed size binary or fixed size list", status.ToString());

  schema = CreateArrowSchema({"field1", "field2"}, {arrow::int64(), arrow::fixed_size_binary(8)});
  status = schema_options.Validate(schema.get());
  ASSERT_TRUE(status.ok());
}

TEST(SchemaOptionsTest, SchemaOptionsProtoTest) {
  SchemaOptions schema_options;
  schema_options.primary_column = "field1";
  schema_options.vector_column = "field2";
  schema_options.version_column = "field3";
  auto proto_schema = schema_options.ToProtobuf();

  schema_proto::SchemaOptions expected_proto_schema;
  expected_proto_schema.set_primary_column("field1");
  expected_proto_schema.set_vector_column("field2");
  expected_proto_schema.set_version_column("field3");
  ASSERT_TRUE(google::protobuf::util::MessageDifferencer::Equals(expected_proto_schema, *proto_schema.get()));

  SchemaOptions schema_options2;
  schema_options2.FromProtobuf(*proto_schema.get());
  ASSERT_EQ(schema_options, schema_options2);
}

}  // namespace milvus_storage
