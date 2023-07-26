#include "storage/space.h"
#include "filter/constant_filter.h"

#include <memory>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include <gtest/gtest.h>

namespace milvus_storage {
/**
 * @brief Test Space::Write
 *
 */
TEST(SpaceTest, SpaceWriteTest) {
  arrow::SchemaBuilder schema_builder;
  auto metadata = arrow::KeyValueMetadata::Make(std::vector<std::string>{"key1", "key2"},
                                                std::vector<std::string>{"value1", "value2"});

  auto pk_field = arrow::field("pk_field", arrow::int64(), /*nullable=*/false, metadata);

  auto ts_field = arrow::field("ts_field", arrow::int64(), /*nullable=*/false, metadata);

  auto vec_field = arrow::field("vec_field", arrow::fixed_size_binary(10), /*nullable=*/false, metadata);
  auto status = schema_builder.AddField(pk_field);
  ASSERT_TRUE(status.ok());
  status = schema_builder.AddField(ts_field);
  ASSERT_TRUE(status.ok());
  status = schema_builder.AddField(vec_field);
  ASSERT_TRUE(status.ok());

  auto schema_options = std::make_shared<SchemaOptions>();
  schema_options->primary_column = "pk_field";
  schema_options->version_column = "ts_field";
  schema_options->vector_column = "vec_field";

  auto schema_metadata =
      arrow::KeyValueMetadata(std::vector<std::string>{"key1", "key2"}, std::vector<std::string>{"value1", "value2"});
  auto metadata_status = schema_builder.AddMetadata(schema_metadata);
  ASSERT_TRUE(metadata_status.ok());

  auto arrow_schema = schema_builder.Finish().ValueOrDie();

  auto space_schema = std::make_shared<Schema>(arrow_schema, schema_options);
  auto sp_status = space_schema->Validate();
  ASSERT_TRUE(sp_status.ok());

  auto uri = "file:///tmp/";
  auto res = Space::Open(uri, Options{space_schema, -1});
  ASSERT_TRUE(res.ok());
  auto space = std::move(res.value());

  // Create RecordBatch
  arrow::Int64Builder pk_builder;
  pk_builder.Append(1);
  pk_builder.Append(2);
  pk_builder.Append(3);
  arrow::Int64Builder ts_builder;
  ts_builder.Append(1);
  ts_builder.Append(2);
  ts_builder.Append(3);
  arrow::FixedSizeBinaryBuilder vec_builder(arrow::fixed_size_binary(10));
  vec_builder.Append("1234567890");
  vec_builder.Append("1234567890");
  vec_builder.Append("1234567890");

  std::shared_ptr<arrow::Array> pk_array;
  pk_builder.Finish(&pk_array);
  std::shared_ptr<arrow::Array> ts_array;
  ts_builder.Finish(&ts_array);
  std::shared_ptr<arrow::Array> vec_array;
  vec_builder.Finish(&vec_array);

  auto rec_batch = arrow::RecordBatch::Make(arrow_schema, 3, {pk_array, ts_array, vec_array});
  auto reader = arrow::RecordBatchReader::Make({rec_batch}, arrow_schema).ValueOrDie();

  auto write_option = WriteOption{10};
  space->Write(reader.get(), &write_option);
}
/**
 * @brief Test Space::Read
 *  TODO: need to implement Next function
 */
TEST(SpaceTest, SpaceReadTest) {}

/**
 * @brief Test Space::Delete
 *  TODO: need to implement
 */
TEST(SpaceTest, SpaceDeleteTest) {}

}  // namespace milvus_storage
