#include <arrow/type.h>
#include <arrow/array/builder_primitive.h>
#include <numeric>
#include "gtest/gtest.h"
#include "storage/manifest.h"
#include "gmock/gmock.h"
#include "google/protobuf/util/message_differencer.h"
#include <arrow/util/key_value_metadata.h>

using ::testing::ElementsAre;

namespace milvus_storage {
TEST(ManifestTest, ManifestGetSetTest) {
  std::shared_ptr<Options> options = std::make_shared<Options>();
  std::shared_ptr<Schema> schema = std::make_shared<Schema>();
  Manifest manifest(options, schema);

  Fragment fragment1(1);
  fragment1.add_file("scalar_file1");
  fragment1.add_file("scalar_file2");
  manifest.add_scalar_fragment(std::move(fragment1));

  Fragment fragment2(2);
  fragment2.add_file("vector_file1");
  fragment2.add_file("vector_file2");

  manifest.add_vector_fragment(std::move(fragment2));

  Fragment fragment3(3);
  fragment3.add_file("delete_file");
  manifest.add_delete_fragment(std::move(fragment3));

  ASSERT_THAT(manifest.scalar_fragments(), ElementsAre(fragment1));
  ASSERT_THAT(manifest.vector_fragments(), ElementsAre(fragment2));
  ASSERT_THAT(manifest.delete_fragments(), ElementsAre(fragment3));
  ASSERT_EQ(manifest.schema(), schema);
  ASSERT_EQ(manifest.space_options(), options);
}

TEST(ManifestTest, ManifestProtoTest) {
  // Create Fields
  std::shared_ptr<arrow::KeyValueMetadata> metadata = arrow::KeyValueMetadata::Make(
      std::vector<std::string>{"key1", "key2"}, std::vector<std::string>{"value1", "value2"});

  std::shared_ptr<arrow::Field> pk_field = arrow::field("pk_field", arrow::int64(), /*nullable=*/false, metadata);

  std::shared_ptr<arrow::Field> ts_field = arrow::field("ts_field", arrow::int64(), /*nullable=*/false, metadata);

  std::shared_ptr<arrow::Field> vec_field =
      arrow::field("vec_field", arrow::fixed_size_binary(10), /*nullable=*/false, metadata);

  // Create Arrow Schema
  arrow::SchemaBuilder schema_builder;
  auto status = schema_builder.AddField(pk_field);
  ASSERT_TRUE(status.ok());
  status = schema_builder.AddField(ts_field);
  ASSERT_TRUE(status.ok());
  status = schema_builder.AddField(vec_field);
  ASSERT_TRUE(status.ok());
  auto schema_metadata =
      arrow::KeyValueMetadata(std::vector<std::string>{"key1", "key2"}, std::vector<std::string>{"value1", "value2"});
  auto metadata_status = schema_builder.AddMetadata(schema_metadata);
  ASSERT_TRUE(metadata_status.ok());
  auto arrow_schema = schema_builder.Finish().ValueOrDie();

  // Create Options
  auto options = std::make_shared<Options>();
  options->uri = "file:///tmp/test";
  auto schema_options = std::make_shared<SchemaOptions>();
  schema_options->primary_column = "pk_field";
  schema_options->version_column = "ts_field";
  schema_options->vector_column = "vec_field";

  // Create Schema
  auto space_schema1 = std::make_shared<Schema>(arrow_schema, schema_options);
  auto space_schema2 = std::make_shared<Schema>(arrow_schema, schema_options);

  auto sp_status = space_schema1->Validate();
  ASSERT_TRUE(sp_status.ok());

  // Create Manifest
  Manifest manifest1(options, space_schema1);
  // Add Fragments
  Fragment fragment1(1);
  fragment1.add_file("test_file1");
  fragment1.add_file("test_file2");
  manifest1.add_scalar_fragment(std::move(fragment1));
  manifest1.add_vector_fragment(std::move(fragment1));
  manifest1.add_delete_fragment(std::move(fragment1));
  // Serialize Manifest
  auto proto_manifest = manifest1.ToProtobuf();
  ASSERT_TRUE(proto_manifest.ok());

  // Create Manifest
  Manifest manifest2(options, space_schema2);
  // Deserialize Manifest
  manifest2.FromProtobuf(proto_manifest.value());

  // Compare Manifests
  ASSERT_EQ(manifest2.delete_fragments().size(), manifest1.delete_fragments().size());
  ASSERT_EQ(manifest2.scalar_fragments().size(), manifest1.scalar_fragments().size());
  ASSERT_EQ(manifest2.vector_fragments().size(), manifest1.vector_fragments().size());
}
}  // namespace milvus_storage
