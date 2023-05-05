#include "gtest/gtest.h"
#include "storage/manifest.h"
#include "gmock/gmock.h"
#include "google/protobuf/util/message_differencer.h"

using ::testing::ElementsAre;

namespace milvus_storage {
TEST(ManifestTest, ManifestGetSetTest) {
  std::shared_ptr<SpaceOptions> options = std::make_shared<SpaceOptions>();
  std::shared_ptr<Schema> schema = std::make_shared<Schema>();
  Manifest manifest(options, schema);
  manifest.add_scalar_files({"scalar_file1", "scalar_file2"});
  manifest.add_vector_files({"vector_file1", "vector_file2"});
  manifest.add_delete_file("delete_file");
  ASSERT_THAT(manifest.scalar_files(), ElementsAre("scalar_file1", "scalar_file2"));
  ASSERT_THAT(manifest.vector_files(), ElementsAre("vector_file1", "vector_file2"));
  ASSERT_THAT(manifest.delete_files(), ElementsAre("delete_file"));
  ASSERT_EQ(manifest.schema(), schema);
  ASSERT_EQ(manifest.space_options(), options);
}

TEST(ManifestTest, ManifestProtoTest) {
  std::shared_ptr<SpaceOptions> options = std::make_shared<SpaceOptions>();
  options->uri = "file:///tmp/test";
  std::shared_ptr<Schema> schema = std::make_shared<Schema>();
  Manifest manifest(options, schema);
  auto proto_manifest = manifest.ToProtobuf();
  manifest.add_scalar_files({"scalar_file"});
  manifest.add_vector_files({"vector_file"});
  manifest.add_delete_file("delete_file");

  manifest_proto::Manifest expected_proto_manifest;
  expected_proto_manifest.set_allocated_options(options->ToProtobuf().release());
  auto status = schema->ToProtobuf();
  ASSERT_TRUE(status.ok());
  expected_proto_manifest.set_allocated_schema(status.value().release());
  expected_proto_manifest.add_scalar_files("scalar_file");
  expected_proto_manifest.add_vector_files("vector_file");
  expected_proto_manifest.add_delete_files("delete_file");

  ASSERT_TRUE(google::protobuf::util::MessageDifferencer::Equals(expected_proto_manifest, proto_manifest.value()));
}
}  // namespace milvus_storage