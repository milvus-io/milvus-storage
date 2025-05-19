

#include "gtest/gtest.h"
#include "milvus-storage/common/metadata.h"

namespace milvus_storage {

class MetadataTest : public testing::Test {};

TEST_F(MetadataTest, TestGroupFieldIDListSerde) {
  GroupFieldIDList list({{0, 1, 2}, {3, 4}, {5, 6, 7, 8}});
  std::string serialized = list.Serialize();
  EXPECT_EQ(serialized, "0,1,2;3,4;5,6,7,8");
  GroupFieldIDList deserialized = GroupFieldIDList::Deserialize(serialized);
  EXPECT_EQ(deserialized, list);

  // Test case: Empty input
  GroupFieldIDList empty_list = {};
  serialized = empty_list.Serialize();
  EXPECT_EQ(serialized, "");
  deserialized = GroupFieldIDList::Deserialize(serialized);
  EXPECT_TRUE(deserialized.empty());

  // Test case: Single group
  GroupFieldIDList single_group({{1, 2, 3}});
  serialized = single_group.Serialize();
  EXPECT_EQ(serialized, "1,2,3");
  deserialized = GroupFieldIDList::Deserialize(serialized);
  EXPECT_EQ(deserialized, single_group);

  // Test case: Single column in each group
  GroupFieldIDList single_column_groups({{0}, {1}, {2}});
  serialized = single_column_groups.Serialize();
  EXPECT_EQ(serialized, "0;1;2");
  deserialized = GroupFieldIDList::Deserialize(serialized);
  EXPECT_EQ(deserialized, single_column_groups);
}
}  // namespace milvus_storage