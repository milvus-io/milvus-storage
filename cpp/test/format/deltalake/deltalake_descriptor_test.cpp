// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#include <limits>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "format/deltalake/deltalake_common.h"

namespace milvus_storage::deltalake {
namespace {

using Json = nlohmann::json;

api::ColumnGroupFile DescriptorFile(const Json& descriptor) {
  api::ColumnGroupFile file{"file:///table/part.parquet", 0, 10, {}};
  file.Set(api::kPropertyFileSize, uint64_t{128});
  file.Set(api::kPropertyMetadata, descriptor.dump());
  return file;
}

TEST(DeltaDescriptorTest, PreservesLegacyWireFormat) {
  for (const auto snapshot_version : {int64_t{0}, int64_t{42}, std::numeric_limits<int64_t>::max()}) {
    // Keep the literal wire schema independent of the codec's constants.
    const Json legacy{
        {"version", 1}, {"table_uri", "file:///table%252F%20space/"}, {"snapshot_version", snapshot_version}};
    const auto parsed = SnapshotDescriptor::Parse(DescriptorFile(legacy));
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    EXPECT_EQ(parsed->version, 1);
    EXPECT_EQ(parsed->table_uri, "file:///table%252F%20space/");
    EXPECT_EQ(parsed->snapshot_version, snapshot_version);
    EXPECT_EQ(parsed->Serialize(), legacy.dump());

    const SnapshotDescriptor created{.table_uri = parsed->table_uri, .snapshot_version = snapshot_version};
    EXPECT_EQ(created.Serialize(), legacy.dump());
  }
}

TEST(DeltaDescriptorTest, FileCacheKeySeparatesFilesAndSnapshots) {
  const Json valid{{"version", 1}, {"table_uri", "file:///table/"}, {"snapshot_version", 0}};
  const auto file = DescriptorFile(valid);
  const auto parsed = SnapshotDescriptor::Parse(file);
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  const auto key = parsed->FileCacheKey(file);
  auto changed = file;
  changed.path = "file:///table/other.parquet";
  EXPECT_NE(key, parsed->FileCacheKey(changed));
  changed = file;
  changed.Set(api::kPropertyFileSize, uint64_t{129});
  EXPECT_NE(key, parsed->FileCacheKey(changed));
  changed = file;
  ++changed.end_index;
  EXPECT_NE(key, parsed->FileCacheKey(changed));
  auto snapshot = *parsed;
  ++snapshot.snapshot_version;
  EXPECT_NE(key, snapshot.FileCacheKey(file));
  snapshot = *parsed;
  snapshot.table_uri = "file:///other/";
  EXPECT_NE(key, snapshot.FileCacheKey(file));

  changed = file;
  changed.Set(api::kPropertyMetadata,
              std::string(R"({"snapshot_version":0,"table_uri":"file:///table/","version":1})"));
  const auto reordered = SnapshotDescriptor::Parse(changed);
  ASSERT_TRUE(reordered.ok()) << reordered.status();
  EXPECT_EQ(key, reordered->FileCacheKey(changed));
}

TEST(DeltaDescriptorTest, RejectsMalformedAndUnsupportedMetadata) {
  const Json valid{{"version", 1}, {"table_uri", "file:///table/"}, {"snapshot_version", 0}};
  for (const auto& field : {"version", "table_uri", "snapshot_version"}) {
    auto missing = valid;
    missing.erase(field);
    EXPECT_TRUE(SnapshotDescriptor::Parse(DescriptorFile(missing)).status().IsInvalid()) << field;
  }
  for (const Json& version : {Json(-1), Json(0), Json(2), Json(1.0), Json("1"), Json(true), Json(nullptr)}) {
    auto invalid = valid;
    invalid["version"] = version;
    EXPECT_TRUE(SnapshotDescriptor::Parse(DescriptorFile(invalid)).status().IsInvalid()) << invalid;
  }
  for (const Json& version :
       {Json(-1), Json(std::numeric_limits<uint64_t>::max()), Json(0.0), Json("0"), Json(nullptr)}) {
    auto invalid = valid;
    invalid["snapshot_version"] = version;
    EXPECT_TRUE(SnapshotDescriptor::Parse(DescriptorFile(invalid)).status().IsInvalid()) << invalid;
  }
  for (const Json& table : {Json(""), Json("file:///table"), Json(1), Json(nullptr)}) {
    auto invalid = valid;
    invalid["table_uri"] = table;
    EXPECT_TRUE(SnapshotDescriptor::Parse(DescriptorFile(invalid)).status().IsInvalid()) << invalid;
  }
  for (const auto& invalid : {Json::array(), Json(nullptr)}) {
    EXPECT_TRUE(SnapshotDescriptor::Parse(DescriptorFile(invalid)).status().IsInvalid());
  }
  auto file = DescriptorFile(valid);
  file.Set(api::kPropertyMetadata, std::string("{"));
  EXPECT_TRUE(SnapshotDescriptor::Parse(file).status().IsInvalid());
  file.properties.erase(api::kPropertyMetadata);
  EXPECT_TRUE(SnapshotDescriptor::Parse(file).status().IsInvalid());
}

TEST(DeltaDescriptorTest, RequiresWholeFileRangeAndValidSize) {
  const Json valid{{"version", 1}, {"table_uri", "file:///table/"}, {"snapshot_version", 0}};
  auto file = DescriptorFile(valid);
  for (const auto start : {int64_t{-1}, int64_t{1}}) {
    file.start_index = start;
    EXPECT_TRUE(SnapshotDescriptor::Parse(file).status().IsInvalid());
  }
  file.start_index = 0;
  file.end_index = -1;
  EXPECT_TRUE(SnapshotDescriptor::Parse(file).status().IsInvalid());
  file.end_index = 0;
  EXPECT_TRUE(SnapshotDescriptor::Parse(file).ok());
  file.properties.erase(api::kPropertyFileSize);
  EXPECT_TRUE(SnapshotDescriptor::Parse(file).status().IsInvalid());
  file.Set(api::kPropertyFileSize, std::string("invalid"));
  EXPECT_TRUE(SnapshotDescriptor::Parse(file).status().IsInvalid());
}

}  // namespace
}  // namespace milvus_storage::deltalake
