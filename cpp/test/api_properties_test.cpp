

#include <gtest/gtest.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/filesystem/fs.h"
#include "test_env.h"
#include <string>
#include <cstdint>
#include <optional>

using namespace milvus_storage::api;
class APIPropertiesTest : public ::testing::Test {};

namespace milvus_storage::test {

TEST_F(APIPropertiesTest, basic) {
  milvus_storage::api::Properties pp{};

  // Test get default value
  EXPECT_EQ(GetValueNoError<std::string>(pp, PROPERTY_WRITER_COMPRESSION), "zstd");
  EXPECT_EQ(GetValueNoError<std::string>(pp, PROPERTY_WRITER_FORMAT), LOON_FORMAT_PARQUET);
  EXPECT_EQ(GetValueNoError<int32_t>(pp, PROPERTY_WRITER_COMPRESSION_LEVEL), 5);
  EXPECT_EQ(GetValueNoError<int32_t>(pp, PROPERTY_WRITER_BUFFER_SIZE), 32 * 1024 * 1024);
  EXPECT_TRUE(GetValueNoError<bool>(pp, PROPERTY_WRITER_ENABLE_DICTIONARY));

  // Test set & get properties
  EXPECT_EQ(SetValue(pp, PROPERTY_WRITER_COMPRESSION, "gzip"), std::nullopt);
  EXPECT_EQ(SetValue(pp, PROPERTY_WRITER_FORMAT, LOON_FORMAT_VORTEX), std::nullopt);
  EXPECT_EQ(SetValue(pp, PROPERTY_WRITER_COMPRESSION_LEVEL, "3"), std::nullopt);
  EXPECT_EQ(SetValue(pp, PROPERTY_WRITER_BUFFER_SIZE, "67108864"), std::nullopt);
  EXPECT_EQ(SetValue(pp, PROPERTY_WRITER_ENABLE_DICTIONARY, "false"), std::nullopt);

  EXPECT_EQ(GetValueNoError<std::string>(pp, PROPERTY_WRITER_COMPRESSION), "gzip");
  EXPECT_EQ(GetValueNoError<std::string>(pp, PROPERTY_WRITER_FORMAT), LOON_FORMAT_VORTEX);
  EXPECT_EQ(GetValueNoError<int32_t>(pp, PROPERTY_WRITER_COMPRESSION_LEVEL), 3);
  EXPECT_EQ(GetValueNoError<int32_t>(pp, PROPERTY_WRITER_BUFFER_SIZE), 64 * 1024 * 1024);
  EXPECT_FALSE(GetValueNoError<bool>(pp, PROPERTY_WRITER_ENABLE_DICTIONARY));

  EXPECT_STREQ(loon_properties_writer_format, PROPERTY_WRITER_FORMAT);
}

TEST_F(APIPropertiesTest, SinglePolicyFallsBackToWriterFormat) {
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  milvus_storage::api::Properties pp{};
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_POLICY, LOON_COLUMN_GROUP_POLICY_SINGLE), std::nullopt);
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_FORMAT, LOON_FORMAT_VORTEX), std::nullopt);

  ASSERT_AND_ASSIGN(auto policy, ColumnGroupPolicy::create_column_group_policy(pp, schema));
  auto groups = policy->get_column_groups();

  ASSERT_EQ(groups.size(), 1);
  EXPECT_EQ(groups[0]->format, LOON_FORMAT_VORTEX);
}

TEST_F(APIPropertiesTest, SchemaBasedFormatsOverrideWriterFormatAndUnmatchedFallsBack) {
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  milvus_storage::api::Properties pp{};
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_POLICY, LOON_COLUMN_GROUP_POLICY_SCHEMA_BASED), std::nullopt);
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_FORMAT, LOON_FORMAT_PARQUET), std::nullopt);
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_SCHEMA_BASE_PATTERNS, "id|value,vector"), std::nullopt);
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_SCHEMA_BASE_FORMATS, "vortex,parquet"), std::nullopt);

  ASSERT_AND_ASSIGN(auto policy, ColumnGroupPolicy::create_column_group_policy(pp, schema));
  auto groups = policy->get_column_groups();

  ASSERT_EQ(groups.size(), 3);
  EXPECT_EQ(groups[0]->columns, (std::vector<std::string>{"id", "value"}));
  EXPECT_EQ(groups[0]->format, LOON_FORMAT_VORTEX);
  EXPECT_EQ(groups[1]->columns, (std::vector<std::string>{"vector"}));
  EXPECT_EQ(groups[1]->format, LOON_FORMAT_PARQUET);
  EXPECT_EQ(groups[2]->columns, (std::vector<std::string>{"name"}));
  EXPECT_EQ(groups[2]->format, LOON_FORMAT_PARQUET);
}

TEST_F(APIPropertiesTest, SchemaBasedFallsBackToWriterFormatWhenFormatsEmpty) {
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  milvus_storage::api::Properties pp{};
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_POLICY, LOON_COLUMN_GROUP_POLICY_SCHEMA_BASED), std::nullopt);
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_FORMAT, LOON_FORMAT_VORTEX), std::nullopt);
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_SCHEMA_BASE_PATTERNS, "id|value,vector"), std::nullopt);

  ASSERT_AND_ASSIGN(auto policy, ColumnGroupPolicy::create_column_group_policy(pp, schema));
  auto groups = policy->get_column_groups();

  ASSERT_EQ(groups.size(), 3);
  for (const auto& group : groups) {
    EXPECT_EQ(group->format, LOON_FORMAT_VORTEX);
  }
}

TEST_F(APIPropertiesTest, SchemaBasedFormatsLengthMismatchFails) {
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  milvus_storage::api::Properties pp{};
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_POLICY, LOON_COLUMN_GROUP_POLICY_SCHEMA_BASED), std::nullopt);
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_FORMAT, LOON_FORMAT_PARQUET), std::nullopt);
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_SCHEMA_BASE_PATTERNS, "id|value,vector"), std::nullopt);
  ASSERT_EQ(SetValue(pp, PROPERTY_WRITER_SCHEMA_BASE_FORMATS, "vortex"), std::nullopt);

  auto policy = ColumnGroupPolicy::create_column_group_policy(pp, schema);
  ASSERT_STATUS_NOT_OK(policy.status());
}

TEST_F(APIPropertiesTest, SchemaBasedInvalidFormatFailsValidation) {
  milvus_storage::api::Properties pp{};
  auto err = SetValue(pp, PROPERTY_WRITER_SCHEMA_BASE_FORMATS, "vortex,unsupported-format");
  ASSERT_NE(err, std::nullopt);
  EXPECT_NE(err->find("unsupported-format"), std::string::npos) << *err;

  ::LoonProperty kvp[] = {
      {const_cast<char*>(PROPERTY_WRITER_SCHEMA_BASE_FORMATS), const_cast<char*>("unsupported-format")}};
  ::LoonProperties ffi_props{kvp, 1};
  err = ConvertFFIProperties(pp, &ffi_props);
  ASSERT_NE(err, std::nullopt);
  EXPECT_NE(err->find("unsupported-format"), std::string::npos) << *err;

  err = SetValue(pp, PROPERTY_WRITER_SCHEMA_BASE_FORMATS, "vortex,");
  ASSERT_NE(err, std::nullopt);
  EXPECT_NE(err->find("not in allowed set"), std::string::npos) << *err;
}

TEST_F(APIPropertiesTest, parquet_reader_prebuffer_properties) {
  milvus_storage::api::Properties pp{};

  EXPECT_EQ(GetValueNoError<int64_t>(pp, PROPERTY_READER_PARQUET_PREBUFFER_HOLE_SIZE_LIMIT), 0);
  EXPECT_EQ(GetValueNoError<int64_t>(pp, PROPERTY_READER_PARQUET_PREBUFFER_RANGE_SIZE_LIMIT), 0);

  EXPECT_EQ(SetValue(pp, PROPERTY_READER_PARQUET_PREBUFFER_HOLE_SIZE_LIMIT, "16384"), std::nullopt);
  EXPECT_EQ(SetValue(pp, PROPERTY_READER_PARQUET_PREBUFFER_RANGE_SIZE_LIMIT, "67108864"), std::nullopt);

  EXPECT_EQ(GetValueNoError<int64_t>(pp, PROPERTY_READER_PARQUET_PREBUFFER_HOLE_SIZE_LIMIT), 16384);
  EXPECT_EQ(GetValueNoError<int64_t>(pp, PROPERTY_READER_PARQUET_PREBUFFER_RANGE_SIZE_LIMIT), 64LL * 1024 * 1024);

  EXPECT_NE(SetValue(pp, PROPERTY_READER_PARQUET_PREBUFFER_HOLE_SIZE_LIMIT, "-1"), std::nullopt);
  EXPECT_EQ(SetValue(pp, PROPERTY_READER_PARQUET_PREBUFFER_RANGE_SIZE_LIMIT, "0"), std::nullopt);

  EXPECT_STREQ(loon_properties_reader_parquet_prebuffer_hole_size_limit,
               PROPERTY_READER_PARQUET_PREBUFFER_HOLE_SIZE_LIMIT);
  EXPECT_STREQ(loon_properties_reader_parquet_prebuffer_range_size_limit,
               PROPERTY_READER_PARQUET_PREBUFFER_RANGE_SIZE_LIMIT);
}

TEST_F(APIPropertiesTest, get_invalid_key) {
  milvus_storage::api::Properties pp{};

  // Unknown keys
  EXPECT_FALSE(GetValue<int32_t>(pp, "nonexistent.key").ok());

  // set a new key without defined
  EXPECT_NE(SetValue(pp, "testkey1", "testval1", false), std::nullopt);

  // register a new key and get it as different type
  EXPECT_EQ(SetValue(pp, "testkey1", "testval1", true), std::nullopt);
  auto testkey1_result = GetValue<int32_t>(pp, "testkey1");  // invalid type
  EXPECT_FALSE(testkey1_result.ok());

  // get a predefined key as different type
  auto buffer_size_result = GetValue<std::string>(pp, PROPERTY_WRITER_BUFFER_SIZE);  // invalid type
  EXPECT_FALSE(buffer_size_result.ok());
}

TEST_F(APIPropertiesTest, test_ffi_convert) {
  milvus_storage::api::Properties pp{};

  // Test FFI properties conversion
  {
    ::LoonProperties ffi_props;
    ffi_props.count = 3;
    ::LoonProperty kvp[3];
    kvp[0].key = const_cast<char*>(PROPERTY_WRITER_COMPRESSION);
    kvp[0].value = const_cast<char*>("gzip");
    kvp[1].key = const_cast<char*>(PROPERTY_WRITER_COMPRESSION_LEVEL);
    kvp[1].value = const_cast<char*>("4");
    kvp[2].key = const_cast<char*>("custom.key");
    kvp[2].value = const_cast<char*>("custom_value");
    ffi_props.properties = kvp;

    EXPECT_EQ(ConvertFFIProperties(pp, &ffi_props), std::nullopt);

    EXPECT_EQ(GetValueNoError<std::string>(pp, PROPERTY_WRITER_COMPRESSION), "gzip");
    EXPECT_EQ(GetValueNoError<int32_t>(pp, PROPERTY_WRITER_COMPRESSION_LEVEL), 4);
    EXPECT_EQ(GetValueNoError<std::string>(pp, "custom.key"), "custom_value");
  }

  // Test FFI properties with invalid type
  {
    ::LoonProperties ffi_props;
    ffi_props.count = 2;
    ::LoonProperty kvp[2];
    // first is valid
    kvp[0].key = const_cast<char*>(PROPERTY_WRITER_BUFFER_SIZE);
    kvp[0].value = const_cast<char*>("3");
    // second is invalid int
    kvp[1].key = const_cast<char*>(PROPERTY_WRITER_COMPRESSION_LEVEL);
    kvp[1].value = const_cast<char*>("invalid_int");
    ffi_props.properties = kvp;

    auto opt = ConvertFFIProperties(pp, &ffi_props);
    EXPECT_NE(opt, std::nullopt);

    // invalid bool
    kvp[1].key = const_cast<char*>(PROPERTY_FS_USE_SSL);
    kvp[1].value = const_cast<char*>("invalid_int");

    opt = ConvertFFIProperties(pp, &ffi_props);
    EXPECT_NE(opt, std::nullopt);

    // invalid int32
    kvp[1].key = const_cast<char*>(PROPERTY_WRITER_BUFFER_SIZE);
    kvp[1].value = const_cast<char*>("214748364700");  // overflow int32

    opt = ConvertFFIProperties(pp, &ffi_props);
    EXPECT_NE(opt, std::nullopt);

    // valid int32
    kvp[1].key = const_cast<char*>(PROPERTY_WRITER_BUFFER_SIZE);
    kvp[1].value = const_cast<char*>("2147483647");  // won't overflow int32

    opt = ConvertFFIProperties(pp, &ffi_props);
    EXPECT_EQ(opt, std::nullopt);
  }

  // Test FFI properties with invalid enum value
  {
    ::LoonProperties ffi_props;
    ffi_props.count = 1;
    ::LoonProperty kvp[1];
    kvp[0].key = const_cast<char*>(PROPERTY_WRITER_COMPRESSION);
    kvp[0].value = const_cast<char*>("unknown_compression");
    ffi_props.properties = kvp;

    auto opt = ConvertFFIProperties(pp, &ffi_props);
    EXPECT_NE(opt, std::nullopt);
  }
}

TEST_F(APIPropertiesTest, invalid_cloud_provider_error) {
  const char* invalid_provider = "unknown-cloud-provider";
  milvus_storage::api::Properties pp{};
  milvus_storage::api::Properties converted{};
  ::LoonProperty kvp[] = {{const_cast<char*>(loon_properties_fs_cloud_provider), const_cast<char*>(invalid_provider)}};
  ::LoonProperties ffi_props{kvp, 1};

  for (const auto& err :
       {SetValue(pp, PROPERTY_FS_CLOUD_PROVIDER, invalid_provider), ConvertFFIProperties(converted, &ffi_props)}) {
    ASSERT_NE(err, std::nullopt);
    for (const auto* expected :
         {"value 'unknown-cloud-provider'", "not in allowed set", kCloudProviderAWS, kCloudProviderGCP,
          kCloudProviderAliyun, kCloudProviderAzure, kCloudProviderTencent, kCloudProviderHuawei}) {
      EXPECT_NE(err->find(expected), std::string::npos) << *err;
    }
  }
}

}  // namespace milvus_storage::test
