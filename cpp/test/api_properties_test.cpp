

#include <gtest/gtest.h>

#include "milvus-storage/properties.h"
#include "milvus-storage/ffi_c.h"
#include <string>
#include <cstdint>
#include <optional>
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/buffer.h>

using namespace milvus_storage::api;
class APIPropertiesTest : public ::testing::Test {};

namespace milvus_storage::api::test {

TEST_F(APIPropertiesTest, basic) {
  milvus_storage::api::Properties pp{};

  // Test get default value
  EXPECT_EQ(GetValueNoError<std::string>(pp, PROPERTY_WRITER_COMPRESSION), "zstd");
  EXPECT_EQ(GetValueNoError<int32_t>(pp, PROPERTY_WRITER_COMPRESSION_LEVEL), 5);
  EXPECT_EQ(GetValueNoError<int32_t>(pp, PROPERTY_WRITER_BUFFER_SIZE), 32 * 1024 * 1024);
  EXPECT_TRUE(GetValueNoError<bool>(pp, PROPERTY_WRITER_ENABLE_DICTIONARY));

  // Test set & get properties
  EXPECT_EQ(SetValue(pp, PROPERTY_WRITER_COMPRESSION, "gzip"), std::nullopt);
  EXPECT_EQ(SetValue(pp, PROPERTY_WRITER_COMPRESSION_LEVEL, "3"), std::nullopt);
  EXPECT_EQ(SetValue(pp, PROPERTY_WRITER_BUFFER_SIZE, "67108864"), std::nullopt);
  EXPECT_EQ(SetValue(pp, PROPERTY_WRITER_ENABLE_DICTIONARY, "false"), std::nullopt);

  EXPECT_EQ(GetValueNoError<std::string>(pp, PROPERTY_WRITER_COMPRESSION), "gzip");
  EXPECT_EQ(GetValueNoError<int32_t>(pp, PROPERTY_WRITER_COMPRESSION_LEVEL), 3);
  EXPECT_EQ(GetValueNoError<int32_t>(pp, PROPERTY_WRITER_BUFFER_SIZE), 64 * 1024 * 1024);
  EXPECT_FALSE(GetValueNoError<bool>(pp, PROPERTY_WRITER_ENABLE_DICTIONARY));
}

TEST_F(APIPropertiesTest, get_invalid_key) {
  milvus_storage::api::Properties pp{};

  // Unknown keys
  EXPECT_FALSE(GetValue<int32_t>(pp, "nonexistent.key").ok());
  std::cout << GetValue<int32_t>(pp, "nonexistent.key").status().ToString() << std::endl;

  // set a new key without defined
  EXPECT_NE(SetValue(pp, "testkey1", "testval1", false), std::nullopt);
  std::cout << SetValue(pp, "testkey1", "testval1", false).value() << std::endl;

  // register a new key and get it as different type
  EXPECT_EQ(SetValue(pp, "testkey1", "testval1", true), std::nullopt);
  auto testkey1_result = GetValue<int32_t>(pp, "testkey1");  // invalid type
  EXPECT_FALSE(testkey1_result.ok());
  std::cout << testkey1_result.status().ToString() << std::endl;

  // get a predefined key as different type
  auto buffer_size_result = GetValue<std::string>(pp, PROPERTY_WRITER_BUFFER_SIZE);  // invalid type
  EXPECT_FALSE(buffer_size_result.ok());
  std::cout << buffer_size_result.status().ToString() << std::endl;
}

TEST_F(APIPropertiesTest, test_ffi_convert) {
  milvus_storage::api::Properties pp{};

  // Test FFI properties conversion
  {
    ::Properties ffi_props;
    ffi_props.count = 3;
    ::Property kvp[3];
    kvp[0].key = const_cast<char*>(PROPERTY_WRITER_COMPRESSION);
    kvp[0].value = const_cast<char*>("gzip");
    kvp[1].key = const_cast<char*>(PROPERTY_WRITER_COMPRESSION_LEVEL);
    kvp[1].value = const_cast<char*>("4");
    kvp[2].key = const_cast<char*>("custom.key");
    kvp[2].value = const_cast<char*>("custom_value");
    ffi_props.properties = kvp;

    EXPECT_EQ(FromFFIProperties(pp, &ffi_props), std::nullopt);

    EXPECT_EQ(GetValueNoError<std::string>(pp, PROPERTY_WRITER_COMPRESSION), "gzip");
    EXPECT_EQ(GetValueNoError<int32_t>(pp, PROPERTY_WRITER_COMPRESSION_LEVEL), 4);
    EXPECT_EQ(GetValueNoError<std::string>(pp, "custom.key"), "custom_value");
  }

  // Test FFI properties with invalid type
  {
    ::Properties ffi_props;
    ffi_props.count = 2;
    ::Property kvp[2];
    // first is valid
    kvp[0].key = const_cast<char*>(PROPERTY_WRITER_BUFFER_SIZE);
    kvp[0].value = const_cast<char*>("3");
    // second is invalid int
    kvp[1].key = const_cast<char*>(PROPERTY_WRITER_COMPRESSION_LEVEL);
    kvp[1].value = const_cast<char*>("invalid_int");
    ffi_props.properties = kvp;

    auto opt = FromFFIProperties(pp, &ffi_props);
    EXPECT_NE(opt, std::nullopt);
    std::cout << "Expected error: " << opt.value() << std::endl;

    // invalid bool
    kvp[1].key = const_cast<char*>(PROPERTY_FS_USE_SSL);
    kvp[1].value = const_cast<char*>("invalid_int");

    opt = FromFFIProperties(pp, &ffi_props);
    EXPECT_NE(opt, std::nullopt);
    std::cout << "Expected error: " << opt.value() << std::endl;

    // invalid int32
    kvp[1].key = const_cast<char*>(PROPERTY_WRITER_BUFFER_SIZE);
    kvp[1].value = const_cast<char*>("214748364700");  // overflow int32

    opt = FromFFIProperties(pp, &ffi_props);
    EXPECT_NE(opt, std::nullopt);
    std::cout << "Expected error: " << opt.value() << std::endl;

    // valid int32
    kvp[1].key = const_cast<char*>(PROPERTY_WRITER_BUFFER_SIZE);
    kvp[1].value = const_cast<char*>("2147483647");  // won't overflow int32

    opt = FromFFIProperties(pp, &ffi_props);
    EXPECT_EQ(opt, std::nullopt);
  }

  // Test FFI properties with invalid enum value
  {
    ::Properties ffi_props;
    ffi_props.count = 1;
    ::Property kvp[1];
    kvp[0].key = const_cast<char*>(PROPERTY_WRITER_COMPRESSION);
    kvp[0].value = const_cast<char*>("unknown_compression");
    ffi_props.properties = kvp;

    auto opt = FromFFIProperties(pp, &ffi_props);
    EXPECT_NE(opt, std::nullopt);
    std::cout << "Expected error: " << opt.value() << std::endl;
  }
}

}  // namespace milvus_storage::api::test