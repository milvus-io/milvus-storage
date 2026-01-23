// Copyright 2024 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <cstring>

#include "milvus-storage/text_column/lob_reference.h"

using namespace milvus_storage::text_column;

class LOBReferenceTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // create a sample UUID string (36 chars: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    test_file_id_str_ = "12345678-9abc-def0-1122-334455667788";
    // create corresponding binary UUID (16 bytes)
    // UUID string "12345678-9abc-def0-1122-334455667788" -> binary
    test_file_id_[0] = 0x12;
    test_file_id_[1] = 0x34;
    test_file_id_[2] = 0x56;
    test_file_id_[3] = 0x78;
    test_file_id_[4] = 0x9a;
    test_file_id_[5] = 0xbc;
    test_file_id_[6] = 0xde;
    test_file_id_[7] = 0xf0;
    test_file_id_[8] = 0x11;
    test_file_id_[9] = 0x22;
    test_file_id_[10] = 0x33;
    test_file_id_[11] = 0x44;
    test_file_id_[12] = 0x55;
    test_file_id_[13] = 0x66;
    test_file_id_[14] = 0x77;
    test_file_id_[15] = 0x88;
  }

  std::string test_file_id_str_;
  uint8_t test_file_id_[UUID_BINARY_SIZE];
};

// test EncodeInlineText - variable length encoding
TEST_F(LOBReferenceTest, EncodeInlineText) {
  std::string text = "hello";
  auto encoded = EncodeInlineText(text);

  // inline text: flag (1 byte) + text bytes
  ASSERT_EQ(encoded.size(), 1 + text.size());
  ASSERT_EQ(encoded[0], FLAG_INLINE_DATA);  // inline flag
  // check text data starts immediately after flag
  ASSERT_EQ(std::string(reinterpret_cast<const char*>(&encoded[1]), text.size()), text);
}

// test EncodeInlineText with empty string
TEST_F(LOBReferenceTest, EncodeInlineTextEmpty) {
  std::string text = "";
  auto encoded = EncodeInlineText(text);

  // empty inline text: just flag byte
  ASSERT_EQ(encoded.size(), 1);
  ASSERT_EQ(encoded[0], FLAG_INLINE_DATA);  // inline flag
}

// test EncodeInlineText with longer text
TEST_F(LOBReferenceTest, EncodeInlineTextLong) {
  std::string text = "this is a longer text that would previously be truncated";
  auto encoded = EncodeInlineText(text);

  // inline text stores the complete text now (variable length)
  ASSERT_EQ(encoded.size(), 1 + text.size());
  ASSERT_EQ(encoded[0], FLAG_INLINE_DATA);  // inline flag
  // check complete text is stored
  ASSERT_EQ(std::string(reinterpret_cast<const char*>(&encoded[1]), text.size()), text);
}

// test EncodeLOBReference - fixed 24 bytes with padding
TEST_F(LOBReferenceTest, EncodeLOBReference) {
  int32_t row_offset = 12345;
  auto encoded = EncodeLOBReference(test_file_id_, row_offset);

  // LOB reference is fixed 24 bytes: flag (1) + padding (3) + file_id (16) + row_offset (4)
  ASSERT_EQ(encoded.size(), LOB_REFERENCE_SIZE);
  ASSERT_EQ(encoded.size(), 24);
  ASSERT_EQ(encoded[0], FLAG_LOB_REFERENCE);  // LOB flag

  // check padding bytes are zero
  ASSERT_EQ(encoded[1], 0x00);
  ASSERT_EQ(encoded[2], 0x00);
  ASSERT_EQ(encoded[3], 0x00);

  // check file_id binary (starts at offset 4 after padding, 16 bytes)
  ASSERT_EQ(std::memcmp(&encoded[4], test_file_id_, UUID_BINARY_SIZE), 0);

  // check row_offset (starts at offset 20 = 4 + 16)
  int32_t decoded_offset;
  std::memcpy(&decoded_offset, &encoded[20], sizeof(int32_t));
  ASSERT_EQ(decoded_offset, row_offset);
}

// test DecodeInlineText - needs size parameter now
TEST_F(LOBReferenceTest, DecodeInlineText) {
  std::string text = "hello world";
  auto encoded = EncodeInlineText(text);
  auto decoded = DecodeInlineText(encoded.data(), encoded.size());

  ASSERT_EQ(decoded, text);
}

// test DecodeInlineText with empty string
TEST_F(LOBReferenceTest, DecodeInlineTextEmpty) {
  std::string text = "";
  auto encoded = EncodeInlineText(text);
  auto decoded = DecodeInlineText(encoded.data(), encoded.size());

  ASSERT_EQ(decoded, text);
}

// test DecodeInlineText with long text
TEST_F(LOBReferenceTest, DecodeInlineTextLong) {
  std::string text = "this is a much longer text that is now stored completely";
  auto encoded = EncodeInlineText(text);
  auto decoded = DecodeInlineText(encoded.data(), encoded.size());

  ASSERT_EQ(decoded, text);
}

// test DecodeLOBReference
TEST_F(LOBReferenceTest, DecodeLOBReference) {
  int32_t row_offset = 54321;
  auto encoded = EncodeLOBReference(test_file_id_, row_offset);

  std::string decoded_file_id_str;
  int32_t decoded_offset;
  DecodeLOBReference(encoded.data(), &decoded_file_id_str, &decoded_offset);

  // check file_id_str (should be converted from binary to string)
  ASSERT_EQ(decoded_file_id_str, test_file_id_str_);

  // check row_offset
  ASSERT_EQ(decoded_offset, row_offset);
}

// test IsLOBReference
TEST_F(LOBReferenceTest, IsLOBReference) {
  // test with LOB reference
  int32_t row_offset = 100;
  auto lob_encoded = EncodeLOBReference(test_file_id_, row_offset);
  ASSERT_TRUE(IsLOBReference(lob_encoded.data()));

  // test with inline text
  std::string text = "test";
  auto inline_encoded = EncodeInlineText(text);
  ASSERT_FALSE(IsLOBReference(inline_encoded.data()));
}

// test IsInlineData
TEST_F(LOBReferenceTest, IsInlineData) {
  // test with inline text
  std::string text = "test";
  auto inline_encoded = EncodeInlineText(text);
  ASSERT_TRUE(IsInlineData(inline_encoded.data()));

  // test with LOB reference
  int32_t row_offset = 100;
  auto lob_encoded = EncodeLOBReference(test_file_id_, row_offset);
  ASSERT_FALSE(IsInlineData(lob_encoded.data()));
}

// test BuildLOBFilePath - takes lob_base_path and file_id_str directly
TEST_F(LOBReferenceTest, BuildLOBFilePath) {
  std::string lob_base_path = "/data/collection/partition/lobs/100";

  std::string file_path = BuildLOBFilePath(lob_base_path, test_file_id_str_);

  // should contain lob_base_path, _data, and UUID with .vx extension
  ASSERT_NE(file_path.find(lob_base_path), std::string::npos);
  ASSERT_NE(file_path.find("_data"), std::string::npos);
  ASSERT_NE(file_path.find(".vx"), std::string::npos);
  ASSERT_NE(file_path.find("12345678-9abc-def0-1122-334455667788"), std::string::npos);
}

// test BuildLOBFilePath with trailing slash
TEST_F(LOBReferenceTest, BuildLOBFilePathTrailingSlash) {
  std::string lob_base_path = "/data/collection/partition/lobs/200/";

  std::string file_path = BuildLOBFilePath(lob_base_path, test_file_id_str_);

  // should not have double slashes
  ASSERT_EQ(file_path.find("//"), std::string::npos);
  ASSERT_NE(file_path.find("_data"), std::string::npos);
}

// test GenerateUUIDBinary and UUIDToString
TEST_F(LOBReferenceTest, GenerateUUIDBinaryAndToString) {
  uint8_t uuid1[UUID_BINARY_SIZE];
  uint8_t uuid2[UUID_BINARY_SIZE];
  GenerateUUIDBinary(uuid1);
  GenerateUUIDBinary(uuid2);

  // two generated UUIDs should be different
  ASSERT_NE(std::memcmp(uuid1, uuid2, UUID_BINARY_SIZE), 0) << "two generated UUIDs should be different";

  // convert to string and check format
  std::string uuid1_str = UUIDToString(uuid1);
  std::string uuid2_str = UUIDToString(uuid2);

  // check UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (36 chars)
  ASSERT_EQ(uuid1_str.size(), 36) << "UUID string should be 36 characters";
  ASSERT_EQ(uuid2_str.size(), 36) << "UUID string should be 36 characters";

  // check hyphens are in correct positions (8, 13, 18, 23)
  ASSERT_EQ(uuid1_str[8], '-');
  ASSERT_EQ(uuid1_str[13], '-');
  ASSERT_EQ(uuid1_str[18], '-');
  ASSERT_EQ(uuid1_str[23], '-');

  // check UUID v4 format: version digit at position 14 should be '4'
  ASSERT_EQ(uuid1_str[14], '4') << "UUID version should be 4";

  // check variant at position 19 should be 8, 9, a, or b
  char variant = uuid1_str[19];
  ASSERT_TRUE(variant == '8' || variant == '9' || variant == 'a' || variant == 'b')
      << "UUID variant should be 8, 9, a, or b";
}

// test UUIDToString with known binary UUID
TEST_F(LOBReferenceTest, UUIDToStringKnown) {
  std::string uuid_str = UUIDToString(test_file_id_);
  ASSERT_EQ(uuid_str, test_file_id_str_);
}

// test round-trip: encode then decode inline text
TEST_F(LOBReferenceTest, RoundTripInlineText) {
  std::string original = "hello";
  auto encoded = EncodeInlineText(original);
  auto decoded = DecodeInlineText(encoded.data(), encoded.size());

  ASSERT_EQ(decoded, original);
}

// test round-trip with various text sizes
TEST_F(LOBReferenceTest, RoundTripInlineTextVariousSizes) {
  std::vector<std::string> test_texts = {
      "",                              // empty
      "a",                             // 1 char
      "ab",                            // 2 chars
      "hello world",                   // 11 chars
      "this is a longer test string",  // longer
      std::string(100, 'x'),           // 100 chars
      std::string(1000, 'y'),          // 1000 chars
  };

  for (const auto& original : test_texts) {
    auto encoded = EncodeInlineText(original);
    ASSERT_EQ(encoded.size(), 1 + original.size()) << "size mismatch for text length " << original.size();

    auto decoded = DecodeInlineText(encoded.data(), encoded.size());
    ASSERT_EQ(decoded, original) << "round-trip failed for text length " << original.size();
  }
}

// test round-trip: encode then decode LOB reference
TEST_F(LOBReferenceTest, RoundTripLOBReference) {
  int32_t original_offset = 99999;
  auto encoded = EncodeLOBReference(test_file_id_, original_offset);

  std::string decoded_file_id_str;
  int32_t decoded_offset;
  DecodeLOBReference(encoded.data(), &decoded_file_id_str, &decoded_offset);

  ASSERT_EQ(decoded_file_id_str, test_file_id_str_);
  ASSERT_EQ(decoded_offset, original_offset);
}

// test LOB_REFERENCE_SIZE constant
TEST_F(LOBReferenceTest, LOBReferenceSizeConstant) {
  // LOB reference: flag (1) + padding (3) + file_id (16) + row_offset (4) = 24 bytes
  ASSERT_EQ(LOB_REFERENCE_SIZE, 24);

  // verify by encoding
  auto encoded = EncodeLOBReference(test_file_id_, 0);
  ASSERT_EQ(encoded.size(), LOB_REFERENCE_SIZE);
}

// test UUID_BINARY_SIZE constant
TEST_F(LOBReferenceTest, UUIDBinarySizeConstant) {
  // UUID binary: 16 bytes
  ASSERT_EQ(UUID_BINARY_SIZE, 16);
}

// test UUID_STRING_SIZE constant
TEST_F(LOBReferenceTest, UUIDStringSizeConstant) {
  // UUID string: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (32 hex + 4 hyphens = 36)
  ASSERT_EQ(UUID_STRING_SIZE, 36);
}
