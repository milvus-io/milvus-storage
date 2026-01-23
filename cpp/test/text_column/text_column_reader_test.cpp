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

#ifdef BUILD_VORTEX_BRIDGE

#include <gtest/gtest.h>
#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/filesystem/localfs.h>
#include <boost/filesystem.hpp>

#include <random>
#include <string>
#include <vector>

#include "test_env.h"
#include "milvus-storage/text_column/text_column_manager.h"
#include "milvus-storage/text_column/text_column_writer.h"
#include "milvus-storage/text_column/text_column_reader.h"
#include "milvus-storage/text_column/lob_reference.h"

namespace milvus_storage::text_column {

class TextColumnReaderTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // create test directory
    test_dir_ = "/tmp/text_column_reader_test_" + std::to_string(std::random_device{}());
    boost::filesystem::create_directories(test_dir_);

    // create filesystem
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();

    // create config with small thresholds for testing
    config_.lob_base_path = test_dir_ + "/lobs/100";
    config_.field_id = 100;
    config_.inline_threshold = 20;              // small threshold for testing inline vs LOB
    config_.max_lob_file_bytes = 1024 * 1024;   // 1MB for testing
    config_.flush_threshold_bytes = 64 * 1024;  // 64KB for testing

    ASSERT_STATUS_OK(InitTestProperties(config_.properties));
  }

  void TearDown() override {
    // cleanup test directory
    boost::filesystem::remove_all(test_dir_);
  }

  std::string GenerateRandomString(size_t length) {
    static const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, sizeof(charset) - 2);

    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; i++) {
      result.push_back(charset[dis(gen)]);
    }
    return result;
  }

  // helper to write test data and return references
  std::vector<std::vector<uint8_t>> WriteTestData(const std::vector<std::string>& texts) {
    auto manager_result = TextColumnManager::Create(fs_, config_);
    EXPECT_TRUE(manager_result.ok());
    auto manager = std::move(manager_result).ValueOrDie();

    auto writer_result = manager->CreateWriter();
    EXPECT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    auto refs_result = writer->WriteBatch(texts);
    EXPECT_TRUE(refs_result.ok());

    auto close_result = writer->Close();
    EXPECT_TRUE(close_result.ok());

    return std::move(refs_result).ValueOrDie();
  }

  protected:
  std::string test_dir_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  TextColumnConfig config_;
};

// ==================== ReadText Tests ====================

// test reading single inline text
TEST_F(TextColumnReaderTest, ReadSingleInlineText) {
  std::string text = "short";
  auto refs = WriteTestData({text});
  ASSERT_EQ(refs.size(), 1);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadText(refs[0].data(), refs[0].size());
  ASSERT_TRUE(read_result.ok()) << read_result.status().message();
  ASSERT_EQ(read_result.ValueOrDie(), text);

  ASSERT_STATUS_OK(reader->Close());
}

// test reading single LOB text
TEST_F(TextColumnReaderTest, ReadSingleLOBText) {
  std::string text = GenerateRandomString(100);  // larger than inline threshold
  auto refs = WriteTestData({text});
  ASSERT_EQ(refs.size(), 1);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadText(refs[0].data(), refs[0].size());
  ASSERT_TRUE(read_result.ok()) << read_result.status().message();
  ASSERT_EQ(read_result.ValueOrDie(), text);

  ASSERT_STATUS_OK(reader->Close());
}

// test reading with invalid reference (null pointer)
TEST_F(TextColumnReaderTest, ReadTextInvalidNullPointer) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadText(nullptr, 0);
  ASSERT_FALSE(read_result.ok());

  ASSERT_STATUS_OK(reader->Close());
}

// test reading with invalid LOB reference size
TEST_F(TextColumnReaderTest, ReadTextInvalidLOBReferenceSize) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // create a fake LOB reference with wrong size
  std::vector<uint8_t> fake_ref(20, 0);
  fake_ref[0] = FLAG_LOB_REFERENCE;  // set LOB flag

  auto read_result = reader->ReadText(fake_ref.data(), fake_ref.size());
  ASSERT_FALSE(read_result.ok());  // should fail due to wrong size

  ASSERT_STATUS_OK(reader->Close());
}

// test reading after close
TEST_F(TextColumnReaderTest, ReadTextAfterClose) {
  std::string text = "test";
  auto refs = WriteTestData({text});

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  ASSERT_STATUS_OK(reader->Close());
  ASSERT_TRUE(reader->IsClosed());

  auto read_result = reader->ReadText(refs[0].data(), refs[0].size());
  ASSERT_FALSE(read_result.ok());  // should fail - reader is closed
}

// ==================== ReadBatch Tests ====================

// test batch read with mixed inline and LOB
TEST_F(TextColumnReaderTest, ReadBatchMixed) {
  std::vector<std::string> texts = {
      "short1",                   // inline
      GenerateRandomString(50),   // LOB
      "short2",                   // inline
      GenerateRandomString(100),  // LOB
      "short3",                   // inline
  };

  auto refs = WriteTestData(texts);
  ASSERT_EQ(refs.size(), 5);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // prepare encoded refs
  std::vector<EncodedRef> encoded_refs;
  for (const auto& ref : refs) {
    encoded_refs.push_back({ref.data(), ref.size()});
  }

  auto read_result = reader->ReadBatch(encoded_refs);
  ASSERT_TRUE(read_result.ok()) << read_result.status().message();
  auto read_texts = std::move(read_result).ValueOrDie();

  ASSERT_EQ(read_texts.size(), texts.size());
  for (size_t i = 0; i < texts.size(); i++) {
    ASSERT_EQ(read_texts[i], texts[i]) << "mismatch at index " << i;
  }

  ASSERT_STATUS_OK(reader->Close());
}

// test batch read with empty refs
TEST_F(TextColumnReaderTest, ReadBatchEmpty) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  std::vector<EncodedRef> empty_refs;
  auto read_result = reader->ReadBatch(empty_refs);
  ASSERT_TRUE(read_result.ok());
  ASSERT_TRUE(read_result.ValueOrDie().empty());

  ASSERT_STATUS_OK(reader->Close());
}

// test batch read with null refs (empty data)
TEST_F(TextColumnReaderTest, ReadBatchWithNullRefs) {
  std::vector<std::string> texts = {"hello", GenerateRandomString(50)};
  auto refs = WriteTestData(texts);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // include null refs in the batch
  std::vector<EncodedRef> encoded_refs = {
      {refs[0].data(), refs[0].size()},
      {nullptr, 0},  // null ref
      {refs[1].data(), refs[1].size()},
  };

  auto read_result = reader->ReadBatch(encoded_refs);
  ASSERT_TRUE(read_result.ok());
  auto read_texts = std::move(read_result).ValueOrDie();

  ASSERT_EQ(read_texts.size(), 3);
  ASSERT_EQ(read_texts[0], texts[0]);
  ASSERT_EQ(read_texts[1], "");  // null ref returns empty string
  ASSERT_EQ(read_texts[2], texts[1]);

  ASSERT_STATUS_OK(reader->Close());
}

// test batch read with only LOB texts (same file)
TEST_F(TextColumnReaderTest, ReadBatchAllLOBSameFile) {
  std::vector<std::string> texts;
  for (int i = 0; i < 10; i++) {
    texts.push_back(GenerateRandomString(50 + i * 10));
  }

  auto refs = WriteTestData(texts);
  ASSERT_EQ(refs.size(), 10);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  std::vector<EncodedRef> encoded_refs;
  for (const auto& ref : refs) {
    encoded_refs.push_back({ref.data(), ref.size()});
  }

  auto read_result = reader->ReadBatch(encoded_refs);
  ASSERT_TRUE(read_result.ok());
  auto read_texts = std::move(read_result).ValueOrDie();

  ASSERT_EQ(read_texts.size(), texts.size());
  for (size_t i = 0; i < texts.size(); i++) {
    ASSERT_EQ(read_texts[i], texts[i]);
  }

  ASSERT_STATUS_OK(reader->Close());
}

// test batch read in reverse order
TEST_F(TextColumnReaderTest, ReadBatchReverseOrder) {
  std::vector<std::string> texts;
  for (int i = 0; i < 5; i++) {
    texts.push_back(GenerateRandomString(50));
  }

  auto refs = WriteTestData(texts);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // reverse order
  std::vector<EncodedRef> encoded_refs;
  for (int i = refs.size() - 1; i >= 0; i--) {
    encoded_refs.push_back({refs[i].data(), refs[i].size()});
  }

  auto read_result = reader->ReadBatch(encoded_refs);
  ASSERT_TRUE(read_result.ok());
  auto read_texts = std::move(read_result).ValueOrDie();

  ASSERT_EQ(read_texts.size(), texts.size());
  for (size_t i = 0; i < texts.size(); i++) {
    ASSERT_EQ(read_texts[i], texts[texts.size() - 1 - i]);
  }

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Take API Tests ====================

// test Take with valid indices
TEST_F(TextColumnReaderTest, TakeValidIndices) {
  std::vector<std::string> texts;
  for (int i = 0; i < 10; i++) {
    texts.push_back(GenerateRandomString(50));
  }

  auto refs = WriteTestData(texts);

  // extract file_id from the first LOB reference
  ASSERT_TRUE(IsLOBReference(refs[0].data()));
  std::string file_id_str;
  int32_t row_offset;
  DecodeLOBReference(refs[0].data(), &file_id_str, &row_offset);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // take specific rows
  std::vector<int32_t> row_offsets = {0, 2, 5, 9};
  auto take_result = reader->Take(file_id_str, row_offsets);
  ASSERT_TRUE(take_result.ok()) << take_result.status().message();
  auto taken_texts = std::move(take_result).ValueOrDie();

  ASSERT_EQ(taken_texts.size(), 4);
  ASSERT_EQ(taken_texts[0], texts[0]);
  ASSERT_EQ(taken_texts[1], texts[2]);
  ASSERT_EQ(taken_texts[2], texts[5]);
  ASSERT_EQ(taken_texts[3], texts[9]);

  ASSERT_STATUS_OK(reader->Close());
}

// test Take with empty indices - Vortex Take API doesn't support empty indices
TEST_F(TextColumnReaderTest, TakeEmptyIndices) {
  std::vector<std::string> texts = {GenerateRandomString(50)};
  auto refs = WriteTestData(texts);

  std::string file_id_str;
  int32_t row_offset;
  DecodeLOBReference(refs[0].data(), &file_id_str, &row_offset);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // Note: Vortex Take API returns error for empty indices
  // This is expected behavior - caller should check for empty indices before calling Take
  std::vector<int32_t> empty_offsets;
  auto take_result = reader->Take(file_id_str, empty_offsets);
  ASSERT_FALSE(take_result.ok()) << "Take with empty indices should return error";

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== ReadArrowArray Tests ====================

// test ReadArrowArray with mixed data
TEST_F(TextColumnReaderTest, ReadArrowArrayMixed) {
  std::vector<std::string> texts = {
      "short1",
      GenerateRandomString(50),
      "short2",
  };

  auto refs = WriteTestData(texts);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  // build BinaryArray from refs
  arrow::BinaryBuilder builder;
  for (const auto& ref : refs) {
    ASSERT_STATUS_OK(builder.Append(ref.data(), ref.size()));
  }
  std::shared_ptr<arrow::BinaryArray> refs_array;
  ASSERT_STATUS_OK(builder.Finish(&refs_array));

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadArrowArray(refs_array);
  ASSERT_TRUE(read_result.ok()) << read_result.status().message();
  auto output_array = std::move(read_result).ValueOrDie();

  ASSERT_EQ(output_array->length(), refs_array->length());
  for (int64_t i = 0; i < output_array->length(); i++) {
    ASSERT_EQ(output_array->GetString(i), texts[i]);
  }

  ASSERT_STATUS_OK(reader->Close());
}

// test ReadArrowArray with nulls
TEST_F(TextColumnReaderTest, ReadArrowArrayWithNulls) {
  std::vector<std::string> texts = {"hello", GenerateRandomString(50)};
  auto refs = WriteTestData(texts);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  // build BinaryArray with nulls
  arrow::BinaryBuilder builder;
  ASSERT_STATUS_OK(builder.Append(refs[0].data(), refs[0].size()));
  ASSERT_STATUS_OK(builder.AppendNull());
  ASSERT_STATUS_OK(builder.Append(refs[1].data(), refs[1].size()));
  std::shared_ptr<arrow::BinaryArray> refs_array;
  ASSERT_STATUS_OK(builder.Finish(&refs_array));

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadArrowArray(refs_array);
  ASSERT_TRUE(read_result.ok());
  auto output_array = std::move(read_result).ValueOrDie();

  ASSERT_EQ(output_array->length(), 3);
  ASSERT_FALSE(output_array->IsNull(0));
  ASSERT_TRUE(output_array->IsNull(1));
  ASSERT_FALSE(output_array->IsNull(2));
  ASSERT_EQ(output_array->GetString(0), texts[0]);
  ASSERT_EQ(output_array->GetString(2), texts[1]);

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Cache Tests ====================

// test ClearCache
TEST_F(TextColumnReaderTest, ClearCache) {
  std::vector<std::string> texts = {GenerateRandomString(50), GenerateRandomString(60)};
  auto refs = WriteTestData(texts);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // read to populate cache
  auto read1 = reader->ReadText(refs[0].data(), refs[0].size());
  ASSERT_TRUE(read1.ok());

  // clear cache
  reader->ClearCache();

  // read again - should still work (re-opens file)
  auto read2 = reader->ReadText(refs[1].data(), refs[1].size());
  ASSERT_TRUE(read2.ok());
  ASSERT_EQ(read2.ValueOrDie(), texts[1]);

  ASSERT_STATUS_OK(reader->Close());
}

// test multiple reads reuse cache
TEST_F(TextColumnReaderTest, MultipleReadsReuseCache) {
  std::vector<std::string> texts;
  for (int i = 0; i < 100; i++) {
    texts.push_back(GenerateRandomString(50));
  }
  auto refs = WriteTestData(texts);

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  // read all texts multiple times
  for (int round = 0; round < 3; round++) {
    for (size_t i = 0; i < texts.size(); i++) {
      auto read_result = reader->ReadText(refs[i].data(), refs[i].size());
      ASSERT_TRUE(read_result.ok()) << "round " << round << ", index " << i;
      ASSERT_EQ(read_result.ValueOrDie(), texts[i]);
    }
  }

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Large Data Tests ====================

// test reading large text
TEST_F(TextColumnReaderTest, ReadLargeText) {
  // create a 1MB text
  std::string large_text = GenerateRandomString(1024 * 1024);
  auto refs = WriteTestData({large_text});

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read_result = reader->ReadText(refs[0].data(), refs[0].size());
  ASSERT_TRUE(read_result.ok()) << read_result.status().message();
  ASSERT_EQ(read_result.ValueOrDie(), large_text);

  ASSERT_STATUS_OK(reader->Close());
}

}  // namespace milvus_storage::text_column

#endif  // BUILD_VORTEX_BRIDGE
