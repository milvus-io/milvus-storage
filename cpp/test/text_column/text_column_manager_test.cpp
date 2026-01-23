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

class TextColumnManagerTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // create test directory
    test_dir_ = "/tmp/text_column_test_" + std::to_string(std::random_device{}());
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

  protected:
  std::string test_dir_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  TextColumnConfig config_;
};

// test manager creation
TEST_F(TextColumnManagerTest, CreateManager) {
  auto result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(result.ok()) << result.status().message();

  auto manager = std::move(result).ValueOrDie();
  ASSERT_NE(manager, nullptr);
  ASSERT_EQ(manager->GetConfig().field_id, 100);
  ASSERT_EQ(manager->GetConfig().lob_base_path, config_.lob_base_path);
}

// test manager creation with invalid config
TEST_F(TextColumnManagerTest, CreateManagerInvalidConfig) {
  // null filesystem
  auto result1 = TextColumnManager::Create(nullptr, config_);
  ASSERT_FALSE(result1.ok());

  // empty lob_base_path
  TextColumnConfig invalid_config = config_;
  invalid_config.lob_base_path = "";
  auto result2 = TextColumnManager::Create(fs_, invalid_config);
  ASSERT_FALSE(result2.ok());

  // zero inline threshold
  invalid_config = config_;
  invalid_config.inline_threshold = 0;
  auto result3 = TextColumnManager::Create(fs_, invalid_config);
  ASSERT_FALSE(result3.ok());
}

// test writing inline text
TEST_F(TextColumnManagerTest, WriteInlineText) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // write short text (should be inline)
  std::string short_text = "hello";
  auto ref_result = writer->WriteText(short_text);
  ASSERT_TRUE(ref_result.ok()) << ref_result.status().message();

  auto ref = std::move(ref_result).ValueOrDie();
  // inline text: flag (1 byte) + text bytes
  ASSERT_EQ(ref.size(), 1 + short_text.size());
  ASSERT_TRUE(IsInlineData(ref.data()));
  ASSERT_EQ(DecodeInlineText(ref.data(), ref.size()), short_text);

  // check stats
  auto stats = writer->GetStats();
  ASSERT_EQ(stats.total_texts, 1);
  ASSERT_EQ(stats.inline_texts, 1);
  ASSERT_EQ(stats.lob_texts, 0);

  // close writer
  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
  auto files = std::move(close_result).ValueOrDie();
  ASSERT_EQ(files.size(), 0);  // no LOB files created for inline text
}

// test writing LOB text
TEST_F(TextColumnManagerTest, WriteLOBText) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // write long text (should be LOB)
  std::string long_text = GenerateRandomString(100);
  auto ref_result = writer->WriteText(long_text);
  ASSERT_TRUE(ref_result.ok()) << ref_result.status().message();

  auto ref = std::move(ref_result).ValueOrDie();
  // LOB reference is fixed 24 bytes (with padding)
  ASSERT_EQ(ref.size(), LOB_REFERENCE_SIZE);
  ASSERT_TRUE(IsLOBReference(ref.data()));

  // check stats
  auto stats = writer->GetStats();
  ASSERT_EQ(stats.total_texts, 1);
  ASSERT_EQ(stats.inline_texts, 0);
  ASSERT_EQ(stats.lob_texts, 1);

  // close writer
  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
  auto files = std::move(close_result).ValueOrDie();
  ASSERT_EQ(files.size(), 1);  // one LOB file created
}

// test batch write
TEST_F(TextColumnManagerTest, WriteBatch) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // prepare mixed texts (some inline, some LOB)
  std::vector<std::string> texts = {
      "short1",                   // inline (6 bytes < 20)
      GenerateRandomString(100),  // LOB
      "short2",                   // inline
      GenerateRandomString(200),  // LOB
      "short3",                   // inline
  };

  auto refs_result = writer->WriteBatch(texts);
  ASSERT_TRUE(refs_result.ok()) << refs_result.status().message();

  auto refs = std::move(refs_result).ValueOrDie();
  ASSERT_EQ(refs.size(), 5);

  // verify inline vs LOB
  ASSERT_TRUE(IsInlineData(refs[0].data()));
  ASSERT_TRUE(IsLOBReference(refs[1].data()));
  ASSERT_TRUE(IsInlineData(refs[2].data()));
  ASSERT_TRUE(IsLOBReference(refs[3].data()));
  ASSERT_TRUE(IsInlineData(refs[4].data()));

  // verify inline text sizes (flag + text length)
  ASSERT_EQ(refs[0].size(), 1 + texts[0].size());
  ASSERT_EQ(refs[2].size(), 1 + texts[2].size());
  ASSERT_EQ(refs[4].size(), 1 + texts[4].size());

  // verify LOB reference sizes (fixed 24 bytes with padding)
  ASSERT_EQ(refs[1].size(), LOB_REFERENCE_SIZE);
  ASSERT_EQ(refs[3].size(), LOB_REFERENCE_SIZE);

  // check stats
  auto stats = writer->GetStats();
  ASSERT_EQ(stats.total_texts, 5);
  ASSERT_EQ(stats.inline_texts, 3);
  ASSERT_EQ(stats.lob_texts, 2);

  // close writer
  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
}

// test write and read round trip
TEST_F(TextColumnManagerTest, WriteAndRead) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  // prepare test texts
  std::vector<std::string> texts = {
      "inline1",                  // inline
      GenerateRandomString(50),   // LOB
      "inline2",                  // inline
      GenerateRandomString(100),  // LOB
      GenerateRandomString(150),  // LOB
  };

  // write texts
  std::vector<std::vector<uint8_t>> refs;
  {
    auto writer_result = manager->CreateWriter();
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    auto refs_result = writer->WriteBatch(texts);
    ASSERT_TRUE(refs_result.ok());
    refs = std::move(refs_result).ValueOrDie();

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
  }

  // read texts back
  {
    auto reader_result = manager->CreateReader();
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

    // read one by one
    for (size_t i = 0; i < texts.size(); i++) {
      auto text_result = reader->ReadText(refs[i].data(), refs[i].size());
      ASSERT_TRUE(text_result.ok()) << "failed to read text " << i << ": " << text_result.status().message();
      ASSERT_EQ(text_result.ValueOrDie(), texts[i]);
    }

    // read in batch
    std::vector<EncodedRef> encoded_refs;
    for (const auto& ref : refs) {
      encoded_refs.push_back({ref.data(), ref.size()});
    }

    auto batch_result = reader->ReadBatch(encoded_refs);
    ASSERT_TRUE(batch_result.ok()) << batch_result.status().message();
    auto read_texts = std::move(batch_result).ValueOrDie();

    ASSERT_EQ(read_texts.size(), texts.size());
    for (size_t i = 0; i < texts.size(); i++) {
      ASSERT_EQ(read_texts[i], texts[i]);
    }

    auto close_result = reader->Close();
    ASSERT_TRUE(close_result.ok());
  }
}

// test file rolling based on bytes
TEST_F(TextColumnManagerTest, FileRolling) {
  // set small max bytes to trigger rolling
  config_.max_lob_file_bytes = 1024;    // 1KB per file
  config_.flush_threshold_bytes = 512;  // flush at 512 bytes

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // write texts that will exceed max_lob_file_bytes
  std::vector<std::string> texts;
  std::vector<std::vector<uint8_t>> refs;

  // each text is 100 bytes, write 25 = 2500 bytes total
  // should create at least 2 files (2500 / 1024 = ~2.4)
  for (int i = 0; i < 25; i++) {
    texts.push_back(GenerateRandomString(100));  // all LOB
  }

  auto refs_result = writer->WriteBatch(texts);
  ASSERT_TRUE(refs_result.ok());
  refs = std::move(refs_result).ValueOrDie();

  // close writer
  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
  auto files = std::move(close_result).ValueOrDie();

  // should have created multiple files
  ASSERT_GE(files.size(), 2);

  // verify we can still read all texts
  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  std::vector<EncodedRef> encoded_refs;
  for (const auto& ref : refs) {
    encoded_refs.push_back({ref.data(), ref.size()});
  }

  auto batch_result = reader->ReadBatch(encoded_refs);
  ASSERT_TRUE(batch_result.ok());
  auto read_texts = std::move(batch_result).ValueOrDie();

  ASSERT_EQ(read_texts.size(), texts.size());
  for (size_t i = 0; i < texts.size(); i++) {
    ASSERT_EQ(read_texts[i], texts[i]);
  }
}

// test abort
TEST_F(TextColumnManagerTest, WriterAbort) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // write some LOB texts
  std::vector<std::string> texts;
  for (int i = 0; i < 5; i++) {
    texts.push_back(GenerateRandomString(50));
  }

  auto refs_result = writer->WriteBatch(texts);
  ASSERT_TRUE(refs_result.ok());

  // abort instead of close
  auto abort_result = writer->Abort();
  ASSERT_TRUE(abort_result.ok());
  ASSERT_TRUE(writer->IsClosed());

  // verify no files remain
  auto lob_data_dir = config_.lob_base_path + "/_data";
  if (boost::filesystem::exists(lob_data_dir)) {
    ASSERT_TRUE(boost::filesystem::is_empty(lob_data_dir));
  }
}

// test Arrow array interface
TEST_F(TextColumnManagerTest, ArrowArrayInterface) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  // prepare string array
  arrow::StringBuilder builder;
  ASSERT_TRUE(builder.Append("inline1").ok());
  ASSERT_TRUE(builder.Append(GenerateRandomString(50)).ok());
  ASSERT_TRUE(builder.Append("inline2").ok());
  ASSERT_TRUE(builder.AppendNull().ok());  // null value
  ASSERT_TRUE(builder.Append(GenerateRandomString(100)).ok());

  std::shared_ptr<arrow::StringArray> input_array;
  ASSERT_TRUE(builder.Finish(&input_array).ok());

  // write using Arrow interface
  std::shared_ptr<arrow::BinaryArray> refs_array;
  {
    auto writer_result = manager->CreateWriter();
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    auto refs_result = writer->WriteArrowArray(input_array);
    ASSERT_TRUE(refs_result.ok()) << refs_result.status().message();
    refs_array = std::move(refs_result).ValueOrDie();

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
  }

  ASSERT_EQ(refs_array->length(), input_array->length());

  // read using Arrow interface
  {
    auto reader_result = manager->CreateReader();
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

    auto output_result = reader->ReadArrowArray(refs_array);
    ASSERT_TRUE(output_result.ok()) << output_result.status().message();
    auto output_array = std::move(output_result).ValueOrDie();

    ASSERT_EQ(output_array->length(), input_array->length());

    // verify values
    for (int64_t i = 0; i < input_array->length(); i++) {
      if (input_array->IsNull(i)) {
        ASSERT_TRUE(output_array->IsNull(i));
      } else {
        ASSERT_FALSE(output_array->IsNull(i));
        ASSERT_EQ(output_array->GetString(i), input_array->GetString(i));
      }
    }

    auto close_result = reader->Close();
    ASSERT_TRUE(close_result.ok());
  }
}

// test inline text with various sizes
TEST_F(TextColumnManagerTest, InlineTextVariableSizes) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // test various inline text sizes
  std::vector<std::string> texts = {
      "",                    // empty
      "a",                   // 1 byte
      "ab",                  // 2 bytes
      "hello world!",        // 12 bytes
      "19 bytes of text!!",  // 19 bytes (just under threshold)
  };

  auto refs_result = writer->WriteBatch(texts);
  ASSERT_TRUE(refs_result.ok());
  auto refs = std::move(refs_result).ValueOrDie();

  // all should be inline
  for (size_t i = 0; i < refs.size(); i++) {
    ASSERT_TRUE(IsInlineData(refs[i].data())) << "text " << i << " should be inline";
    ASSERT_EQ(refs[i].size(), 1 + texts[i].size()) << "text " << i << " has wrong size";
    ASSERT_EQ(DecodeInlineText(refs[i].data(), refs[i].size()), texts[i]) << "text " << i << " decoded incorrectly";
  }

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
}

}  // namespace milvus_storage::text_column

#endif  // BUILD_VORTEX_BRIDGE
