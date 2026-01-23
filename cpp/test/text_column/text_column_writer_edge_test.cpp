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

class TextColumnWriterEdgeTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // create test directory
    test_dir_ = "/tmp/text_column_writer_edge_test_" + std::to_string(std::random_device{}());
    boost::filesystem::create_directories(test_dir_);

    // create filesystem
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();

    // create config with small thresholds for testing
    config_.lob_base_path = test_dir_ + "/lobs/100";
    config_.field_id = 100;
    config_.inline_threshold = 20;              // small threshold for testing
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

// ==================== Boundary Threshold Tests ====================

// test text exactly at inline threshold (should be inline)
TEST_F(TextColumnWriterEdgeTest, TextExactlyAtThreshold) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // text with length = threshold - 1 (should be inline)
  std::string text_under = GenerateRandomString(config_.inline_threshold - 1);
  auto ref_under = writer->WriteText(text_under);
  ASSERT_TRUE(ref_under.ok());
  ASSERT_TRUE(IsInlineData(ref_under.ValueOrDie().data()));

  // text with length = threshold (should be LOB)
  std::string text_at = GenerateRandomString(config_.inline_threshold);
  auto ref_at = writer->WriteText(text_at);
  ASSERT_TRUE(ref_at.ok());
  ASSERT_TRUE(IsLOBReference(ref_at.ValueOrDie().data()));

  // text with length = threshold + 1 (should be LOB)
  std::string text_over = GenerateRandomString(config_.inline_threshold + 1);
  auto ref_over = writer->WriteText(text_over);
  ASSERT_TRUE(ref_over.ok());
  ASSERT_TRUE(IsLOBReference(ref_over.ValueOrDie().data()));

  auto stats = writer->GetStats();
  ASSERT_EQ(stats.inline_texts, 1);
  ASSERT_EQ(stats.lob_texts, 2);

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
}

// ==================== Unicode/Multi-byte Character Tests ====================

// test Unicode text (Chinese characters)
TEST_F(TextColumnWriterEdgeTest, UnicodeTextChinese) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // short Chinese text (should be inline)
  std::string short_chinese = "你好";  // 6 bytes in UTF-8
  auto ref1 = writer->WriteText(short_chinese);
  ASSERT_TRUE(ref1.ok());
  ASSERT_TRUE(IsInlineData(ref1.ValueOrDie().data()));

  // longer Chinese text (should be LOB)
  std::string long_chinese = "这是一段很长的中文文本，用于测试LOB存储功能是否正常工作。";
  auto ref2 = writer->WriteText(long_chinese);
  ASSERT_TRUE(ref2.ok());
  ASSERT_TRUE(IsLOBReference(ref2.ValueOrDie().data()));

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());

  // verify read back
  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read1 = reader->ReadText(ref1.ValueOrDie().data(), ref1.ValueOrDie().size());
  ASSERT_TRUE(read1.ok());
  ASSERT_EQ(read1.ValueOrDie(), short_chinese);

  auto read2 = reader->ReadText(ref2.ValueOrDie().data(), ref2.ValueOrDie().size());
  ASSERT_TRUE(read2.ok());
  ASSERT_EQ(read2.ValueOrDie(), long_chinese);

  ASSERT_STATUS_OK(reader->Close());
}

// test Unicode text (emoji)
TEST_F(TextColumnWriterEdgeTest, UnicodeTextEmoji) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // emoji text
  std::string emoji_text = "Hello 👋 World 🌍 Test 🧪";
  auto ref = writer->WriteText(emoji_text);
  ASSERT_TRUE(ref.ok());

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());

  // verify read back
  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read = reader->ReadText(ref.ValueOrDie().data(), ref.ValueOrDie().size());
  ASSERT_TRUE(read.ok());
  ASSERT_EQ(read.ValueOrDie(), emoji_text);

  ASSERT_STATUS_OK(reader->Close());
}

// test mixed Unicode and ASCII
TEST_F(TextColumnWriterEdgeTest, MixedUnicodeAndASCII) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  std::string mixed_text = "Hello 你好 مرحبا שלום こんにちは 🎉";
  auto ref = writer->WriteText(mixed_text);
  ASSERT_TRUE(ref.ok());

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());

  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read = reader->ReadText(ref.ValueOrDie().data(), ref.ValueOrDie().size());
  ASSERT_TRUE(read.ok());
  ASSERT_EQ(read.ValueOrDie(), mixed_text);

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Empty and Special Cases ====================

// test empty batch write
TEST_F(TextColumnWriterEdgeTest, EmptyBatchWrite) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  std::vector<std::string> empty_batch;
  auto refs_result = writer->WriteBatch(empty_batch);
  ASSERT_TRUE(refs_result.ok());
  ASSERT_TRUE(refs_result.ValueOrDie().empty());

  ASSERT_EQ(writer->WrittenRows(), 0);

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
  ASSERT_TRUE(close_result.ValueOrDie().empty());  // no LOB files created
}

// test multiple empty strings
TEST_F(TextColumnWriterEdgeTest, MultipleEmptyStrings) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  std::vector<std::string> texts = {"", "", "", ""};
  auto refs_result = writer->WriteBatch(texts);
  ASSERT_TRUE(refs_result.ok());
  auto refs = std::move(refs_result).ValueOrDie();

  ASSERT_EQ(refs.size(), 4);
  for (const auto& ref : refs) {
    ASSERT_TRUE(IsInlineData(ref.data()));
    ASSERT_EQ(ref.size(), 1);  // just the flag byte
  }

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());

  // verify read back
  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  for (size_t i = 0; i < refs.size(); i++) {
    auto read = reader->ReadText(refs[i].data(), refs[i].size());
    ASSERT_TRUE(read.ok());
    ASSERT_EQ(read.ValueOrDie(), "");
  }

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Multiple Flush Tests ====================

// test multiple flushes
TEST_F(TextColumnWriterEdgeTest, MultipleFlushes) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  std::vector<std::vector<uint8_t>> all_refs;

  // write and flush multiple times
  for (int round = 0; round < 5; round++) {
    std::string text = GenerateRandomString(50);
    auto ref_result = writer->WriteText(text);
    ASSERT_TRUE(ref_result.ok());
    all_refs.push_back(std::move(ref_result).ValueOrDie());

    ASSERT_STATUS_OK(writer->Flush());
  }

  ASSERT_EQ(writer->WrittenRows(), 5);

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
}

// test flush on empty writer
TEST_F(TextColumnWriterEdgeTest, FlushOnEmptyWriter) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // flush without writing anything
  ASSERT_STATUS_OK(writer->Flush());
  ASSERT_STATUS_OK(writer->Flush());
  ASSERT_STATUS_OK(writer->Flush());

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
  ASSERT_TRUE(close_result.ValueOrDie().empty());
}

// ==================== Abort Tests ====================

// test abort immediately after write
TEST_F(TextColumnWriterEdgeTest, AbortAfterWrite) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // write some LOB texts
  for (int i = 0; i < 10; i++) {
    auto ref = writer->WriteText(GenerateRandomString(100));
    ASSERT_TRUE(ref.ok());
  }

  // abort
  ASSERT_STATUS_OK(writer->Abort());
  ASSERT_TRUE(writer->IsClosed());

  // verify no files remain
  auto lob_data_dir = config_.lob_base_path + "/_data";
  if (boost::filesystem::exists(lob_data_dir)) {
    ASSERT_TRUE(boost::filesystem::is_empty(lob_data_dir));
  }
}

// test abort without writing
TEST_F(TextColumnWriterEdgeTest, AbortWithoutWrite) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // abort immediately
  ASSERT_STATUS_OK(writer->Abort());
  ASSERT_TRUE(writer->IsClosed());
}

// test double abort
TEST_F(TextColumnWriterEdgeTest, DoubleAbort) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  auto ref = writer->WriteText(GenerateRandomString(100));
  ASSERT_TRUE(ref.ok());

  ASSERT_STATUS_OK(writer->Abort());
  ASSERT_STATUS_OK(writer->Abort());  // should be idempotent
}

// ==================== Write After Close/Abort Tests ====================

// test write after close
TEST_F(TextColumnWriterEdgeTest, WriteAfterClose) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());

  // try to write after close
  auto ref = writer->WriteText("test");
  ASSERT_FALSE(ref.ok());
}

// test write after abort
TEST_F(TextColumnWriterEdgeTest, WriteAfterAbort) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  ASSERT_STATUS_OK(writer->Abort());

  // try to write after abort
  auto ref = writer->WriteText("test");
  ASSERT_FALSE(ref.ok());
}

// ==================== Large Text Tests ====================

// test very large text (10MB)
TEST_F(TextColumnWriterEdgeTest, VeryLargeText) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // 10MB text
  std::string large_text = GenerateRandomString(10 * 1024 * 1024);
  auto ref = writer->WriteText(large_text);
  ASSERT_TRUE(ref.ok()) << ref.status().message();
  ASSERT_TRUE(IsLOBReference(ref.ValueOrDie().data()));

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());

  // verify read back
  auto reader_result = manager->CreateReader();
  ASSERT_TRUE(reader_result.ok());
  auto reader = std::move(reader_result).ValueOrDie();

  auto read = reader->ReadText(ref.ValueOrDie().data(), ref.ValueOrDie().size());
  ASSERT_TRUE(read.ok()) << read.status().message();
  ASSERT_EQ(read.ValueOrDie(), large_text);

  ASSERT_STATUS_OK(reader->Close());
}

// ==================== Stats Tests ====================

// test stats accuracy
TEST_F(TextColumnWriterEdgeTest, StatsAccuracy) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // 5 inline texts
  for (int i = 0; i < 5; i++) {
    auto ref = writer->WriteText("short");
    ASSERT_TRUE(ref.ok());
  }

  // 10 LOB texts
  size_t lob_bytes = 0;
  for (int i = 0; i < 10; i++) {
    std::string text = GenerateRandomString(100);
    lob_bytes += text.size();
    auto ref = writer->WriteText(text);
    ASSERT_TRUE(ref.ok());
  }

  auto stats = writer->GetStats();
  ASSERT_EQ(stats.total_texts, 15);
  ASSERT_EQ(stats.inline_texts, 5);
  ASSERT_EQ(stats.lob_texts, 10);
  ASSERT_EQ(stats.total_bytes, 5 * 5 + lob_bytes);  // 5 "short" + lob_bytes

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
}

// ==================== Arrow Array Interface Tests ====================

// test WriteArrowArray with all nulls
TEST_F(TextColumnWriterEdgeTest, WriteArrowArrayAllNulls) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // build array with all nulls
  arrow::StringBuilder builder;
  ASSERT_STATUS_OK(builder.AppendNull());
  ASSERT_STATUS_OK(builder.AppendNull());
  ASSERT_STATUS_OK(builder.AppendNull());

  std::shared_ptr<arrow::StringArray> input_array;
  ASSERT_STATUS_OK(builder.Finish(&input_array));

  auto refs_result = writer->WriteArrowArray(input_array);
  ASSERT_TRUE(refs_result.ok());
  auto refs_array = std::move(refs_result).ValueOrDie();

  ASSERT_EQ(refs_array->length(), 3);
  for (int64_t i = 0; i < 3; i++) {
    ASSERT_TRUE(refs_array->IsNull(i));
  }

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
}

// test WriteArrowArray with empty array
TEST_F(TextColumnWriterEdgeTest, WriteArrowArrayEmpty) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  auto writer_result = manager->CreateWriter();
  ASSERT_TRUE(writer_result.ok());
  auto writer = std::move(writer_result).ValueOrDie();

  // build empty array
  arrow::StringBuilder builder;
  std::shared_ptr<arrow::StringArray> input_array;
  ASSERT_STATUS_OK(builder.Finish(&input_array));

  auto refs_result = writer->WriteArrowArray(input_array);
  ASSERT_TRUE(refs_result.ok());
  auto refs_array = std::move(refs_result).ValueOrDie();

  ASSERT_EQ(refs_array->length(), 0);

  auto close_result = writer->Close();
  ASSERT_TRUE(close_result.ok());
}

}  // namespace milvus_storage::text_column

#endif  // BUILD_VORTEX_BRIDGE
