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

#include <random>
#include <string>
#include <vector>

#include "test_env.h"
#include "milvus-storage/text_column/text_column_manager.h"
#include "milvus-storage/text_column/text_column_writer.h"
#include "milvus-storage/text_column/text_column_reader.h"
#include "milvus-storage/text_column/lob_reference.h"

namespace milvus_storage::text_column {

// Integration tests for text column on cloud storage (S3/MinIO)
// These tests are skipped in local environment
class TextColumnCloudTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (!IsCloudEnv()) {
      GTEST_SKIP() << "Cloud storage tests skipped in local environment";
    }

    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    // create test base path
    test_base_path_ = GetTestBasePath("text-column-cloud-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, test_base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, test_base_path_));

    // create config
    config_.lob_base_path = test_base_path_ + "/lobs/100";
    config_.field_id = 100;
    config_.inline_threshold = 20;
    config_.max_lob_file_bytes = 1024 * 1024;   // 1MB
    config_.flush_threshold_bytes = 64 * 1024;  // 64KB
    config_.properties = properties_;
  }

  void TearDown() override {
    if (IsCloudEnv()) {
      ASSERT_STATUS_OK(DeleteTestDir(fs_, test_base_path_));
    }
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
  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string test_base_path_;
  TextColumnConfig config_;
};

// ==================== Basic Cloud Storage Tests ====================

// test basic write and read on cloud storage
TEST_F(TextColumnCloudTest, BasicWriteAndRead) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok()) << manager_result.status().message();
  auto manager = std::move(manager_result).ValueOrDie();

  // prepare test data - mix of inline and LOB
  std::vector<std::string> texts = {
      "short1",                   // inline
      GenerateRandomString(50),   // LOB
      "short2",                   // inline
      GenerateRandomString(100),  // LOB
      "short3",                   // inline
  };

  // write
  std::vector<std::vector<uint8_t>> refs;
  {
    auto writer_result = manager->CreateWriter();
    ASSERT_TRUE(writer_result.ok()) << writer_result.status().message();
    auto writer = std::move(writer_result).ValueOrDie();

    auto refs_result = writer->WriteBatch(texts);
    ASSERT_TRUE(refs_result.ok()) << refs_result.status().message();
    refs = std::move(refs_result).ValueOrDie();

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok()) << close_result.status().message();
  }

  // read
  {
    auto reader_result = manager->CreateReader();
    ASSERT_TRUE(reader_result.ok()) << reader_result.status().message();
    auto reader = std::move(reader_result).ValueOrDie();

    for (size_t i = 0; i < texts.size(); i++) {
      auto read_result = reader->ReadText(refs[i].data(), refs[i].size());
      ASSERT_TRUE(read_result.ok()) << "index " << i << ": " << read_result.status().message();
      ASSERT_EQ(read_result.ValueOrDie(), texts[i]);
    }

    ASSERT_STATUS_OK(reader->Close());
  }
}

// test batch write and read on cloud storage
TEST_F(TextColumnCloudTest, BatchWriteAndRead) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  // prepare larger batch
  std::vector<std::string> texts;
  for (int i = 0; i < 100; i++) {
    if (i % 3 == 0) {
      texts.push_back("inline_" + std::to_string(i));
    } else {
      texts.push_back(GenerateRandomString(50 + i % 50));
    }
  }

  // write
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

  // batch read
  {
    auto reader_result = manager->CreateReader();
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

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
}

// test file rolling on cloud storage
TEST_F(TextColumnCloudTest, FileRolling) {
  // use small max bytes to trigger rolling
  config_.max_lob_file_bytes = 1024;    // 1KB per file
  config_.flush_threshold_bytes = 512;  // flush at 512 bytes

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  // write texts that will exceed max_lob_file_bytes
  std::vector<std::string> texts;
  std::vector<std::vector<uint8_t>> refs;

  {
    auto writer_result = manager->CreateWriter();
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    // each text is 100 bytes, write 25 = 2500 bytes total
    // should create at least 2 files
    for (int i = 0; i < 25; i++) {
      texts.push_back(GenerateRandomString(100));
    }

    auto refs_result = writer->WriteBatch(texts);
    ASSERT_TRUE(refs_result.ok());
    refs = std::move(refs_result).ValueOrDie();

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
    auto files = std::move(close_result).ValueOrDie();

    // should have created multiple files
    ASSERT_GE(files.size(), 2);
  }

  // verify we can still read all texts
  {
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
}

// test cross-file batch read (reads from multiple LOB files)
TEST_F(TextColumnCloudTest, CrossFileBatchRead) {
  // use small max bytes to create multiple files
  config_.max_lob_file_bytes = 512;
  config_.flush_threshold_bytes = 256;

  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  std::vector<std::string> texts;
  std::vector<std::vector<uint8_t>> refs;

  // write texts that will span multiple files
  {
    auto writer_result = manager->CreateWriter();
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    for (int i = 0; i < 20; i++) {
      texts.push_back(GenerateRandomString(100));
    }

    auto refs_result = writer->WriteBatch(texts);
    ASSERT_TRUE(refs_result.ok());
    refs = std::move(refs_result).ValueOrDie();

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
    ASSERT_GE(close_result.ValueOrDie().size(), 2);  // at least 2 files
  }

  // read in random order (should still work with cross-file reads)
  {
    auto reader_result = manager->CreateReader();
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

    // shuffle indices
    std::vector<size_t> indices(refs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // read in shuffled order
    std::vector<EncodedRef> shuffled_refs;
    for (size_t idx : indices) {
      shuffled_refs.push_back({refs[idx].data(), refs[idx].size()});
    }

    auto read_result = reader->ReadBatch(shuffled_refs);
    ASSERT_TRUE(read_result.ok());
    auto read_texts = std::move(read_result).ValueOrDie();

    ASSERT_EQ(read_texts.size(), texts.size());
    for (size_t i = 0; i < indices.size(); i++) {
      ASSERT_EQ(read_texts[i], texts[indices[i]]);
    }

    ASSERT_STATUS_OK(reader->Close());
  }
}

// test large text on cloud storage
TEST_F(TextColumnCloudTest, LargeText) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  // 5MB text
  std::string large_text = GenerateRandomString(5 * 1024 * 1024);
  std::vector<uint8_t> ref;

  // write
  {
    auto writer_result = manager->CreateWriter();
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    auto ref_result = writer->WriteText(large_text);
    ASSERT_TRUE(ref_result.ok()) << ref_result.status().message();
    ref = std::move(ref_result).ValueOrDie();

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
  }

  // read
  {
    auto reader_result = manager->CreateReader();
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

    auto read_result = reader->ReadText(ref.data(), ref.size());
    ASSERT_TRUE(read_result.ok()) << read_result.status().message();
    ASSERT_EQ(read_result.ValueOrDie(), large_text);

    ASSERT_STATUS_OK(reader->Close());
  }
}

// test abort on cloud storage
TEST_F(TextColumnCloudTest, AbortCleansUpFiles) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  {
    auto writer_result = manager->CreateWriter();
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    // write some LOB texts
    for (int i = 0; i < 5; i++) {
      auto ref = writer->WriteText(GenerateRandomString(100));
      ASSERT_TRUE(ref.ok());
    }

    // abort
    ASSERT_STATUS_OK(writer->Abort());
  }

  // verify LOB data directory is empty or doesn't exist
  auto lob_data_path = config_.lob_base_path + "/_data";
  arrow::fs::FileSelector selector;
  selector.base_dir = lob_data_path;
  selector.recursive = false;

  auto files_result = fs_->GetFileInfo(selector);
  if (files_result.ok()) {
    // directory exists - should be empty
    ASSERT_TRUE(files_result.ValueOrDie().empty());
  }
  // if directory doesn't exist, that's also OK
}

// test Arrow array interface on cloud storage
TEST_F(TextColumnCloudTest, ArrowArrayInterface) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  // prepare string array
  arrow::StringBuilder builder;
  ASSERT_STATUS_OK(builder.Append("inline1"));
  ASSERT_STATUS_OK(builder.Append(GenerateRandomString(50)));
  ASSERT_STATUS_OK(builder.AppendNull());
  ASSERT_STATUS_OK(builder.Append("inline2"));
  ASSERT_STATUS_OK(builder.Append(GenerateRandomString(100)));

  std::shared_ptr<arrow::StringArray> input_array;
  ASSERT_STATUS_OK(builder.Finish(&input_array));

  // write
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

  // read
  {
    auto reader_result = manager->CreateReader();
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

    auto output_result = reader->ReadArrowArray(refs_array);
    ASSERT_TRUE(output_result.ok()) << output_result.status().message();
    auto output_array = std::move(output_result).ValueOrDie();

    ASSERT_EQ(output_array->length(), input_array->length());

    for (int64_t i = 0; i < input_array->length(); i++) {
      if (input_array->IsNull(i)) {
        ASSERT_TRUE(output_array->IsNull(i));
      } else {
        ASSERT_FALSE(output_array->IsNull(i));
        ASSERT_EQ(output_array->GetString(i), input_array->GetString(i));
      }
    }

    ASSERT_STATUS_OK(reader->Close());
  }
}

// test Take API on cloud storage
TEST_F(TextColumnCloudTest, TakeAPI) {
  auto manager_result = TextColumnManager::Create(fs_, config_);
  ASSERT_TRUE(manager_result.ok());
  auto manager = std::move(manager_result).ValueOrDie();

  std::vector<std::string> texts;
  std::vector<std::vector<uint8_t>> refs;

  // write
  {
    auto writer_result = manager->CreateWriter();
    ASSERT_TRUE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();

    for (int i = 0; i < 20; i++) {
      texts.push_back(GenerateRandomString(50));
    }

    auto refs_result = writer->WriteBatch(texts);
    ASSERT_TRUE(refs_result.ok());
    refs = std::move(refs_result).ValueOrDie();

    auto close_result = writer->Close();
    ASSERT_TRUE(close_result.ok());
  }

  // extract file_id from first ref
  std::string file_id_str;
  int32_t row_offset;
  DecodeLOBReference(refs[0].data(), &file_id_str, &row_offset);

  // read using Take
  {
    auto reader_result = manager->CreateReader();
    ASSERT_TRUE(reader_result.ok());
    auto reader = std::move(reader_result).ValueOrDie();

    std::vector<int32_t> take_indices = {0, 5, 10, 15, 19};
    auto take_result = reader->Take(file_id_str, take_indices);
    ASSERT_TRUE(take_result.ok()) << take_result.status().message();
    auto taken_texts = std::move(take_result).ValueOrDie();

    ASSERT_EQ(taken_texts.size(), take_indices.size());
    for (size_t i = 0; i < take_indices.size(); i++) {
      ASSERT_EQ(taken_texts[i], texts[take_indices[i]]);
    }

    ASSERT_STATUS_OK(reader->Close());
  }
}

}  // namespace milvus_storage::text_column

#endif  // BUILD_VORTEX_BRIDGE
