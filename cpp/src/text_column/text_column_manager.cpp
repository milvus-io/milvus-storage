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

#include "milvus-storage/text_column/text_column_manager.h"

#include "milvus-storage/text_column/text_column_reader.h"
#include "milvus-storage/text_column/text_column_writer.h"

namespace milvus_storage::text_column {

// implementation of TextColumnManager
class TextColumnManagerImpl : public TextColumnManager {
  public:
  TextColumnManagerImpl(std::shared_ptr<arrow::fs::FileSystem> fs, TextColumnConfig config)
      : fs_(std::move(fs)), config_(std::move(config)) {}

  arrow::Result<std::unique_ptr<TextColumnWriter>> CreateWriter() override {
    return CreateTextColumnWriter(fs_, config_);
  }

  arrow::Result<std::unique_ptr<TextColumnReader>> CreateReader() override {
    return CreateTextColumnReader(fs_, config_);
  }

  const TextColumnConfig& GetConfig() const override { return config_; }

  std::shared_ptr<arrow::fs::FileSystem> GetFileSystem() const override { return fs_; }

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  TextColumnConfig config_;
};

// static factory method
arrow::Result<std::unique_ptr<TextColumnManager>> TextColumnManager::Create(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                            const TextColumnConfig& config) {
  if (!fs) {
    return arrow::Status::Invalid("filesystem is null");
  }

  if (config.lob_base_path.empty()) {
    return arrow::Status::Invalid("lob_base_path is empty");
  }

  if (config.inline_threshold == 0) {
    return arrow::Status::Invalid("inline_threshold must be greater than 0");
  }

  if (config.max_lob_file_bytes == 0) {
    return arrow::Status::Invalid("max_lob_file_bytes must be greater than 0");
  }

  if (config.flush_threshold_bytes == 0) {
    return arrow::Status::Invalid("flush_threshold_bytes must be greater than 0");
  }

  if (config.flush_threshold_bytes > config.max_lob_file_bytes) {
    return arrow::Status::Invalid("flush_threshold_bytes should not exceed max_lob_file_bytes");
  }

  return std::make_unique<TextColumnManagerImpl>(std::move(fs), config);
}

}  // namespace milvus_storage::text_column
