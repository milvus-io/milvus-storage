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

#include "milvus-storage/text_column/text_column_writer.h"

#include <arrow/array/builder_binary.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include <numeric>

#include "milvus-storage/text_column/lob_reference.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/log.h"

#ifdef BUILD_VORTEX_BRIDGE
#include "milvus-storage/format/vortex/vortex_writer.h"
#endif

namespace milvus_storage::text_column {

#ifdef BUILD_VORTEX_BRIDGE

// implementation of TextColumnWriter using Vortex format
class TextColumnWriterImpl : public TextColumnWriter {
  public:
  TextColumnWriterImpl(std::shared_ptr<arrow::fs::FileSystem> fs, const TextColumnConfig& config)
      : fs_(std::move(fs)),
        config_(config),
        closed_(false),
        written_rows_(0),
        current_file_rows_(0),
        current_file_bytes_(0),
        pending_bytes_(0) {
    // generate binary UUID for the first LOB file
    GenerateUUIDBinary(current_file_id_);
  }

  ~TextColumnWriterImpl() override {
    if (!closed_) {
      // best effort abort, ignore errors in destructor
      (void)Abort();
    }
  }

  arrow::Result<std::vector<uint8_t>> WriteText(const std::string& text) override {
    if (closed_) {
      return arrow::Status::Invalid("writer is closed");
    }

    // check if text should be stored inline
    if (text.size() < config_.inline_threshold) {
      stats_.inline_texts++;
      stats_.total_texts++;
      stats_.total_bytes += text.size();
      written_rows_++;
      return EncodeInlineText(text);
    }

    // text is too large, store as LOB
    ARROW_RETURN_NOT_OK(EnsureVortexWriter());

    // get current row offset in the LOB file
    int32_t row_offset = static_cast<int32_t>(current_file_rows_);

    // add text to pending batch
    pending_texts_.push_back(text);
    pending_bytes_ += text.size();
    current_file_rows_++;

    // encode LOB reference
    auto ref = EncodeLOBReference(current_file_id_, row_offset);

    stats_.lob_texts++;
    stats_.total_texts++;
    stats_.total_bytes += text.size();
    written_rows_++;

    // flush if pending data exceeds threshold (will also roll file if needed)
    if (pending_bytes_ >= config_.flush_threshold_bytes) {
      ARROW_RETURN_NOT_OK(FlushPendingTexts());
    }

    return ref;
  }

  arrow::Result<std::vector<std::vector<uint8_t>>> WriteBatch(const std::vector<std::string>& texts) override {
    if (closed_) {
      return arrow::Status::Invalid("writer is closed");
    }

    std::vector<std::vector<uint8_t>> results;
    results.reserve(texts.size());

    for (const auto& text : texts) {
      if (text.size() < config_.inline_threshold) {
        // inline text
        results.push_back(EncodeInlineText(text));
        stats_.inline_texts++;
      } else {
        // LOB text
        ARROW_RETURN_NOT_OK(EnsureVortexWriter());

        int32_t row_offset = static_cast<int32_t>(current_file_rows_);
        pending_texts_.push_back(text);
        pending_bytes_ += text.size();
        current_file_rows_++;

        results.push_back(EncodeLOBReference(current_file_id_, row_offset));
        stats_.lob_texts++;

        // flush if pending data exceeds threshold (will also roll file if needed)
        if (pending_bytes_ >= config_.flush_threshold_bytes) {
          ARROW_RETURN_NOT_OK(FlushPendingTexts());
        }
      }

      stats_.total_texts++;
      stats_.total_bytes += text.size();
    }

    written_rows_ += texts.size();
    return results;
  }

  arrow::Result<std::shared_ptr<arrow::BinaryArray>> WriteArrowArray(
      const std::shared_ptr<arrow::StringArray>& texts) override {
    if (closed_) {
      return arrow::Status::Invalid("writer is closed");
    }

    arrow::BinaryBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(texts->length()));

    for (int64_t i = 0; i < texts->length(); i++) {
      if (texts->IsNull(i)) {
        // preserve null values - write null to output array
        ARROW_RETURN_NOT_OK(builder.AppendNull());
        stats_.total_texts++;
      } else {
        auto text = texts->GetString(i);
        ARROW_ASSIGN_OR_RAISE(auto ref, WriteText(text));
        ARROW_RETURN_NOT_OK(builder.Append(ref.data(), ref.size()));
      }
    }

    std::shared_ptr<arrow::BinaryArray> result;
    ARROW_RETURN_NOT_OK(builder.Finish(&result));
    return result;
  }

  arrow::Status Flush() override {
    if (closed_) {
      return arrow::Status::Invalid("writer is closed");
    }

    if (!pending_texts_.empty() && vortex_writer_) {
      ARROW_RETURN_NOT_OK(FlushPendingTexts());
    }

    return arrow::Status::OK();
  }

  arrow::Result<std::vector<LobFileResult>> Close() override {
    if (closed_) {
      return arrow::Status::Invalid("writer is already closed");
    }

    // flush any pending texts
    ARROW_RETURN_NOT_OK(Flush());

    // close current vortex writer if exists and update last file's metadata
    if (vortex_writer_) {
      ARROW_ASSIGN_OR_RAISE(auto cgfile, vortex_writer_->Close());
      (void)cgfile;  // we don't need the column group file info

      // update last file's metadata
      if (!created_files_.empty()) {
        auto& last_file = created_files_.back();
        last_file.total_rows = current_file_rows_;
        last_file.valid_rows = current_file_rows_;  // initially all rows are valid
        last_file.file_size_bytes = static_cast<int64_t>(current_file_bytes_);
      }

      vortex_writer_.reset();
    }

    closed_ = true;
    return created_files_;
  }

  arrow::Status Abort() override {
    if (closed_) {
      return arrow::Status::OK();
    }

    // discard pending texts
    pending_texts_.clear();

    // close vortex writer without committing
    if (vortex_writer_) {
      vortex_writer_.reset();
    }

    // delete created files (best effort, ignore errors)
    for (const auto& file_result : created_files_) {
      (void)fs_->DeleteFile(file_result.path);
    }

    created_files_.clear();
    closed_ = true;
    return arrow::Status::OK();
  }

  int64_t WrittenRows() const override { return written_rows_; }

  TextColumnWriterStats GetStats() const override { return stats_; }

  bool IsClosed() const override { return closed_; }

  private:
  // ensure vortex writer is initialized
  arrow::Status EnsureVortexWriter() {
    if (vortex_writer_) {
      return arrow::Status::OK();
    }

    // create schema for LOB file: single text column
    auto schema = arrow::schema({
        arrow::field("text_data", arrow::utf8(), false),
    });

    // build file path (convert binary UUID to string for path)
    auto file_path = BuildLOBFilePath(config_.lob_base_path, UUIDToString(current_file_id_));

    // ensure parent directory exists (only needed for local filesystem;
    // remote/S3 filesystems auto-create parent paths on put)
    if (IsLocalFileSystem(fs_)) {
      auto parent_dir = file_path.substr(0, file_path.rfind('/'));
      ARROW_RETURN_NOT_OK(fs_->CreateDir(parent_dir, true));
    }

    // create vortex writer
    vortex_writer_ = std::make_unique<vortex::VortexFileWriter>(fs_, schema, file_path, config_.properties);

    // add placeholder entry - will be updated with actual stats when file is closed
    created_files_.push_back(LobFileResult{file_path, 0, 0, 0});
    stats_.lob_files_created++;

    return arrow::Status::OK();
  }

  // flush pending texts to vortex file
  arrow::Status FlushPendingTexts() {
    if (pending_texts_.empty()) {
      return arrow::Status::OK();
    }

    // build string array from pending texts
    arrow::StringBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(pending_texts_.size()));
    ARROW_RETURN_NOT_OK(builder.ReserveData(pending_bytes_));

    for (const auto& text : pending_texts_) {
      ARROW_RETURN_NOT_OK(builder.Append(text));
    }

    std::shared_ptr<arrow::StringArray> text_array;
    ARROW_RETURN_NOT_OK(builder.Finish(&text_array));

    // create record batch
    auto schema = arrow::schema({
        arrow::field("text_data", arrow::utf8(), false),
    });
    auto batch = arrow::RecordBatch::Make(schema, text_array->length(), {text_array});

    // write to vortex
    ARROW_RETURN_NOT_OK(vortex_writer_->Write(batch));
    ARROW_RETURN_NOT_OK(vortex_writer_->Flush());

    // update file bytes and clear pending
    current_file_bytes_ += pending_bytes_;
    pending_texts_.clear();
    pending_bytes_ = 0;

    // check if we need to roll to a new file after flush
    if (current_file_bytes_ >= config_.max_lob_file_bytes) {
      ARROW_RETURN_NOT_OK(RollToNewFile());
    }

    return arrow::Status::OK();
  }

  // roll to a new LOB file
  arrow::Status RollToNewFile() {
    // flush and close current file
    ARROW_RETURN_NOT_OK(FlushPendingTexts());

    if (vortex_writer_) {
      ARROW_ASSIGN_OR_RAISE(auto cgfile, vortex_writer_->Close());
      (void)cgfile;

      // update current file's metadata before rolling
      if (!created_files_.empty()) {
        auto& current_file = created_files_.back();
        current_file.total_rows = current_file_rows_;
        current_file.valid_rows = current_file_rows_;  // initially all rows are valid
        current_file.file_size_bytes = static_cast<int64_t>(current_file_bytes_);
      }

      vortex_writer_.reset();
    }

    // generate new binary UUID and reset counters
    GenerateUUIDBinary(current_file_id_);
    current_file_rows_ = 0;
    current_file_bytes_ = 0;

    return arrow::Status::OK();
  }

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  TextColumnConfig config_;
  bool closed_;
  int64_t written_rows_;

  // current LOB file state
  uint8_t current_file_id_[UUID_BINARY_SIZE];  // binary UUID (16 bytes)
  int64_t current_file_rows_;
  size_t current_file_bytes_;  // bytes written to current file
  std::unique_ptr<vortex::VortexFileWriter> vortex_writer_;

  // pending texts to be written to LOB file
  std::vector<std::string> pending_texts_;
  size_t pending_bytes_;  // bytes in pending_texts_

  // created file results (for cleanup on abort and return on close)
  std::vector<LobFileResult> created_files_;

  // statistics
  TextColumnWriterStats stats_;
};

#else  // BUILD_VORTEX_BRIDGE

// stub implementation when Vortex is not available
class TextColumnWriterImpl : public TextColumnWriter {
  public:
  TextColumnWriterImpl(std::shared_ptr<arrow::fs::FileSystem> fs, const TextColumnConfig& config) {}

  arrow::Result<std::vector<uint8_t>> WriteText(const std::string& text) override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Result<std::vector<std::vector<uint8_t>>> WriteBatch(const std::vector<std::string>& texts) override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Result<std::shared_ptr<arrow::BinaryArray>> WriteArrowArray(
      const std::shared_ptr<arrow::StringArray>& texts) override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Status Flush() override { return arrow::Status::NotImplemented("Vortex support is not enabled"); }

  arrow::Result<std::vector<LobFileResult>> Close() override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Status Abort() override { return arrow::Status::NotImplemented("Vortex support is not enabled"); }

  int64_t WrittenRows() const override { return 0; }

  TextColumnWriterStats GetStats() const override { return {}; }

  bool IsClosed() const override { return true; }
};

#endif  // BUILD_VORTEX_BRIDGE

// factory function to create TextColumnWriter
arrow::Result<std::unique_ptr<TextColumnWriter>> CreateTextColumnWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                        const TextColumnConfig& config) {
  return std::make_unique<TextColumnWriterImpl>(std::move(fs), config);
}

}  // namespace milvus_storage::text_column
