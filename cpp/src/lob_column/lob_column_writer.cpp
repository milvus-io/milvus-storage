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

#include "milvus-storage/lob_column/lob_column_writer.h"

#include <arrow/array/builder_binary.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include <numeric>

#include "milvus-storage/lob_column/lob_reference.h"
#include "milvus-storage/filesystem/fs.h"

#ifdef BUILD_VORTEX_BRIDGE
#include "milvus-storage/format/vortex/vortex_writer.h"
#endif

namespace milvus_storage::lob_column {

#ifdef BUILD_VORTEX_BRIDGE

// implementation of LobColumnWriter using Vortex format
class LobColumnWriterImpl : public LobColumnWriter {
  public:
  LobColumnWriterImpl(std::shared_ptr<arrow::fs::FileSystem> fs, const LobColumnConfig& config)
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

  ~LobColumnWriterImpl() override {
    if (!closed_) {
      // best effort abort, ignore errors in destructor
      (void)Abort();
    }
  }

  arrow::Result<std::vector<uint8_t>> WriteData(const uint8_t* data, size_t data_size) override {
    if (closed_) {
      return arrow::Status::Invalid("writer is closed");
    }

    // check if data should be stored inline
    if (data_size < config_.inline_threshold) {
      stats_.inline_entries++;
      stats_.total_entries++;
      stats_.total_bytes += data_size;
      written_rows_++;
      return EncodeInlineData(data, data_size);
    }

    // data is too large, store as LOB
    ARROW_RETURN_NOT_OK(EnsureVortexWriter());

    // get current row offset in the LOB file
    int32_t row_offset = static_cast<int32_t>(current_file_rows_);

    // append directly to Arrow builder (single copy, no intermediate buffer)
    ARROW_RETURN_NOT_OK(pending_builder_.Append(data, static_cast<int32_t>(data_size)));
    pending_bytes_ += data_size;
    current_file_rows_++;

    // encode LOB reference
    auto ref = EncodeLOBReference(current_file_id_, row_offset);

    stats_.lob_entries++;
    stats_.total_entries++;
    stats_.total_bytes += data_size;
    written_rows_++;

    if (pending_bytes_ >= config_.flush_threshold_bytes) {
      ARROW_RETURN_NOT_OK(FlushPending());
      ARROW_RETURN_NOT_OK(MaybeRollFile());
    }

    return ref;
  }

  arrow::Result<std::vector<std::vector<uint8_t>>> WriteBatchData(
      const std::vector<std::pair<const uint8_t*, size_t>>& items) override {
    if (closed_) {
      return arrow::Status::Invalid("writer is closed");
    }

    std::vector<std::vector<uint8_t>> results;
    results.reserve(items.size());

    for (const auto& [data, data_size] : items) {
      if (data_size < config_.inline_threshold) {
        results.push_back(EncodeInlineData(data, data_size));
        stats_.inline_entries++;
      } else {
        ARROW_RETURN_NOT_OK(EnsureVortexWriter());

        int32_t row_offset = static_cast<int32_t>(current_file_rows_);
        ARROW_RETURN_NOT_OK(pending_builder_.Append(data, static_cast<int32_t>(data_size)));
        pending_bytes_ += data_size;
        current_file_rows_++;

        results.push_back(EncodeLOBReference(current_file_id_, row_offset));
        stats_.lob_entries++;

        if (pending_bytes_ >= config_.flush_threshold_bytes) {
          ARROW_RETURN_NOT_OK(FlushPending());
          ARROW_RETURN_NOT_OK(MaybeRollFile());
        }
      }

      stats_.total_entries++;
      stats_.total_bytes += data_size;
    }

    written_rows_ += items.size();
    return results;
  }

  arrow::Result<std::shared_ptr<arrow::BinaryArray>> WriteArrowArray(
      const std::shared_ptr<arrow::BinaryArray>& data_array) override {
    if (closed_) {
      return arrow::Status::Invalid("writer is closed");
    }

    arrow::BinaryBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(data_array->length()));

    for (int64_t i = 0; i < data_array->length(); i++) {
      if (data_array->IsNull(i)) {
        ARROW_RETURN_NOT_OK(builder.AppendNull());
        stats_.total_entries++;
        written_rows_++;
      } else {
        int32_t length;
        const uint8_t* value = data_array->GetValue(i, &length);
        ARROW_ASSIGN_OR_RAISE(auto ref, WriteData(value, static_cast<size_t>(length)));
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

    ARROW_RETURN_NOT_OK(FlushPending());
    ARROW_RETURN_NOT_OK(MaybeRollFile());
    return arrow::Status::OK();
  }

  arrow::Result<std::vector<LobFileResult>> Close() override {
    if (closed_) {
      return arrow::Status::Invalid("writer is already closed");
    }

    ARROW_RETURN_NOT_OK(FlushPending());
    ARROW_RETURN_NOT_OK(CloseCurrentFile());

    closed_ = true;
    return created_files_;
  }

  arrow::Status Abort() override {
    if (closed_) {
      return arrow::Status::OK();
    }

    // discard pending data
    pending_builder_.Reset();

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

  LobColumnWriterStats GetStats() const override { return stats_; }

  bool IsClosed() const override { return closed_; }

  private:
  std::shared_ptr<arrow::DataType> ArrowType() const {
    return config_.data_type == LobDataType::kText ? arrow::utf8() : arrow::binary();
  }

  std::string FieldName() const {
    return config_.data_type == LobDataType::kText ? "text_data" : "binary_data";
  }

  // ensure vortex writer is initialized
  arrow::Status EnsureVortexWriter() {
    if (vortex_writer_) {
      return arrow::Status::OK();
    }

    auto schema = arrow::schema({
        arrow::field(FieldName(), ArrowType(), false),
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

  // flush pending data to the current vortex file
  arrow::Status FlushPending() {
    if (pending_builder_.length() == 0) {
      return arrow::Status::OK();
    }

    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(pending_builder_.Finish(&array));

    auto schema = arrow::schema({
        arrow::field(FieldName(), ArrowType(), false),
    });
    auto batch = arrow::RecordBatch::Make(schema, array->length(), {array});

    ARROW_RETURN_NOT_OK(vortex_writer_->Write(batch));
    ARROW_RETURN_NOT_OK(vortex_writer_->Flush());

    current_file_bytes_ += pending_bytes_;
    pending_bytes_ = 0;

    return arrow::Status::OK();
  }

  // close the current vortex file and finalize its metadata
  arrow::Status CloseCurrentFile() {
    if (!vortex_writer_) {
      return arrow::Status::OK();
    }

    ARROW_ASSIGN_OR_RAISE(auto cgfile, vortex_writer_->Close());
    (void)cgfile;

    if (!created_files_.empty()) {
      auto& last_file = created_files_.back();
      last_file.total_rows = current_file_rows_;
      last_file.valid_rows = current_file_rows_;
      last_file.file_size_bytes = static_cast<int64_t>(current_file_bytes_);
    }

    vortex_writer_.reset();
    return arrow::Status::OK();
  }

  // if the current file exceeds max size, close it and prepare for a new one
  arrow::Status MaybeRollFile() {
    if (current_file_bytes_ < config_.max_lob_file_bytes) {
      return arrow::Status::OK();
    }

    ARROW_RETURN_NOT_OK(CloseCurrentFile());

    GenerateUUIDBinary(current_file_id_);
    current_file_rows_ = 0;
    current_file_bytes_ = 0;

    return arrow::Status::OK();
  }

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  LobColumnConfig config_;
  bool closed_;
  int64_t written_rows_;

  // current LOB file state
  uint8_t current_file_id_[UUID_BINARY_SIZE];  // binary UUID (16 bytes)
  int64_t current_file_rows_;
  size_t current_file_bytes_;  // bytes written to current file
  std::unique_ptr<vortex::VortexFileWriter> vortex_writer_;

  // pending data accumulates directly in Arrow builder to avoid double-copy.
  // BinaryBuilder works for both text (utf8 is binary-compatible) and binary data.
  arrow::BinaryBuilder pending_builder_;
  size_t pending_bytes_;

  // created file results (for cleanup on abort and return on close)
  std::vector<LobFileResult> created_files_;

  // statistics
  LobColumnWriterStats stats_;
};

#else  // BUILD_VORTEX_BRIDGE

// stub implementation when Vortex is not available
class LobColumnWriterImpl : public LobColumnWriter {
  public:
  LobColumnWriterImpl(std::shared_ptr<arrow::fs::FileSystem> fs, const LobColumnConfig& config) {}

  arrow::Result<std::vector<uint8_t>> WriteData(const uint8_t* data, size_t data_size) override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Result<std::vector<std::vector<uint8_t>>> WriteBatchData(
      const std::vector<std::pair<const uint8_t*, size_t>>& items) override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Result<std::shared_ptr<arrow::BinaryArray>> WriteArrowArray(
      const std::shared_ptr<arrow::BinaryArray>& data) override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Status Flush() override { return arrow::Status::NotImplemented("Vortex support is not enabled"); }

  arrow::Result<std::vector<LobFileResult>> Close() override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Status Abort() override { return arrow::Status::NotImplemented("Vortex support is not enabled"); }

  int64_t WrittenRows() const override { return 0; }

  LobColumnWriterStats GetStats() const override { return {}; }

  bool IsClosed() const override { return true; }
};

#endif  // BUILD_VORTEX_BRIDGE

// factory function to create LobColumnWriter
arrow::Result<std::unique_ptr<LobColumnWriter>> CreateLobColumnWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                      const LobColumnConfig& config) {
  return std::make_unique<LobColumnWriterImpl>(std::move(fs), config);
}

}  // namespace milvus_storage::lob_column
