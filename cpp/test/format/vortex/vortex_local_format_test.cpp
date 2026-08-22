// Copyright 2025 Zilliz
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
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/concatenate.h>
#include <arrow/buffer.h>
#include <arrow/c/bridge.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/interfaces.h>
#include <arrow/record_batch.h>
#include <arrow/util/io_util.h>

#include <boost/filesystem/operations.hpp>

#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/vortex/vortex_footer_internal.h"
#include "milvus-storage/format/vortex/vortex_field_layout_internal.h"
#include "milvus-storage/format/vortex/vortex_footer_reader.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/format/vortex/vortex_planner.h"
#include "milvus-storage/format/vortex/vortex_translater.h"
#include "milvus-storage/format/vortex/vortex_types.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "test_env.h"

namespace milvus_storage {

using namespace vortex;

namespace {

class InMemoryVortexRangeFile : public VortexRangeFile, public arrow::io::RandomAccessFile {
  public:
  void Resize(uint64_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);
    data_.resize(size);
  }

  uint64_t Size() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_.size();
  }

  arrow::Status WriteAt(const uint64_t& offset, const std::shared_ptr<arrow::Buffer>& data) override {
    if (!data) {
      return arrow::Status::Invalid("InMemoryVortexRangeFile::WriteAt requires non-null data");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto end = offset + static_cast<uint64_t>(data->size());
    if (end < offset) {
      return arrow::Status::Invalid("InMemoryVortexRangeFile::WriteAt offset overflow");
    }
    if (end > data_.size()) {
      data_.resize(end);
    }
    std::memcpy(data_.data() + offset, data->data(), data->size());
    write_ranges_.push_back(ByteRange{offset, static_cast<uint64_t>(data->size())});
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) const override {
    if (position < 0 || nbytes < 0) {
      return arrow::Status::Invalid("InMemoryVortexRangeFile::ReadAt requires non-negative position and size");
    }
    if (nbytes == 0) {
      return int64_t{0};
    }
    if (out == nullptr) {
      return arrow::Status::Invalid("InMemoryVortexRangeFile::ReadAt requires non-null output");
    }

    std::memset(out, 0, static_cast<size_t>(nbytes));
    std::lock_guard<std::mutex> lock(mutex_);
    const auto offset = static_cast<uint64_t>(position);
    if (offset >= data_.size()) {
      return nbytes;
    }
    const auto available = std::min<uint64_t>(static_cast<uint64_t>(nbytes), data_.size() - offset);
    std::memcpy(out, data_.data() + offset, available);
    return nbytes;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) const override {
    if (position < 0 || nbytes < 0) {
      return arrow::Status::Invalid("InMemoryVortexRangeFile::ReadAt requires non-negative position and size");
    }
    ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateResizableBuffer(nbytes));
    ARROW_RETURN_NOT_OK(ReadAt(position, nbytes, buffer->mutable_data()).status());
    return std::shared_ptr<arrow::Buffer>(std::move(buffer));
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    ARROW_ASSIGN_OR_RAISE(auto bytes_read, ReadAt(position_, nbytes, out));
    position_ += bytes_read;
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(position_, nbytes));
    position_ += nbytes;
    return buffer;
  }

  arrow::Status Close() override {
    closed_ = true;
    return arrow::Status::OK();
  }

  bool closed() const override { return closed_; }

  arrow::Result<int64_t> Tell() const override { return position_; }

  arrow::Status Seek(int64_t position) override {
    if (position < 0) {
      return arrow::Status::Invalid("InMemoryVortexRangeFile::Seek requires non-negative position");
    }
    position_ = position;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> GetSize() override { return static_cast<int64_t>(Size()); }

  void Punch(uint64_t offset, uint64_t length) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (offset >= data_.size() || length == 0) {
      return;
    }
    const auto available = std::min<uint64_t>(length, data_.size() - offset);
    std::memset(data_.data() + offset, 0, available);
  }

  std::vector<ByteRange> WriteRanges() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return write_ranges_;
  }

  private:
  mutable std::mutex mutex_;
  std::vector<uint8_t> data_;
  std::vector<ByteRange> write_ranges_;
  int64_t position_ = 0;
  bool closed_ = false;
};

class InMemoryVortexRangeFileSystem : public arrow::fs::LocalFileSystem, public VortexRangeFileProvider {
  public:
  arrow::Result<std::shared_ptr<VortexRangeFile>> GetVortexRangeFile(const std::string& path) const override {
    return GetOrCreateFile(path);
  }

  arrow::Result<std::shared_ptr<InMemoryVortexRangeFile>> GetInMemoryFile(const std::string& path) const {
    return GetOrCreateFile(path);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    ARROW_ASSIGN_OR_RAISE(auto file, GetOrCreateFile(path));
    return std::static_pointer_cast<arrow::io::RandomAccessFile>(file);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    return OpenInputFile(info.path());
  }

  private:
  arrow::Result<std::shared_ptr<InMemoryVortexRangeFile>> GetOrCreateFile(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& file = files_[path];
    if (!file) {
      file = std::make_shared<InMemoryVortexRangeFile>();
    }
    return file;
  }

  mutable std::mutex mutex_;
  mutable std::unordered_map<std::string, std::shared_ptr<InMemoryVortexRangeFile>> files_;
};

void AssertLoadableEmptyCellMetas(const VortexCellMetasPtr& cell_metas) {
  ASSERT_NE(cell_metas, nullptr);
  for (const auto& meta : *cell_metas) {
    ASSERT_EQ(meta.row_count, 0);
    ASSERT_TRUE(meta.flat_segment_ids.empty());
    ASSERT_TRUE(meta.flat_segment_ranges.empty());
    ASSERT_EQ(meta.memory_bytes, 0);
    ASSERT_EQ(meta.storage_bytes, 0);
  }
}

}  // namespace

class VortexLocalFormatTest : public ::testing::Test {
  protected:
  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    api::SetValue(properties_, PROPERTY_WRITER_VORTEX_FORMAT_VERSION, "2");

    ASSERT_AND_ASSIGN(schema_, CreateTestSchema(needed_columns_));
    for (int64_t batch_idx = 0; batch_idx < batch_count_; ++batch_idx) {
      ASSERT_AND_ASSIGN(auto rb, CreateTestData(schema_, batch_idx * rows_per_batch_, false, rows_per_batch_, 4, 50,
                                                needed_columns_));
      record_batches_.emplace_back(std::move(rb));
    }

    ASSERT_AND_ASSIGN(file_system_, GetFileSystem(properties_));
  }

  void TearDown() override {
    auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
    if (storage_type == "local" || storage_type.empty()) {
      boost::filesystem::remove_all(test_file_name_);
    }
  }

  [[nodiscard]] int64_t recordBatchsRows() const { return batch_count_ * rows_per_batch_; }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> ChunkedArrayToRecordBatch(
      const std::shared_ptr<arrow::ChunkedArray>& chunkedarray) {
    auto chunk_size = chunkedarray->num_chunks();
    if (chunk_size == 1) {
      return arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0));
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
    for (int i = 0; i < chunk_size; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(i)));
      rbs.emplace_back(rb);
    }

    return arrow::ConcatenateRecordBatches(rbs);
  }

  const std::vector<std::string>& data_columns() const { return data_columns_; }

  arrow::Result<api::ColumnGroupFile> WriteVortexFile() {
    ARROW_ASSIGN_OR_RAISE(auto vx_writer,
                          vortex::VortexFileWriter::Open(file_system_, schema_, test_file_name_, properties_));
    for (const auto& rb : record_batches_) {
      ARROW_RETURN_NOT_OK(vx_writer->Write(rb));
    }
    ARROW_RETURN_NOT_OK(vx_writer->Flush());
    return vx_writer->Close();
  }

  arrow::Result<api::ColumnGroupFile> WriteEmptyVortexFile() {
    ARROW_ASSIGN_OR_RAISE(auto vx_writer,
                          vortex::VortexFileWriter::Open(file_system_, schema_, test_file_name_, properties_));
    ARROW_ASSIGN_OR_RAISE(auto empty_rb, CreateTestData(schema_, 0, false, 0, 4, 50, needed_columns_));
    ARROW_RETURN_NOT_OK(vx_writer->Write(empty_rb));
    ARROW_RETURN_NOT_OK(vx_writer->Flush());
    return vx_writer->Close();
  }

  std::shared_ptr<VortexFooterReader> MakeFooterReader(const api::ColumnGroupFile& cgfile,
                                                       const std::shared_ptr<arrow::fs::FileSystem>& sparse_fs) const {
    return std::make_shared<VortexFooterReader>(sparse_fs, "test-file.vx.sparse", test_file_name_,
                                                cgfile.Get<uint64_t>(api::kPropertyFileSize),
                                                cgfile.Get<uint64_t>(api::kPropertyFooterSize));
  }

  protected:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::fs::FileSystem> file_system_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_;
  const char* test_file_name_ = "test-file.vx";
  api::Properties properties_;

  private:
  const std::array<bool, 4> needed_columns_ = {true, true, true, false};
  const std::vector<std::string> data_columns_ = {"id", "name", "value"};
  const int64_t rows_per_batch_ = 8192;
  const int64_t batch_count_ = 4;
};

TEST_F(VortexLocalFormatTest, TestFooterReaderOpensZeroRowVortexFileWithoutFooterFallback) {
  ASSERT_AND_ASSIGN(auto cgfile, WriteEmptyVortexFile());
  ASSERT_EQ(0, cgfile.end_index);
  ASSERT_GT(cgfile.Get<uint64_t>(api::kPropertyFileSize), 0);
  const auto actual_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  ASSERT_GT(actual_footer_size, 1);

  for (const uint64_t supplied_footer_size : {actual_footer_size - 1, uint64_t{1}}) {
    auto understated_footer_file = cgfile;
    understated_footer_file.Set(api::kPropertyFooterSize, supplied_footer_size);
    auto sparse_fs = std::make_shared<InMemoryVortexRangeFileSystem>();
    auto reader = MakeFooterReader(understated_footer_file, sparse_fs);

    const auto status = reader->Open(file_system_);
    ASSERT_FALSE(status.ok()) << "understated footer size " << supplied_footer_size;
    EXPECT_FALSE(reader->opened());
    EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageError);
    ASSERT_AND_ASSIGN(auto sparse_file, sparse_fs->GetInMemoryFile("test-file.vx.sparse"));
    EXPECT_EQ(sparse_file->WriteRanges().size(), 1u) << "footer failure must not trigger another range read";
  }

  auto footer_reader = MakeFooterReader(cgfile, std::make_shared<InMemoryVortexRangeFileSystem>());
  ASSERT_STATUS_OK(footer_reader->Open(file_system_));
  ASSERT_TRUE(footer_reader->opened());
  ASSERT_EQ(footer_reader->rows(), 0);
  ASSERT_EQ(footer_reader->footer_size(), cgfile.Get<uint64_t>(api::kPropertyFooterSize));
  ASSERT_NE(footer_reader->file_schema(), nullptr);

  ASSERT_AND_ASSIGN(auto cell_metas, BuildVortexCellMetas(footer_reader, "id"));
  AssertLoadableEmptyCellMetas(cell_metas);
  ASSERT_AND_ASSIGN(auto group_cell_metas, BuildVortexGroupCellMetas(footer_reader, data_columns()));
  AssertLoadableEmptyCellMetas(group_cell_metas);
}

TEST_F(VortexLocalFormatTest, TestFooterReaderOpenAfterWriterCloseWithoutWriteIfFileExists) {
  ASSERT_AND_ASSIGN(auto vx_writer,
                    vortex::VortexFileWriter::Open(file_system_, schema_, test_file_name_, properties_));
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer->Close());
  ASSERT_EQ(0, cgfile.end_index);

  ASSERT_AND_ASSIGN(auto file_info, file_system_->GetFileInfo(test_file_name_));
  ASSERT_TRUE(file_info.IsFile());
  ASSERT_GT(file_info.size(), 0);

  auto footer_reader = MakeFooterReader(cgfile, std::make_shared<InMemoryVortexRangeFileSystem>());
  ASSERT_STATUS_OK(footer_reader->Open(file_system_));
  ASSERT_TRUE(footer_reader->opened());
  ASSERT_EQ(footer_reader->rows(), 0);
  ASSERT_NE(footer_reader->file_schema(), nullptr);

  ASSERT_AND_ASSIGN(auto cell_metas, BuildVortexCellMetas(footer_reader, "id"));
  AssertLoadableEmptyCellMetas(cell_metas);
  ASSERT_AND_ASSIGN(auto group_cell_metas, BuildVortexGroupCellMetas(footer_reader, data_columns()));
  AssertLoadableEmptyCellMetas(group_cell_metas);
}

TEST_F(VortexLocalFormatTest, TestFooterReaderMissingFilePreservesEnoent) {
  constexpr const char* kMissingPath = "missing-vortex-file-for-enoent-test.vx";
  boost::filesystem::remove(kMissingPath);
  auto footer_reader = std::make_shared<VortexFooterReader>(std::make_shared<InMemoryVortexRangeFileSystem>(),
                                                            "missing-file.vx.sparse", kMissingPath);

  auto status = footer_reader->Open(std::make_shared<arrow::fs::LocalFileSystem>());

  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(status.IsIOError());
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(status), ENOENT);
}

TEST_F(VortexLocalFormatTest, TestFooterReaderMissingFieldIsCallerErrorNotDataFormat) {
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());
  auto footer_reader = MakeFooterReader(cgfile, std::make_shared<InMemoryVortexRangeFileSystem>());
  ASSERT_STATUS_OK(footer_reader->Open(file_system_));

  auto result = footer_reader->GetFieldLayout("missing-field");

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsKeyError()) << result.status().ToString();
  auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
  EXPECT_TRUE(detail == nullptr || detail->code() != ExtendStatusCode::VortexDataFormat) << result.status().ToString();
}

TEST_F(VortexLocalFormatTest, TestFooterReaderDoesNotPrefetchHeaderRangeWhenFooterSizeKnown) {
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());
  const auto file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  const auto footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  const auto tail_read_size = footer_size + VortexEofSize();
  ASSERT_LT(tail_read_size, file_size);

  constexpr const char* kSparsePath = "test-file.vx.sparse";
  auto sparse_fs = std::make_shared<InMemoryVortexRangeFileSystem>();
  auto footer_reader = MakeFooterReader(cgfile, sparse_fs);
  ASSERT_STATUS_OK(footer_reader->Open(file_system_, false));

  ASSERT_AND_ASSIGN(auto sparse_file, sparse_fs->GetInMemoryFile(kSparsePath));
  const auto write_ranges = sparse_file->WriteRanges();
  ASSERT_FALSE(write_ranges.empty());

  const auto tail_offset = file_size - tail_read_size;
  bool saw_footer_tail = false;
  for (const auto& range : write_ranges) {
    EXPECT_NE(range.offset, 0) << "known footer_size should not trigger a separate header prefetch";
    saw_footer_tail = saw_footer_tail || (range.offset == tail_offset && range.length == tail_read_size);
  }
  ASSERT_TRUE(saw_footer_tail);
}

TEST_F(VortexLocalFormatTest, TestFooterReaderOptionalZoneMapLoadControlsPruning) {
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "true");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());

  auto fs_holder = std::make_shared<FileSystemWrapper>(file_system_);
  ASSERT_AND_ASSIGN(auto vxfile, VortexFile::Open(reinterpret_cast<uint8_t*>(fs_holder.get()), test_file_name_,
                                                  cgfile.Get<uint64_t>(api::kPropertyFileSize),
                                                  cgfile.Get<uint64_t>(api::kPropertyFooterSize)));
  ASSERT_EQ(vxfile.RootLayoutEncoding(), "milvus.v2_zoned_row_group");
  ASSERT_AND_ASSIGN(auto row_group_zonemap_count, vxfile.RowGroupZoneMapCount());
  ASSERT_GT(row_group_zonemap_count, 1u);

  std::vector<uint64_t> candidate_row_group_ids;
  candidate_row_group_ids.reserve(row_group_zonemap_count);
  for (uint64_t row_group_id = 0; row_group_id < row_group_zonemap_count; ++row_group_id) {
    candidate_row_group_ids.emplace_back(row_group_id);
  }

  auto no_zonemap_footer_reader = MakeFooterReader(cgfile, std::make_shared<InMemoryVortexRangeFileSystem>());
  ASSERT_STATUS_OK(no_zonemap_footer_reader->Open(file_system_, false));
  ASSERT_AND_ASSIGN(auto unpruned_row_groups,
                    no_zonemap_footer_reader->PruneRowGroups("id >= 1000000", candidate_row_group_ids));
  ASSERT_EQ(unpruned_row_groups, candidate_row_group_ids);
  ASSERT_STATUS_NOT_OK(no_zonemap_footer_reader->Open(file_system_));

  auto zonemap_footer_reader = MakeFooterReader(cgfile, std::make_shared<InMemoryVortexRangeFileSystem>());
  ASSERT_STATUS_OK(zonemap_footer_reader->Open(file_system_));
  ASSERT_AND_ASSIGN(auto pruned_row_groups,
                    zonemap_footer_reader->PruneRowGroups("id >= 1000000", candidate_row_group_ids));
  ASSERT_TRUE(pruned_row_groups.empty());
}

TEST_F(VortexLocalFormatTest, TestPlannerBuildsRangeAndTakePlans) {
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "true");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());

  auto footer_reader = MakeFooterReader(cgfile, std::make_shared<InMemoryVortexRangeFileSystem>());
  ASSERT_STATUS_OK(footer_reader->Open(file_system_));
  ASSERT_AND_ASSIGN(auto cell_metas, BuildVortexCellMetas(footer_reader, "id"));
  ASSERT_AND_ASSIGN(auto planner, VortexPlanner::Make(footer_reader, "id", cell_metas));
  ASSERT_GT(planner->num_cells(), 1u);

  const auto& first_cell = (*cell_metas)[0];
  ASSERT_GT(first_cell.row_count, 2u);

  const auto row_start = first_cell.row_offset + 1;
  const auto row_end = first_cell.row_offset + first_cell.row_count - 1;
  ASSERT_AND_ASSIGN(auto range_plan, planner->PlanForRowRange(row_start, row_end));
  ASSERT_EQ(range_plan.cell_ids, (std::vector<uint64_t>{first_cell.cell_id}));
  auto* range_scan = std::get_if<VortexReadPlan::RangeScan>(&range_plan.read_plan.op);
  ASSERT_NE(range_scan, nullptr);
  ASSERT_EQ(range_scan->ranges.size(), 1u);
  ASSERT_EQ(range_scan->ranges[0].start, row_start);
  ASSERT_EQ(range_scan->ranges[0].end, row_end);
  ASSERT_TRUE(range_plan.read_plan.apply_predicate);

  std::vector<int64_t> offsets{static_cast<int64_t>((*cell_metas)[0].row_offset),
                               static_cast<int64_t>((*cell_metas)[1].row_offset)};
  ASSERT_AND_ASSIGN(auto take_plan, planner->PlanForOffsets(offsets));
  ASSERT_EQ(take_plan.cell_ids, (std::vector<uint64_t>{(*cell_metas)[0].cell_id, (*cell_metas)[1].cell_id}));
  auto* take = std::get_if<VortexReadPlan::Take>(&take_plan.read_plan.op);
  ASSERT_NE(take, nullptr);
  ASSERT_EQ(take->row_indices, offsets);
  ASSERT_EQ(take->ranges.size(), 2u);
  ASSERT_FALSE(take_plan.read_plan.apply_predicate);
}

TEST_F(VortexLocalFormatTest, TestPlannerRejectsInvalidTakeOffsets) {
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "true");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());

  auto footer_reader = MakeFooterReader(cgfile, std::make_shared<InMemoryVortexRangeFileSystem>());
  ASSERT_STATUS_OK(footer_reader->Open(file_system_));
  ASSERT_AND_ASSIGN(auto cell_metas, BuildVortexCellMetas(footer_reader, "id"));
  ASSERT_AND_ASSIGN(auto planner, VortexPlanner::Make(footer_reader, "id", std::move(cell_metas)));

  auto duplicate_offsets = planner->PlanForOffsets(std::vector<int64_t>{1, 1});
  ASSERT_STATUS_NOT_OK(duplicate_offsets.status());
  auto unsorted_offsets = planner->PlanForOffsets(std::vector<int64_t>{2, 1});
  ASSERT_STATUS_NOT_OK(unsorted_offsets.status());
  auto negative_offset = planner->PlanForOffsets(std::vector<int64_t>{-1});
  ASSERT_STATUS_NOT_OK(negative_offset.status());
  auto out_of_range_offset = planner->PlanForOffsets(std::vector<int64_t>{recordBatchsRows()});
  ASSERT_STATUS_NOT_OK(out_of_range_offset.status());
}

TEST_F(VortexLocalFormatTest, TestPlannerPredicatePruningFollowsFooterReaderZoneMapState) {
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "true");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());

  auto no_zonemap_footer_reader = MakeFooterReader(cgfile, std::make_shared<InMemoryVortexRangeFileSystem>());
  ASSERT_STATUS_OK(no_zonemap_footer_reader->Open(file_system_, false));
  ASSERT_AND_ASSIGN(auto no_zonemap_cell_metas, BuildVortexCellMetas(no_zonemap_footer_reader, "id"));
  ASSERT_AND_ASSIGN(auto no_zonemap_planner,
                    VortexPlanner::Make(no_zonemap_footer_reader, "id", std::move(no_zonemap_cell_metas)));
  ASSERT_GT(no_zonemap_planner->num_cells(), 1u);

  ASSERT_AND_ASSIGN(auto unpruned_plan, no_zonemap_planner->PlanForRowRange(0, recordBatchsRows(), "id >= 1000000"));
  ASSERT_EQ(unpruned_plan.cell_ids.size(), no_zonemap_planner->num_cells());

  auto zonemap_footer_reader = MakeFooterReader(cgfile, std::make_shared<InMemoryVortexRangeFileSystem>());
  ASSERT_STATUS_OK(zonemap_footer_reader->Open(file_system_));
  ASSERT_AND_ASSIGN(auto zonemap_cell_metas, BuildVortexCellMetas(zonemap_footer_reader, "id"));
  ASSERT_AND_ASSIGN(auto zonemap_planner,
                    VortexPlanner::Make(zonemap_footer_reader, "id", std::move(zonemap_cell_metas)));
  ASSERT_EQ(zonemap_planner->num_cells(), no_zonemap_planner->num_cells());

  ASSERT_AND_ASSIGN(auto pruned_plan, zonemap_planner->PlanForRowRange(0, recordBatchsRows(), "id >= 1000000"));
  ASSERT_TRUE(pruned_plan.cell_ids.empty());
}

TEST_F(VortexLocalFormatTest, TestReadByPlanAppliesPredicate) {
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "true");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns(),
                                              cgfile.Get<uint64_t>(api::kPropertyFileSize),
                                              cgfile.Get<uint64_t>(api::kPropertyFooterSize));
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto array_stream, vx_reader.read_with_plan(VortexReadPlan{
                                           .op =
                                               VortexReadPlan::RangeScan{
                                                   .ranges = {RowRange{.start = 1000, .end = 1500}},
                                               },
                                           .predicate = "id >= 1200 AND id < 1300",
                                           .apply_predicate = true,
                                       }));
  ASSERT_AND_ASSIGN(auto chunked_array, arrow::ImportChunkedArray(&array_stream));
  ASSERT_AND_ASSIGN(auto rb, ChunkedArrayToRecordBatch(chunked_array));
  ASSERT_EQ(rb->num_rows(), 100);

  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
  for (int i = 0; i < id_array->length(); ++i) {
    ASSERT_EQ(id_array->Value(i), static_cast<int64_t>(1200 + i));
  }
}

TEST_F(VortexLocalFormatTest, TestReadByPlanEmptyRangeReturnsEmptyStream) {
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns(),
                                              cgfile.Get<uint64_t>(api::kPropertyFileSize),
                                              cgfile.Get<uint64_t>(api::kPropertyFooterSize));
  ASSERT_STATUS_OK(vx_reader.open());

  ASSERT_AND_ASSIGN(auto array_stream, vx_reader.read_with_plan(VortexReadPlan{
                                           .op =
                                               VortexReadPlan::RangeScan{
                                                   .ranges = {RowRange{.start = 1000, .end = 1000}},
                                               },
                                           .apply_predicate = false,
                                       }));
  ASSERT_AND_ASSIGN(auto chunked_array, arrow::ImportChunkedArray(&array_stream));
  ASSERT_EQ(chunked_array->length(), 0);
}

TEST_F(VortexLocalFormatTest, PublicArgumentValidationReturnsPlainInvalid) {
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());

  auto expect_plain_invalid = [](const arrow::Status& status) {
    EXPECT_TRUE(status.IsInvalid()) << status.ToString();
    EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr) << status.ToString();
  };

  auto incomplete_metadata =
      vortex::VortexFormatReader::MetaTrait::create_from_metadata(nullptr, cgfile, schema_, data_columns(), "");
  ASSERT_FALSE(incomplete_metadata.ok());
  expect_plain_invalid(incomplete_metadata.status());

  auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns(),
                                              cgfile.Get<uint64_t>(api::kPropertyFileSize),
                                              cgfile.Get<uint64_t>(api::kPropertyFooterSize));
  ASSERT_STATUS_OK(vx_reader.open());

  auto unsorted_ranges = vx_reader.read_with_plan(VortexReadPlan{
      .op =
          VortexReadPlan::RangeScan{
              .ranges = {RowRange{.start = 100, .end = 200}, RowRange{.start = 50, .end = 75}},
          },
      .apply_predicate = false,
  });
  ASSERT_FALSE(unsorted_ranges.ok());
  expect_plain_invalid(unsorted_ranges.status());

  auto column_sizes = vx_reader.get_rg_column_memsz(-1);
  ASSERT_FALSE(column_sizes.ok());
  expect_plain_invalid(column_sizes.status());

  auto chunk = vx_reader.get_chunk(-1);
  ASSERT_FALSE(chunk.ok());
  expect_plain_invalid(chunk.status());

  auto chunks = vx_reader.get_chunks({0, -1});
  ASSERT_FALSE(chunks.ok());
  expect_plain_invalid(chunks.status());
}

TEST_F(VortexLocalFormatTest, TestTranslaterLoadsAndReleasesCellRanges) {
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "true");
  api::SetValue(properties_, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());

  auto sparse_fs = std::make_shared<InMemoryVortexRangeFileSystem>();
  auto footer_reader = MakeFooterReader(cgfile, sparse_fs);
  ASSERT_STATUS_OK(footer_reader->Open(file_system_));
  ASSERT_AND_ASSIGN(auto cell_metas, BuildVortexCellMetas(footer_reader, "id"));
  ASSERT_GT(cell_metas->size(), 1u);

  constexpr const char* kSparsePath = "test-file.vx.sparse";
  ASSERT_AND_ASSIGN(auto translater,
                    VortexTranslater::Make(cell_metas, file_system_, test_file_name_, sparse_fs, kSparsePath));
  ASSERT_EQ(translater->num_cells(), cell_metas->size());
  ASSERT_EQ(translater->key(), test_file_name_);
  ASSERT_EQ(translater->cell_id_of(1), 1);
  ASSERT_EQ(translater->cells_storage_bytes({0, 1}),
            static_cast<int64_t>((*cell_metas)[0].storage_bytes + (*cell_metas)[1].storage_bytes));

  uint64_t expected_pinned_bytes = 0;
  for (const auto& range : MergeByteRanges((*cell_metas)[0].flat_segment_ranges)) {
    expected_pinned_bytes += range.length;
  }
  auto [loaded, loading_overhead] = translater->estimated_byte_size_of_cell(0);
  ASSERT_GT(loaded.memory_bytes, 0);
  ASSERT_EQ(loaded.memory_bytes, loading_overhead.memory_bytes);

  ASSERT_AND_ASSIGN(auto sparse_file, sparse_fs->GetInMemoryFile(kSparsePath));
  ASSERT_EQ(sparse_file->Size(), cgfile.Get<uint64_t>(api::kPropertyFileSize));

  {
    auto cells = translater->get_cells(nullptr, {0});
    ASSERT_EQ(cells.size(), 1u);
    ASSERT_EQ(cells[0].first, 0);
    ASSERT_EQ(cells[0].second->meta().cell_id, 0u);
    ASSERT_EQ(cells[0].second->pinned_bytes(),
              std::max<uint64_t>((*cell_metas)[0].storage_bytes, expected_pinned_bytes));
    ASSERT_EQ(cells[0].second->CellByteSize().memory_bytes, static_cast<int64_t>(cells[0].second->pinned_bytes()));

    const auto& loaded_range = (*cell_metas)[0].flat_segment_ranges[0];
    const auto bytes_to_check = std::min<uint64_t>(loaded_range.length, 64);
    ASSERT_AND_ASSIGN(auto source_file, file_system_->OpenInputFile(test_file_name_));
    ASSERT_AND_ASSIGN(auto source_data, source_file->ReadAt(static_cast<int64_t>(loaded_range.offset), bytes_to_check));
    ASSERT_AND_ASSIGN(auto sparse_data, sparse_file->ReadAt(static_cast<int64_t>(loaded_range.offset), bytes_to_check));
    ASSERT_EQ(std::memcmp(source_data->data(), sparse_data->data(), bytes_to_check), 0);
  }

  ASSERT_AND_ASSIGN(auto punched_data,
                    sparse_file->ReadAt(static_cast<int64_t>((*cell_metas)[0].flat_segment_ranges[0].offset), 1));
  ASSERT_EQ(punched_data->data()[0], 0);

  EXPECT_THROW((void)translater->estimated_byte_size_of_cell(translater->num_cells()), std::out_of_range);
  EXPECT_THROW(
      (void)translater->cells_storage_bytes({static_cast<milvus::cachinglayer::cid_t>(translater->num_cells())}),
      std::out_of_range);
  EXPECT_THROW((void)translater->get_cells(nullptr, {-1}), std::out_of_range);
}

TEST_F(VortexLocalFormatTest, TestTranslaterRejectsInvalidInputs) {
  auto sparse_fs = std::make_shared<InMemoryVortexRangeFileSystem>();
  auto cell_metas = std::make_shared<const VortexCellMetas>();

  ASSERT_STATUS_NOT_OK(
      VortexTranslater::Make(cell_metas, nullptr, test_file_name_, sparse_fs, "test-file.vx.sparse").status());
  ASSERT_STATUS_NOT_OK(
      VortexTranslater::Make(cell_metas, file_system_, test_file_name_, nullptr, "test-file.vx.sparse").status());
  ASSERT_STATUS_NOT_OK(
      VortexTranslater::Make(nullptr, file_system_, test_file_name_, sparse_fs, "test-file.vx.sparse").status());

  auto local_fs = std::make_shared<arrow::fs::LocalFileSystem>();
  ASSERT_STATUS_NOT_OK(
      VortexTranslater::Make(cell_metas, file_system_, test_file_name_, local_fs, "test-file.vx.sparse").status());
}

// The footer descriptor claims a byte range; this checks it against the file it
// came from. This is a data-format error, not a bad caller argument: the
// persisted layout was parsed and found to contradict the file.
//
// Tests the rule, not the path to it. The guard sits behind
// VortexFile::OpenUnique, which is handed the same file_size the check later
// uses, so no amount of tampering with file_size reaches it -- inflate and the
// tail read runs past EOF into sparse zeroes, deflate and it lands mid-file;
// either way the EOF trailer fails to parse and OpenUnique rejects the file
// first. An end-to-end case needs a hand-built vortex file whose trailer parses
// but whose descriptor lies, and no such fixture exists in this tree. Saying so
// here rather than leaving a green test to imply coverage it does not have.
TEST(VortexFooterRangeTest, InvalidRangeIsDataFormat) {
  constexpr uint64_t kFileSize = 1000;

  struct Case {
    std::vector<uint64_t> range;
    const char* what;
  };
  const Case bad[] = {
      {{}, "empty -- not a pair"},
      {{100}, "one element -- not a pair"},
      {{100, 200, 300}, "three elements -- not a pair"},
      {{kFileSize + 1, 0}, "offset past the end"},
      {{0, kFileSize + 1}, "length past the end from offset 0"},
      {{900, 200}, "offset in range, but offset+length past the end"},
      // The check is written as `length > file_size - offset` precisely so this
      // does not wrap to a pass.
      {{1, std::numeric_limits<uint64_t>::max()}, "length that would overflow offset+length"},
  };

  for (const auto& c : bad) {
    auto status = vortex::internal::CheckVortexFooterRange(c.range, kFileSize, "some.vortex");
    ASSERT_FALSE(status.ok()) << c.what;

    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << c.what << ": " << status.ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::VortexDataFormat) << c.what;
    EXPECT_EQ(CategoryForExtendStatusCode(detail->code()), ErrorCategory::DataFormat) << c.what;
    EXPECT_FALSE(RetryableForExtendStatusCode(detail->code())) << c.what;
    EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::DataFormatBroken) << c.what;
    EXPECT_NE(status.ToString().find("some.vortex"), std::string::npos) << c.what;
  }

  const Case good[] = {
      {{0, 0}, "empty range at the start"},
      {{0, kFileSize}, "the whole file"},
      {{kFileSize, 0}, "empty range exactly at the end"},
      {{900, 100}, "ends exactly at the end"},
  };
  for (const auto& c : good) {
    EXPECT_TRUE(vortex::internal::CheckVortexFooterRange(c.range, kFileSize, "some.vortex").ok()) << c.what;
  }
}

TEST(VortexFooterRangeTest, ZeroLengthFlatSegmentIsLegal) {
  auto range = vortex::internal::FlatSegmentByteRangeFromBytes({4096, 0}, 7);
  ASSERT_TRUE(range.ok()) << range.status().ToString();
  EXPECT_EQ(range->offset, 4096u);
  EXPECT_EQ(range->length, 0u);
}

TEST(VortexFooterRangeTest, WrongSizedBridgeResultIsInternal) {
  for (const std::vector<uint64_t>& bytes :
       {std::vector<uint64_t>{}, std::vector<uint64_t>{4096}, std::vector<uint64_t>{4096, 128, 9}}) {
    auto range = vortex::internal::FlatSegmentByteRangeFromBytes(bytes, 7);
    ASSERT_FALSE(range.ok()) << "size " << bytes.size();
    EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(range.status()), nullptr)
        << "size " << bytes.size() << ": " << range.status().ToString();
    EXPECT_EQ(ToSegcoreError(range.status()).get_error_code(), milvus::StorageError)
        << "size " << bytes.size() << ": " << range.status().ToString();
  }
}

TEST(VortexFooterRangeTest, SegmentLookupPreservesCaughtPanicAsUnexpected) {
  auto status =
      vortex::internal::ClassifyVortexLayoutSegmentLookupFailure(arrow::Status::UnknownError("decoder panicked"), 7);

  EXPECT_TRUE(status.IsUnknownError()) << status.ToString();
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr);
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageError);
}

TEST(VortexFooterRangeTest, OrdinaryPersistedSegmentLookupFailureIsDataFormat) {
  auto status = vortex::internal::ClassifyVortexLayoutSegmentLookupFailure(
      arrow::Status::Invalid("segment index is out of bounds"), 7);

  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::VortexDataFormat);
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::DataFormatBroken);
}

TEST(VortexFieldLayoutEncodingTest, InvalidEncodingIsDataFormat) {
  struct Case {
    std::vector<uint64_t> encoded;
    uint64_t rows;
    const char* what;
  };
  const Case bad[] = {
      {{}, 1, "missing header"},
      {{1}, 1, "missing unit count"},
      {{99, 0}, 0, "unknown granularity"},
      {{1, 0}, 3, "non-empty file without units"},
      {{1, std::numeric_limits<uint64_t>::max()}, 3, "impossible unit count"},
      {{1, 1, 0, 0, 3}, 3, "truncated unit"},
      {{1, 1, 0, 0, 3, 2, 7}, 3, "truncated segment list"},
      {{1, 1, 0, 0, 3, 0, 42}, 3, "trailing word"},
      {{1, 1, 0, 4, 0, 0}, 3, "row offset past file rows"},
      {{1, 1, 0, 2, 2, 0}, 3, "row range past file rows"},
      {{1, 1, 0, std::numeric_limits<uint64_t>::max(), 2, 0},
       std::numeric_limits<uint64_t>::max(),
       "row range addition would overflow"},
  };

  for (const auto& c : bad) {
    auto status = vortex::internal::ValidateVortexFieldLayoutEncoding(c.encoded, c.rows, "id");
    ASSERT_FALSE(status.ok()) << c.what;
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << c.what << ": " << status.ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::VortexDataFormat) << c.what;
    EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::DataFormatBroken) << c.what;
    EXPECT_NE(status.message().find("id"), std::string::npos) << c.what;
  }
}

TEST(VortexFieldLayoutEncodingTest, HealthyEncodingsAreAccepted) {
  ASSERT_STATUS_OK(vortex::internal::ValidateVortexFieldLayoutEncoding({1, 0}, 0, "id"));
  ASSERT_STATUS_OK(vortex::internal::ValidateVortexFieldLayoutEncoding({1, 1, 0, 0, 3, 2, 7, 8}, 3, "id"));
  ASSERT_STATUS_OK(vortex::internal::ValidateVortexFieldLayoutEncoding({2, 1, 4, 0, 3, 0}, 3, "id"));
}

TEST_F(VortexLocalFormatTest, StaleFooterSizeFailsAfterOneTailRead) {
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());
  const auto full_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  const auto real_footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  ASSERT_GT(real_footer_size, 1u);

  for (uint64_t understated : {uint64_t{1}, real_footer_size / 2}) {
    auto sparse_fs = std::make_shared<InMemoryVortexRangeFileSystem>();
    auto reader =
        std::make_shared<VortexFooterReader>(sparse_fs, "test-file.vx.sparse", test_file_name_, full_size, understated);

    auto status = reader->Open(file_system_);
    ASSERT_FALSE(status.ok()) << "understated footer size " << understated;
    ASSERT_AND_ASSIGN(auto sparse_file, sparse_fs->GetInMemoryFile("test-file.vx.sparse"));
    EXPECT_EQ(sparse_file->WriteRanges().size(), 1u) << "footer failure must not trigger another range read";
  }

  // The same file opens when its recorded footer size is correct.
  auto sparse_fs = std::make_shared<InMemoryVortexRangeFileSystem>();
  auto reader = MakeFooterReader(cgfile, sparse_fs);
  ASSERT_STATUS_OK(reader->Open(file_system_));
}

// An invalid trailer is a Vortex data-format failure. Classification is
// attached only after the complete persisted object has been loaded and the
// final decoder attempt still fails, without matching producer text.
TEST_F(VortexLocalFormatTest, VortexInvalidTrailerFailsClosedWithoutClaimingCorruption) {
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());
  const auto full_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  ASSERT_GT(full_size, 64u);

  // Keep the length unchanged so the decoder, rather than a short-read guard,
  // rejects the trailer.
  ASSERT_AND_ASSIGN(auto input, file_system_->OpenInputFile(test_file_name_));
  ASSERT_AND_ASSIGN(auto whole, input->ReadAt(0, static_cast<int64_t>(full_size)));
  ASSERT_STATUS_OK(input->Close());
  {
    std::vector<uint8_t> smashed(whole->data(), whole->data() + full_size);
    for (size_t i = smashed.size() - 32; i < smashed.size(); ++i) {
      smashed[i] ^= 0xff;
    }
    ASSERT_AND_ASSIGN(auto out, file_system_->OpenOutputStream(test_file_name_));
    ASSERT_STATUS_OK(out->Write(smashed.data(), static_cast<int64_t>(smashed.size())));
    ASSERT_STATUS_OK(out->Close());
  }

  auto sparse_fs = std::make_shared<InMemoryVortexRangeFileSystem>();
  auto footer_reader = MakeFooterReader(cgfile, sparse_fs);
  auto status = footer_reader->Open(file_system_);

  ASSERT_FALSE(status.ok()) << "a vortex file with a smashed trailer opened successfully";
  // Deliberately NOT DataFormatBroken. Vortex reports footer-deserialization
  // failures through its catch-all `Other` variant, so nothing below us
  // positively identifies corruption, and neither this layer nor the bridge
  // infers it: an unclassified failure stays in the conservative non-retriable
  // bucket rather than claiming a verdict nobody established. What is pinned
  // here is that it fails closed, carries no invented classification, and does
  // not leak the internal marker.
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr) << status.ToString();
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageError) << status.ToString();
  EXPECT_EQ(status.ToString().find("__LOON_RUST_BRIDGE_ERRCODE__"), std::string::npos) << status.ToString();
}

// A corrupt file must not be able to kill the process.
//
// These are extern "Rust" entry points: cxx turns a returned Err into a C++
// exception, but a PANIC unwinding across that boundary aborts. Three altered
// bytes inside a healthy file's footer were enough to reach one -- so a single
// bad object could take the node down, and no error code helps after that.
//
// The assertion is deliberately weak on WHICH error comes back: different
// mutations reach different failure modes, and pinning one would make this
// test about vortex's internals. What must hold is that the process survives
// and the caller is told something -- if the guard regresses, this test does
// not fail, it aborts the whole binary with 134.
TEST_F(VortexLocalFormatTest, CorruptFooterCannotAbortTheProcess) {
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());
  const auto full_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  ASSERT_GT(full_size, 512u);

  ASSERT_AND_ASSIGN(auto input, file_system_->OpenInputFile(test_file_name_));
  ASSERT_AND_ASSIGN(auto whole, input->ReadAt(0, static_cast<int64_t>(full_size)));
  ASSERT_STATUS_OK(input->Close());
  const std::vector<uint8_t> pristine(whole->data(), whole->data() + full_size);

  // Walk the footer region a few bytes at a time. Any one of these may or may
  // not provoke a panic; the point is that none of them may end the process.
  for (uint64_t back : {uint64_t{16}, uint64_t{24}, uint64_t{40}, uint64_t{64}, uint64_t{128}}) {
    auto smashed = pristine;
    for (uint64_t i = 0; i < 3; ++i) {
      smashed[smashed.size() - back + i] ^= 0x5a;
    }
    {
      ASSERT_AND_ASSIGN(auto out, file_system_->OpenOutputStream(test_file_name_));
      ASSERT_STATUS_OK(out->Write(smashed.data(), static_cast<int64_t>(smashed.size())));
      ASSERT_STATUS_OK(out->Close());
    }

    auto sparse_fs = std::make_shared<InMemoryVortexRangeFileSystem>();
    auto reader =
        std::make_shared<VortexFooterReader>(sparse_fs, "test-file.vx.sparse", test_file_name_, full_size, uint64_t{0});
    auto status = reader->Open(file_system_);
    // Surviving the call IS the assertion. A verdict either way is fine.
    SUCCEED() << "back=" << back << " -> " << (status.ok() ? std::string("opened") : status.ToString());
  }

  // Restore, so a later test in this fixture is not reading smashed bytes.
  ASSERT_AND_ASSIGN(auto out, file_system_->OpenOutputStream(test_file_name_));
  ASSERT_STATUS_OK(out->Write(pristine.data(), static_cast<int64_t>(pristine.size())));
  ASSERT_STATUS_OK(out->Close());
}

// Corruption in the DATA region: opens cleanly, dies while streaming.
//
// A different boundary from CorruptFooterCannotAbortTheProcess. Decoding is
// lazy, so the panic fires inside iter.next() -- called per batch from the
// Arrow C stream callback -- long after every entry-point guard has returned.
// Smashing bytes well before the footer leaves the file openable and defers the
// failure to exactly there.
//
// As with the footer test, the assertion is survival, not a specific verdict:
// where the decoder dies depends on which byte moved. If the guard regresses
// this does not fail, it takes the whole binary down with 134.
TEST_F(VortexLocalFormatTest, CorruptDataRegionCannotAbortTheStream) {
  ASSERT_AND_ASSIGN(auto cgfile, WriteVortexFile());
  const auto full_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);
  const auto footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  ASSERT_GT(full_size, footer_size + 4096);

  ASSERT_AND_ASSIGN(auto input, file_system_->OpenInputFile(test_file_name_));
  ASSERT_AND_ASSIGN(auto whole, input->ReadAt(0, static_cast<int64_t>(full_size)));
  ASSERT_STATUS_OK(input->Close());
  const std::vector<uint8_t> pristine(whole->data(), whole->data() + full_size);

  // Swept rather than pinned to a few offsets. Which byte provokes a panic
  // depends on how the writer laid the file out, and a first version that
  // guessed three positions hit none of them -- it passed, and passed just as
  // happily with the guard removed. Sweeping costs about a second and does not
  // rot when the layout changes.
  const uint64_t stride = (full_size / 64) + 1;
  for (uint64_t offset = 64; offset + 64 < full_size - footer_size; offset += stride) {
    auto smashed = pristine;
    for (uint64_t i = 0; i < 8; ++i) {
      smashed[offset + i] ^= 0xa5;
    }
    {
      ASSERT_AND_ASSIGN(auto out, file_system_->OpenOutputStream(test_file_name_));
      ASSERT_STATUS_OK(out->Write(smashed.data(), static_cast<int64_t>(smashed.size())));
      ASSERT_STATUS_OK(out->Close());
    }

    auto vx_reader = vortex::VortexFormatReader(file_system_, schema_, test_file_name_, properties_, data_columns(),
                                                full_size, footer_size);
    auto open_status = vx_reader.open();
    if (!open_status.ok()) {
      continue;  // died at open -- the other test's territory, still no abort
    }

    auto stream_result = vx_reader.read_with_plan(VortexReadPlan{
        .op = VortexReadPlan::RangeScan{.ranges = {RowRange{.start = 0, .end = 4096}}},
    });
    if (!stream_result.ok()) {
      continue;
    }
    auto array_stream = std::move(stream_result).ValueOrDie();
    // Draining is where the lazy decode -- and the panic -- actually happens.
    auto chunked = arrow::ImportChunkedArray(&array_stream);
    SUCCEED() << "offset=" << offset << " -> " << (chunked.ok() ? std::string("read") : chunked.status().ToString());
  }

  ASSERT_AND_ASSIGN(auto out, file_system_->OpenOutputStream(test_file_name_));
  ASSERT_STATUS_OK(out->Write(pristine.data(), static_cast<int64_t>(pristine.size())));
  ASSERT_STATUS_OK(out->Close());
}

TEST_F(VortexLocalFormatTest, ShortFileIsDataFormatEvenWhenTheSizeWasHandedToUs) {
  const std::string short_path = "truly-short.vx";
  const std::vector<uint8_t> stub(VortexEofSize() - 1, 0x00);
  {
    ASSERT_AND_ASSIGN(auto out, file_system_->OpenOutputStream(short_path));
    ASSERT_STATUS_OK(out->Write(stub.data(), static_cast<int64_t>(stub.size())));
    ASSERT_STATUS_OK(out->Close());
  }

  // Supplied size (matching reality) and stat'd size (0 sentinel) must reach the
  // same verdict.
  for (uint64_t supplied : {static_cast<uint64_t>(stub.size()), uint64_t{0}}) {
    auto sparse_fs = std::make_shared<InMemoryVortexRangeFileSystem>();
    auto reader = std::make_shared<VortexFooterReader>(sparse_fs, "short.vx.sparse", short_path, supplied, uint64_t{0});

    auto status = reader->Open(file_system_);
    ASSERT_FALSE(status.ok()) << "supplied " << supplied;

    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << "supplied " << supplied << " arrived unclassified: " << status.ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::VortexDataFormat) << supplied;
    EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::DataFormatBroken) << supplied;
  }

  ASSERT_STATUS_OK(file_system_->DeleteFile(short_path));
}

}  // namespace milvus_storage
