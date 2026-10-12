// Copyright 2023 Zilliz
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

#include <memory>
#include <chrono>
#include <filesystem>

#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/table.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/filesystem/localfs.h>
#include <parquet/arrow/writer.h>

#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/packed/reader.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/config.h"
#include "test_env.h"

namespace milvus_storage {

class PackedRecordBatchReaderTest : public ::testing::Test {
  protected:
  void SetUp() override {}
};

TEST_F(PackedRecordBatchReaderTest, RowOffsetMinHeapTest) {
  RowOffsetMinHeap minHeap;

  minHeap.emplace(1, 30);
  minHeap.emplace(2, 20);
  minHeap.emplace(3, 40);
  minHeap.emplace(4, 10);

  EXPECT_EQ(minHeap.top().second, 10);
  minHeap.pop();
  EXPECT_EQ(minHeap.top().second, 20);
  minHeap.pop();
  EXPECT_EQ(minHeap.top().second, 30);
  minHeap.pop();
  EXPECT_EQ(minHeap.top().second, 40);
}

struct ProjectionRefillCase {
  const char* name;
  std::vector<int> selected_groups;
  int64_t buffer_size;
  bool reverse_path_names;
  int expected_batches;
};

class PackedProjectionRefillTest : public ::testing::TestWithParam<ProjectionRefillCase> {
  protected:
  void SetUp() override {
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();
    path_ = (std::filesystem::temp_directory_path() /
             ("packed-projection-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
                .string();
    ASSERT_STATUS_OK(fs_->CreateDir(path_));
  }

  void TearDown() override { ASSERT_STATUS_OK(fs_->DeleteDir(path_)); }

  std::shared_ptr<arrow::fs::LocalFileSystem> fs_;
  std::string path_;
};

TEST_P(PackedProjectionRefillTest, PreservesValuesAcrossRefills) {
  constexpr int64_t rows = 8;
  const auto& config = GetParam();
  std::vector<std::string> paths;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  GroupFieldIDList groups;
  for (int group = 0; group < 4; ++group) {
    const auto field_id = 100 + group;
    groups.AddFieldIDList(FieldIDList({field_id}));
    fields.push_back(arrow::field("field_" + std::to_string(group), arrow::int64(), false,
                                  arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {std::to_string(field_id)})));
    paths.push_back(path_ + "/" + std::to_string(config.reverse_path_names ? 3 - group : group) + ".parquet");
  }

  // The first three groups hold all eight rows in one chunk. The last group
  // has four two-row groups, so its refills must preserve other groups' offsets.
  for (int group = 0; group < 4; ++group) {
    arrow::Int64Builder builder;
    for (int64_t row = 0; row < rows; ++row) {
      ASSERT_STATUS_OK(builder.Append(group * 100 + row));
    }
    ASSERT_AND_ASSIGN(auto values, builder.Finish());
    auto schema = arrow::schema({fields[group]});
    auto table = arrow::Table::Make(schema, {values});
    const int64_t row_group_rows = group == 3 ? 2 : rows;
    RowGroupMetadataVector row_groups;
    for (int64_t offset = 0; offset < rows; offset += row_group_rows) {
      row_groups.Add(RowGroupMetadata(row_group_rows * sizeof(int64_t), row_group_rows, offset));
    }
    ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(paths[group]));
    ASSERT_AND_ASSIGN(auto writer, ::parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), sink,
                                                                      ::parquet::default_writer_properties(),
                                                                      ::parquet::default_arrow_writer_properties()));
    ASSERT_STATUS_OK(writer->WriteTable(*table, row_group_rows));
    ASSERT_STATUS_OK(writer->AddKeyValueMetadata(
        arrow::key_value_metadata({GROUP_FIELD_ID_LIST_META_KEY, ROW_GROUP_META_KEY, STORAGE_VERSION_KEY},
                                  {groups.Serialize(), row_groups.Serialize(), "1.0.0"})));
    ASSERT_STATUS_OK(writer->Close());
    ASSERT_STATUS_OK(sink->Close());
  }

  std::vector<std::shared_ptr<arrow::Field>> selected_fields;
  for (auto group : config.selected_groups) {
    selected_fields.push_back(fields[group]);
  }
  auto arrow_props = ::parquet::default_arrow_reader_properties();
  arrow_props.set_batch_size(rows);
  ASSERT_AND_ASSIGN(auto reader,
                    PackedRecordBatchReader::Make(fs_, paths, arrow::schema(selected_fields), config.buffer_size,
                                                  ::parquet::default_reader_properties(), arrow_props));
  int64_t offset = 0;
  int batches = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_STATUS_OK(reader->ReadNext(&batch));
    if (!batch) {
      break;
    }
    ASSERT_GT(batch->num_rows(), 0);
    ASSERT_LE(offset + batch->num_rows(), rows);
    ASSERT_EQ(batch->num_columns(), config.selected_groups.size());
    for (int column = 0; column < batch->num_columns(); ++column) {
      auto values = std::static_pointer_cast<arrow::Int64Array>(batch->column(column));
      for (int64_t row = 0; row < batch->num_rows(); ++row) {
        EXPECT_EQ(values->Value(row), config.selected_groups[column] * 100 + offset + row)
            << "group=" << config.selected_groups[column] << " row=" << offset + row;
      }
    }
    offset += batch->num_rows();
    ++batches;
  }
  EXPECT_EQ(offset, rows);
  if (config.expected_batches > 0) {
    EXPECT_EQ(batches, config.expected_batches);
  }
  ASSERT_STATUS_OK(reader->Close());
}

INSTANTIATE_TEST_SUITE_P(PackedReader,
                         PackedProjectionRefillTest,
                         ::testing::Values(ProjectionRefillCase{"OmitMiddleGroup", {0, 2, 3}, 1, false, 4},
                                           ProjectionRefillCase{"FullSchema", {0, 1, 2, 3}, 1, false, 4},
                                           ProjectionRefillCase{"OmitTrailingGroup", {0, 1, 2}, 1, false, 1},
                                           ProjectionRefillCase{"FullSchemaReorderedPaths", {0, 1, 2, 3}, 1, true, 4},
                                           ProjectionRefillCase{
                                               "OmitMiddleGroupUnlimitedBuffer", {0, 2, 3}, 0, false, 0}),
                         [](const ::testing::TestParamInfo<ProjectionRefillCase>& info) { return info.param.name; });

}  // namespace milvus_storage
