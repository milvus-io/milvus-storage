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

#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/testing/gtest_util.h>

#include "milvus-storage/common/constants.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/format/format_reader.h"
#include "test_env.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

// Validates ParquetFormatReader::get_column_sizes(): the per-column footer sizes must be
// keyed by projected TOP-LEVEL column (in projection order), with nested columns summing
// their Parquet leaf ColumnChunk sizes into a single entry.
class ParquetColumnSizesTest : public ::testing::Test {
  protected:
  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    base_path_ = GetTestBasePath("parquet-column-sizes-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));
  }

  void TearDown() override {
    if (!IsCloudEnv()) {
      ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    }
  }

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
  Properties properties_;
};

TEST_F(ParquetColumnSizesTest, NestedLeafGroupingAndProjectionOrder) {
  auto field_metadata = [](const std::string& field_id) {
    return arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {field_id});
  };

  // profile (struct)  -> 2 Parquet leaf columns (score, label)
  // events  (list<struct>) -> 2 Parquet leaf columns (code, message)
  // id, note -> 1 leaf column each. Total leaves = 6, top-level columns = 4.
  auto profile_type =
      arrow::struct_({arrow::field("score", arrow::int32(), false), arrow::field("label", arrow::utf8(), false)});
  auto event_type =
      arrow::struct_({arrow::field("code", arrow::int32(), false), arrow::field("message", arrow::utf8(), false)});
  auto events_type = arrow::list(arrow::field("item", event_type, false));
  auto nested_schema = arrow::schema({
      arrow::field("id", arrow::int64(), false, field_metadata("0")),
      arrow::field("profile", profile_type, false, field_metadata("1")),
      arrow::field("events", events_type, false, field_metadata("2")),
      arrow::field("note", arrow::utf8(), false, field_metadata("3")),
  });

  arrow::Int64Builder id_builder;
  ASSERT_STATUS_OK(id_builder.AppendValues({0, 1, 2, 3}));
  ASSERT_AND_ASSIGN(auto ids, id_builder.Finish());

  arrow::Int32Builder score_builder;
  ASSERT_STATUS_OK(score_builder.AppendValues({10, 20, 30, 40}));
  ASSERT_AND_ASSIGN(auto scores, score_builder.Finish());
  arrow::StringBuilder label_builder;
  ASSERT_STATUS_OK(label_builder.AppendValues({"cold", "warm", "hot", "peak"}));
  ASSERT_AND_ASSIGN(auto labels, label_builder.Finish());
  auto profiles =
      std::make_shared<arrow::StructArray>(profile_type, 4, std::vector<std::shared_ptr<arrow::Array>>{scores, labels});

  arrow::Int32Builder event_code_builder;
  ASSERT_STATUS_OK(event_code_builder.AppendValues({1, 2, 3, 4, 5}));
  ASSERT_AND_ASSIGN(auto event_codes, event_code_builder.Finish());
  arrow::StringBuilder event_message_builder;
  ASSERT_STATUS_OK(event_message_builder.AppendValues({"created", "queued", "running", "done", "archived"}));
  ASSERT_AND_ASSIGN(auto event_messages, event_message_builder.Finish());
  auto event_values = std::make_shared<arrow::StructArray>(
      event_type, 5, std::vector<std::shared_ptr<arrow::Array>>{event_codes, event_messages});
  arrow::Int32Builder event_offsets_builder;
  ASSERT_STATUS_OK(event_offsets_builder.AppendValues({0, 2, 3, 3, 5}));
  ASSERT_AND_ASSIGN(auto event_offsets, event_offsets_builder.Finish());
  ASSERT_AND_ASSIGN(auto events, arrow::ListArray::FromArrays(events_type, *event_offsets, *event_values));

  arrow::StringBuilder note_builder;
  ASSERT_STATUS_OK(note_builder.AppendValues({"note-a", "note-b", "note-c", "note-d"}));
  ASSERT_AND_ASSIGN(auto notes, note_builder.Finish());

  auto record_batch = arrow::RecordBatch::Make(nested_schema, 4, {ids, profiles, events, notes});

  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, nested_schema));
  auto writer = Writer::create(base_path_, nested_schema, std::move(policy), properties_);
  ASSERT_OK(writer->write(record_batch));
  ASSERT_AND_ASSIGN(auto column_groups, writer->close());
  ASSERT_EQ(column_groups->size(), 1);
  ASSERT_EQ(column_groups->front()->files.size(), 1);
  const auto& cg_file = column_groups->front()->files.front();

  // Case 1: full projection. Four top-level columns, so four inner entries -- NOT six
  // (the number of Parquet leaf columns).
  {
    std::vector<std::string> needed_columns = {"id", "profile", "events", "note"};
    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nested_schema, LOON_FORMAT_PARQUET, cg_file, properties_,
                                                        needed_columns, nullptr));
    ASSERT_AND_ASSIGN(auto sizes, reader->get_column_sizes(0));
    ASSERT_EQ(sizes.size(), 4u) << "one weight per top-level column, not per Parquet leaf";
    for (const uint64_t s : sizes) {
      EXPECT_GT(s, 0u);
    }
  }

  // Case 2: reordered projection {note, profile, id}. Inner index follows projection order,
  // and the nested "profile" struct sums its two leaf ColumnChunks into a single weight that
  // matches an independently-projected "profile" read.
  {
    std::vector<std::string> profile_only = {"profile"};
    ASSERT_AND_ASSIGN(auto profile_reader, FormatReader::create(nested_schema, LOON_FORMAT_PARQUET, cg_file,
                                                                properties_, profile_only, nullptr));
    ASSERT_AND_ASSIGN(auto profile_sizes, profile_reader->get_column_sizes(0));
    ASSERT_EQ(profile_sizes.size(), 1u);
    EXPECT_GT(profile_sizes[0], 0u);

    std::vector<std::string> reordered = {"note", "profile", "id"};
    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(nested_schema, LOON_FORMAT_PARQUET, cg_file, properties_,
                                                        reordered, nullptr));
    ASSERT_AND_ASSIGN(auto sizes, reader->get_column_sizes(0));
    ASSERT_EQ(sizes.size(), 3u);
    // sizes[1] corresponds to "profile" and must equal the standalone profile projection,
    // proving the leaf->top-level grouping is projection-order-independent.
    EXPECT_EQ(sizes[1], profile_sizes[0]);
    for (const uint64_t s : sizes) {
      EXPECT_GT(s, 0u);
    }
  }
}

}  // namespace milvus_storage::test
