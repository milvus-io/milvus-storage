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
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include <boost/filesystem/operations.hpp>

#include "milvus-storage/common/constants.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "test_env.h"

namespace milvus_storage {

using namespace vortex;

// Validates VortexFormatReader::get_column_sizes(): whole-file per-column uncompressed sizes
// (from footer statistics) mapped to the projected columns as raw weights.
class VortexColumnSizesTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (IsCloudEnv()) {
      GTEST_SKIP() << "Vortex writer/reader is local-fs only in this test.";
    }
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    // Statistics are on by default, but be explicit: get_column_sizes relies on the
    // per-column UncompressedSizeInBytes footer statistic.
    api::SetValue(properties_, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, "true");
    ASSERT_AND_ASSIGN(file_system_, GetFileSystem(properties_));
  }

  void TearDown() override {
    if (file_system_ != nullptr && !test_path_.empty()) {
      (void)file_system_->DeleteFile(test_path_);
      boost::filesystem::remove_all(test_path_);
    }
  }

  std::shared_ptr<arrow::fs::FileSystem> file_system_;
  api::Properties properties_;
  std::string test_path_;
};

TEST_F(VortexColumnSizesTest, GetColumnSizesShapeAndProportions) {
  constexpr int64_t kRows = 20000;
  constexpr size_t kStrLen = 128;

  auto schema = arrow::schema({
      arrow::field("id", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
      arrow::field("name", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"101"})),
  });

  arrow::Int64Builder id_builder;
  arrow::StringBuilder name_builder;
  std::string value(kStrLen, 'x');
  for (int64_t i = 0; i < kRows; ++i) {
    ASSERT_STATUS_OK(id_builder.Append(i));
    ASSERT_STATUS_OK(name_builder.Append(value));
  }
  std::shared_ptr<arrow::Array> ids;
  std::shared_ptr<arrow::Array> names;
  ASSERT_STATUS_OK(id_builder.Finish(&ids));
  ASSERT_STATUS_OK(name_builder.Finish(&names));
  auto batch = arrow::RecordBatch::Make(schema, kRows, {ids, names});

  test_path_ = "vortex-column-sizes.vx";
  (void)file_system_->DeleteFile(test_path_);
  boost::filesystem::remove_all(test_path_);

  ASSERT_AND_ASSIGN(auto vx_writer, VortexFileWriter::Open(file_system_, schema, test_path_, properties_));
  ASSERT_STATUS_OK(vx_writer->Write(batch));
  ASSERT_STATUS_OK(vx_writer->Flush());
  ASSERT_AND_ASSIGN(auto cgfile, vx_writer->Close());
  ASSERT_EQ(cgfile.end_index, kRows);

  auto footer_size = cgfile.Get<uint64_t>(api::kPropertyFooterSize);
  auto file_size = cgfile.Get<uint64_t>(api::kPropertyFileSize);

  // Project id (small, index 0) then name (large, index 1) in that order.
  VortexFormatReader reader(file_system_, schema, test_path_, properties_, std::vector<std::string>{"id", "name"},
                            file_size, footer_size);
  ASSERT_STATUS_OK(reader.open());
  ASSERT_AND_ASSIGN(auto rg_infos, reader.get_row_group_infos());
  ASSERT_FALSE(rg_infos.empty());

  for (size_t rg_idx = 0; rg_idx < rg_infos.size(); ++rg_idx) {
    // get_column_sizes returns raw whole-file per-column weights (not chunk-normalized);
    // ColumnGroupReader normalizes them to the chunk's real memory.
    ASSERT_AND_ASSIGN(auto column_sizes, reader.get_column_sizes(static_cast<int>(rg_idx)));
    ASSERT_FALSE(column_sizes.empty()) << "footer statistics should populate per-column weights";
    // Inner index: one weight per projected column, in projection (id, name) order.
    ASSERT_EQ(column_sizes.size(), 2u);

    const uint64_t id_size = column_sizes[0];
    const uint64_t name_size = column_sizes[1];

    // The variable-length string column weighs more than the 8-byte int column.
    EXPECT_GT(name_size, id_size);
    EXPECT_GT(name_size, 0u);
  }
}

}  // namespace milvus_storage
