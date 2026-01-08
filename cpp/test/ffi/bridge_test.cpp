
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

#include <sstream>
#include <random>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/bridge.h"
#include "test_env.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

class BridgeTest : public ::testing::Test {};

TEST_F(BridgeTest, ExportImportColumnGroups) {
  // create column groups
  ColumnGroups cgs;

  ColumnGroupFile cgf1{
      .path = "test_file1", .start_index = 0, .end_index = 10, .metadata = std::vector<uint8_t>({1, 2, 3, 4})};

  ColumnGroupFile cgf2{
      .path = "test_file2",
      .start_index = 2000,
      .end_index = 10000,
      .metadata = std::vector<uint8_t>(),
  };

  ColumnGroupFile cgf3{
      .path = "test_file2",
      .start_index = 100,
      .end_index = 200,
      .metadata = std::vector<uint8_t>({1, 3, 4, 5}),
  };

  ColumnGroup cg1{
      .columns = std::vector<std::string>({"col1", "col2"}),
      .format = "parquet",
      .files = std::vector<ColumnGroupFile>({cgf1, cgf2}),
  };

  ColumnGroup cg2{
      .columns = std::vector<std::string>({"col3"}),
      .format = "parquet",
      .files = std::vector<ColumnGroupFile>({cgf3}),
  };

  cgs.push_back(std::make_shared<ColumnGroup>(cg1));
  cgs.push_back(std::make_shared<ColumnGroup>(cg2));

  // test export and import
  CColumnGroups* ccgs = nullptr;
  ASSERT_STATUS_OK(export_column_groups(cgs, &ccgs));

  // verify export C struct
  {
    ASSERT_NE(ccgs, nullptr);
    ASSERT_NE(ccgs->column_group_array, nullptr);
    ASSERT_EQ(ccgs->num_of_column_groups, 2);
    ASSERT_NE(ccgs->column_group_array[0].columns, nullptr);
    ASSERT_EQ(ccgs->column_group_array[0].num_of_columns, 2);
    ASSERT_EQ(ccgs->column_group_array[0].columns[0], cg1.columns[0]);
    ASSERT_EQ(ccgs->column_group_array[0].columns[1], cg1.columns[1]);

    ASSERT_EQ(ccgs->column_group_array[0].format, cg1.format);
    ASSERT_NE(ccgs->column_group_array[0].files, nullptr);
    ASSERT_EQ(ccgs->column_group_array[0].num_of_files, 2);
    ASSERT_EQ(ccgs->column_group_array[0].files[0].path, cg1.files[0].path);
    ASSERT_EQ(ccgs->column_group_array[0].files[0].start_index, cg1.files[0].start_index);
    ASSERT_EQ(ccgs->column_group_array[0].files[0].end_index, cg1.files[0].end_index);
    ASSERT_EQ(std::vector<uint8_t>(
                  ccgs->column_group_array[0].files[0].metadata,
                  ccgs->column_group_array[0].files[0].metadata + ccgs->column_group_array[0].files[0].metadata_size),
              cg1.files[0].metadata);

    ASSERT_EQ(ccgs->column_group_array[0].files[1].path, cg1.files[1].path);
    ASSERT_EQ(ccgs->column_group_array[0].files[1].start_index, cg1.files[1].start_index);
    ASSERT_EQ(ccgs->column_group_array[0].files[1].end_index, cg1.files[1].end_index);
    ASSERT_EQ(ccgs->column_group_array[0].files[1].metadata, nullptr);

    ASSERT_NE(ccgs->column_group_array[1].columns, nullptr);
    ASSERT_EQ(ccgs->column_group_array[1].num_of_columns, 1);
    ASSERT_EQ(ccgs->column_group_array[1].columns[0], cg2.columns[0]);

    ASSERT_EQ(ccgs->column_group_array[1].format, cg2.format);
    ASSERT_NE(ccgs->column_group_array[1].files, nullptr);
    ASSERT_EQ(ccgs->column_group_array[1].num_of_files, 1);
    ASSERT_EQ(ccgs->column_group_array[1].files[0].path, cgf3.path);
    ASSERT_EQ(ccgs->column_group_array[1].files[0].start_index, cgf3.start_index);
    ASSERT_EQ(ccgs->column_group_array[1].files[0].end_index, cgf3.end_index);
    ASSERT_EQ(std::vector<uint8_t>(
                  ccgs->column_group_array[1].files[0].metadata,
                  ccgs->column_group_array[1].files[0].metadata + ccgs->column_group_array[1].files[0].metadata_size),
              cgf3.metadata);
  }

  ColumnGroups imported_cgs;
  ASSERT_STATUS_OK(import_column_groups(ccgs, &imported_cgs));

  // verify import column groups should same with origin one
  {
    ASSERT_EQ(cgs.size(), imported_cgs.size());
    for (size_t i = 0; i < cgs.size(); ++i) {
      auto left_cg = cgs[i];
      auto right_cg = imported_cgs[i];

      ASSERT_EQ(left_cg->columns, right_cg->columns);
      ASSERT_EQ(left_cg->format, right_cg->format);
      ASSERT_EQ(left_cg->files.size(), right_cg->files.size());
      for (size_t j = 0; j < left_cg->files.size(); ++j) {
        ASSERT_EQ(left_cg->files[j].path, right_cg->files[j].path);
        ASSERT_EQ(left_cg->files[j].start_index, right_cg->files[j].start_index);
        ASSERT_EQ(left_cg->files[j].end_index, right_cg->files[j].end_index);
        ASSERT_EQ(left_cg->files[j].metadata, right_cg->files[j].metadata);
      }
    }
  }
  column_groups_destroy(ccgs);
}

}  // namespace milvus_storage::test