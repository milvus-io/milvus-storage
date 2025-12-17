
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
      .path = "test_file1", .start_index = 0, .end_index = 10, .private_data = std::vector<uint8_t>({1, 2, 3, 4})};

  ColumnGroupFile cgf2{
      .path = "test_file2",
      .start_index = 2000,
      .end_index = 10000,
      .private_data = std::nullopt,
  };

  ColumnGroupFile cgf3{
      .path = "test_file2",
      .start_index = 100,
      .end_index = 200,
      .private_data = std::vector<uint8_t>({1, 3, 4, 5}),
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

  ASSERT_STATUS_OK(cgs.add_column_group(std::make_shared<ColumnGroup>(cg1)));
  ASSERT_STATUS_OK(cgs.add_column_group(std::make_shared<ColumnGroup>(cg2)));
  ASSERT_STATUS_OK(cgs.add_metadatas({"key1", "key2"}, {"value1", "value2"}));

  // test export and import
  CColumnGroups ccgs;
  ASSERT_STATUS_OK(export_column_groups(&cgs, &ccgs));

  // verify export C struct
  {
    ASSERT_NE(ccgs.column_group_array, nullptr);
    ASSERT_EQ(ccgs.num_of_column_groups, 2);

    ASSERT_NE(ccgs.meta_keys, nullptr);
    ASSERT_EQ(ccgs.meta_len, 2);
    ASSERT_TRUE(strcmp(ccgs.meta_keys[0], "key1") == 0);
    ASSERT_TRUE(strcmp(ccgs.meta_keys[1], "key2") == 0);
    ASSERT_TRUE(strcmp(ccgs.meta_values[0], "value1") == 0);
    ASSERT_TRUE(strcmp(ccgs.meta_values[1], "value2") == 0);

    ASSERT_NE(ccgs.release, nullptr);
    ASSERT_NE(ccgs.private_data, nullptr);

    ASSERT_NE(ccgs.column_group_array[0].columns, nullptr);
    ASSERT_EQ(ccgs.column_group_array[0].num_of_columns, 2);
    ASSERT_EQ(ccgs.column_group_array[0].columns[0], cg1.columns[0]);
    ASSERT_EQ(ccgs.column_group_array[0].columns[1], cg1.columns[1]);

    ASSERT_EQ(ccgs.column_group_array[0].format, cg1.format);
    ASSERT_NE(ccgs.column_group_array[0].files, nullptr);
    ASSERT_EQ(ccgs.column_group_array[0].num_of_files, 2);
    ASSERT_EQ(ccgs.column_group_array[0].files[0].path, cg1.files[0].path);
    ASSERT_EQ(ccgs.column_group_array[0].files[0].start_index, cg1.files[0].start_index);
    ASSERT_EQ(ccgs.column_group_array[0].files[0].end_index, cg1.files[0].end_index);
    ASSERT_EQ(std::vector<uint8_t>(ccgs.column_group_array[0].files[0].private_data,
                                   ccgs.column_group_array[0].files[0].private_data +
                                       ccgs.column_group_array[0].files[0].private_data_size),
              cg1.files[0].private_data.value());

    ASSERT_EQ(ccgs.column_group_array[0].files[1].path, cg1.files[1].path);
    ASSERT_EQ(ccgs.column_group_array[0].files[1].start_index, cg1.files[1].start_index);
    ASSERT_EQ(ccgs.column_group_array[0].files[1].end_index, cg1.files[1].end_index);
    ASSERT_EQ(ccgs.column_group_array[0].files[1].private_data, nullptr);

    ASSERT_NE(ccgs.column_group_array[1].columns, nullptr);
    ASSERT_EQ(ccgs.column_group_array[1].num_of_columns, 1);
    ASSERT_EQ(ccgs.column_group_array[1].columns[0], cg2.columns[0]);

    ASSERT_EQ(ccgs.column_group_array[1].format, cg2.format);
    ASSERT_NE(ccgs.column_group_array[1].files, nullptr);
    ASSERT_EQ(ccgs.column_group_array[1].num_of_files, 1);
    ASSERT_EQ(ccgs.column_group_array[1].files[0].path, cgf3.path);
    ASSERT_EQ(ccgs.column_group_array[1].files[0].start_index, cgf3.start_index);
    ASSERT_EQ(ccgs.column_group_array[1].files[0].end_index, cgf3.end_index);
    ASSERT_EQ(std::vector<uint8_t>(ccgs.column_group_array[1].files[0].private_data,
                                   ccgs.column_group_array[1].files[0].private_data +
                                       ccgs.column_group_array[1].files[0].private_data_size),
              cgf3.private_data.value());
  }

  ColumnGroups imported_cgs;
  ASSERT_STATUS_OK(import_column_groups(&ccgs, &imported_cgs));

  // verify import column groups should same with origin one
  {
    ASSERT_EQ(cgs.size(), imported_cgs.size());
    for (size_t i = 0; i < cgs.size(); ++i) {
      auto left_cg = cgs.get_column_group(i);
      auto right_cg = imported_cgs.get_column_group(i);

      ASSERT_EQ(left_cg->columns, right_cg->columns);
      ASSERT_EQ(left_cg->format, right_cg->format);
      ASSERT_EQ(left_cg->files.size(), right_cg->files.size());
      for (size_t j = 0; j < left_cg->files.size(); ++j) {
        ASSERT_EQ(left_cg->files[j].path, right_cg->files[j].path);
        ASSERT_EQ(left_cg->files[j].start_index, right_cg->files[j].start_index);
        ASSERT_EQ(left_cg->files[j].end_index, right_cg->files[j].end_index);
        ASSERT_EQ(left_cg->files[j].private_data, right_cg->files[j].private_data);
      }
    }

    ASSERT_EQ(cgs.meta_size(), imported_cgs.meta_size());
    for (size_t i = 0; i < cgs.meta_size(); ++i) {
      ASSERT_AND_ASSIGN(auto leftmeta, cgs.get_metadata(i));
      ASSERT_AND_ASSIGN(auto rightmeta, imported_cgs.get_metadata(i));
      ASSERT_EQ(leftmeta, rightmeta);
    }
  }

  ccgs.release(&ccgs);
}

}  // namespace milvus_storage::test