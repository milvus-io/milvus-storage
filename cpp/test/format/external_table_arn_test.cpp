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

// Integration tests for reading external tables via AWS AssumeRole (ARN).
//
// These tests verify that the storage layer can use an IAM role ARN to
// access data in an external S3 bucket. Test data is written with explicit
// AKSK credentials, then read back using only the role ARN.
//
// Required environment variables (all must be set, otherwise tests are skipped):
#define ARN_ENV_ADDRESS "ARN_TEST_ENV_ADDRESS"          // S3 endpoint (e.g., "s3.us-west-2.amazonaws.com")
#define ARN_ENV_REGION "ARN_TEST_ENV_REGION"            // AWS region (e.g., "us-west-2")
#define ARN_ENV_BUCKET_NAME "ARN_TEST_ENV_BUCKET_NAME"  // Target bucket (e.g., "file-transfering-bucket")
#define ARN_ENV_ACCESS_KEY "ARN_TEST_ENV_ACCESS_KEY"    // AWS access key with write access to the bucket
#define ARN_ENV_SECRET_KEY "ARN_TEST_ENV_SECRET_KEY"    // AWS secret key
#define ARN_ENV_ROLE_ARN "ARN_TEST_ENV_ROLE_ARN"        // IAM role ARN to assume for reading

#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <thread>

#include <arrow/api.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/format/lance/lance_table_writer.h"
#include "milvus-storage/format/iceberg/iceberg_common.h"
#include "milvus-storage/ffi_exttable_c.h"
#include "milvus-storage/manifest.h"
#include "lance_bridge.h"
#include "iceberg_bridge.h"
#include "test_env.h"

namespace milvus_storage {

static void ValidateRowGroupInfos(const std::vector<RowGroupInfo>& rg_infos, uint64_t expected_logical_rows) {
  ASSERT_FALSE(rg_infos.empty());
  ASSERT_EQ(rg_infos.front().start_offset, 0u);
  for (size_t i = 1; i < rg_infos.size(); ++i) {
    ASSERT_EQ(rg_infos[i].start_offset, rg_infos[i - 1].end_offset)
        << "Row group " << i << " start_offset is not contiguous with previous end_offset";
  }
  ASSERT_EQ(rg_infos.back().end_offset, expected_logical_rows);
}

struct ArnWriteResult {
  api::ColumnGroupFile cgfile;
  std::shared_ptr<arrow::Schema> schema;  // nullptr for Iceberg
  uint64_t num_rows;
  std::string explore_dir;      // Full URI with address for loon_exttable_explore
  int64_t iceberg_snapshot_id;  // Only used for iceberg
};

class ExternalTableArnTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    // All ARN_TEST_ENV_* env vars are required — skip if any is missing.
    address_ = GetEnvVar(ARN_ENV_ADDRESS).ValueOr("");
    region_ = GetEnvVar(ARN_ENV_REGION).ValueOr("");
    arn_bucket_ = GetEnvVar(ARN_ENV_BUCKET_NAME).ValueOr("");
    arn_ak_ = GetEnvVar(ARN_ENV_ACCESS_KEY).ValueOr("");
    arn_sk_ = GetEnvVar(ARN_ENV_SECRET_KEY).ValueOr("");
    role_arn_ = GetEnvVar(ARN_ENV_ROLE_ARN).ValueOr("");

    if (address_.empty() || region_.empty() || arn_bucket_.empty() || arn_ak_.empty() || arn_sk_.empty() ||
        role_arn_.empty()) {
      GTEST_SKIP() << "ARN tests require all env vars: " << ARN_ENV_ADDRESS << ", " << ARN_ENV_REGION << ", "
                   << ARN_ENV_BUCKET_NAME << ", " << ARN_ENV_ACCESS_KEY << ", " << ARN_ENV_SECRET_KEY << ", "
                   << ARN_ENV_ROLE_ARN;
    }

    // --- Write properties: AKSK credentials for writing test data ---
    api::SetValue(write_props_, PROPERTY_FS_STORAGE_TYPE, "remote");
    api::SetValue(write_props_, PROPERTY_FS_CLOUD_PROVIDER, "aws");
    api::SetValue(write_props_, PROPERTY_FS_ADDRESS, address_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_BUCKET_NAME, arn_bucket_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_REGION, region_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_ACCESS_KEY_ID, arn_ak_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_ACCESS_KEY_VALUE, arn_sk_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_USE_SSL, "true");

    // --- Read properties for Iceberg: extfs.arn.* with role_arn ---
    // Iceberg reader resolves URI through extfs.* properties.
    api::SetValue(read_props_, "extfs.arn.storage_type", "remote");
    api::SetValue(read_props_, "extfs.arn.cloud_provider", "aws");
    api::SetValue(read_props_, "extfs.arn.address", address_.c_str());
    api::SetValue(read_props_, "extfs.arn.bucket_name", arn_bucket_.c_str());
    api::SetValue(read_props_, "extfs.arn.region", region_.c_str());
    api::SetValue(read_props_, "extfs.arn.use_ssl", "true");
    api::SetValue(read_props_, "extfs.arn.role_arn", role_arn_.c_str());

    // --- Read properties for Lance: fs.* with role_arn ---
    // Lance reader reads fs.* directly (not extfs.*).
    api::SetValue(lance_read_props_, PROPERTY_FS_STORAGE_TYPE, "remote");
    api::SetValue(lance_read_props_, PROPERTY_FS_CLOUD_PROVIDER, "aws");
    api::SetValue(lance_read_props_, PROPERTY_FS_ADDRESS, address_.c_str());
    api::SetValue(lance_read_props_, PROPERTY_FS_BUCKET_NAME, arn_bucket_.c_str());
    api::SetValue(lance_read_props_, PROPERTY_FS_REGION, region_.c_str());
    api::SetValue(lance_read_props_, PROPERTY_FS_USE_SSL, "true");
    api::SetValue(lance_read_props_, PROPERTY_FS_ROLE_ARN, role_arn_.c_str());

    // Create write filesystem for cleanup
    ASSERT_AND_ASSIGN(write_fs_, GetFileSystem(write_props_));

    FilesystemCache::getInstance().clean();

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "zc/arn-test-" + std::to_string(ts);
  }

  void TearDown() override {
    if (write_fs_) {
      (void)DeleteTestDir(write_fs_, test_base_);
    }
    FilesystemCache::getInstance().clean();
  }

  const api::Properties& ReadPropsFor(const std::string& format) const {
    if (format == LOON_FORMAT_LANCE_TABLE) {
      return lance_read_props_;
    }
    return read_props_;
  }

  arrow::Result<ArnWriteResult> CreateTestTable(const std::string& format, uint64_t num_rows) {
    if (format == LOON_FORMAT_LANCE_TABLE) {
      return CreateLanceTable(num_rows);
    } else if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return CreateIcebergTable(num_rows);
    }
    return arrow::Status::Invalid("Unknown format: " + format);
  }

  std::string address_;
  std::string region_;
  std::string arn_bucket_;
  std::string arn_ak_;
  std::string arn_sk_;
  std::string role_arn_;

  api::Properties write_props_;
  api::Properties read_props_;
  api::Properties lance_read_props_;
  ArrowFileSystemPtr write_fs_;
  std::string test_base_;

  private:
  arrow::Result<ArnWriteResult> CreateLanceTable(uint64_t num_rows) {
    ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
    ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, 0, false, num_rows, 4, 50, {true, true, true, false}));
    auto path = test_base_ + "/lance";
    lance::LanceTableWriter writer(path, schema, write_props_);
    ARROW_RETURN_NOT_OK(writer.Write(batch));
    ARROW_ASSIGN_OR_RAISE(auto cgfile, writer.Close());
    // explore_dir: s3://address/bucket/path (with address for extfs matching)
    auto explore_dir = "s3://" + address_ + "/" + arn_bucket_ + "/" + path;
    return ArnWriteResult{std::move(cgfile), schema, num_rows, explore_dir, 0};
  }

  arrow::Result<ArnWriteResult> CreateIcebergTable(uint64_t num_rows) {
    auto path = test_base_ + "/iceberg";
    auto table_uri = "s3://" + arn_bucket_ + "/" + path;

    ArrowFileSystemConfig write_config;
    ARROW_RETURN_NOT_OK(ArrowFileSystemConfig::create_file_system_config(write_props_, write_config));
    auto storage_options = iceberg::ToStorageOptions(write_config);

    auto table_info = iceberg::CreateTestTable(table_uri, num_rows, false, {}, storage_options);
    auto file_infos = iceberg::PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
    if (file_infos.empty()) {
      return arrow::Status::Invalid("PlanFiles returned no files");
    }

    auto milvus_path = iceberg::ToMilvusUri(file_infos[0].data_file_path, address_);
    api::ColumnGroupFile cg_file{milvus_path, 0, static_cast<int64_t>(file_infos[0].record_count), {}};
    // explore_dir for iceberg: metadata location converted to milvus URI format (with address)
    auto explore_dir = iceberg::ToMilvusUri(table_info.metadata_location, address_);
    return ArnWriteResult{std::move(cg_file), nullptr, num_rows, explore_dir, table_info.snapshot_id};
  }
};

TEST_P(ExternalTableArnTest, ReadWithArnRole) {
  const auto& format = GetParam();
  const uint64_t num_rows = 100;

  // Step 1: Write test data using AKSK credentials
  ASSERT_AND_ASSIGN(auto result, CreateTestTable(format, num_rows));

  std::cout << "[ARN Test] Format: " << format << std::endl;
  std::cout << "[ARN Test] Written to: " << result.cgfile.path << std::endl;
  std::cout << "[ARN Test] Explore dir: " << result.explore_dir << std::endl;
  std::cout << "[ARN Test] Role ARN: " << role_arn_ << std::endl;

  // Step 2: Build properties for loon_exttable_explore
  //   - fs.*: local filesystem for writing manifest (base_path)
  //   - extfs.arn.*: ARN role for reading external data (explore_dir)
  auto local_base =
      "/tmp/arn-test-manifest-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

  std::vector<std::string> prop_keys, prop_values;
  // Default fs: local for manifest storage
  prop_keys.push_back(PROPERTY_FS_STORAGE_TYPE);
  prop_values.push_back("local");
  prop_keys.push_back(PROPERTY_FS_ROOT_PATH);
  prop_values.push_back(local_base);
  // extfs.arn: ARN role for external data access
  prop_keys.push_back("extfs.arn.storage_type");
  prop_values.push_back("remote");
  prop_keys.push_back("extfs.arn.cloud_provider");
  prop_values.push_back("aws");
  prop_keys.push_back("extfs.arn.address");
  prop_values.push_back(address_);
  prop_keys.push_back("extfs.arn.bucket_name");
  prop_values.push_back(arn_bucket_);
  prop_keys.push_back("extfs.arn.region");
  prop_values.push_back(region_);
  prop_keys.push_back("extfs.arn.use_ssl");
  prop_values.push_back("true");
  prop_keys.push_back("extfs.arn.role_arn");
  prop_values.push_back(role_arn_);
  // Iceberg needs snapshot_id
  if (format == LOON_FORMAT_ICEBERG_TABLE) {
    prop_keys.push_back(PROPERTY_ICEBERG_SNAPSHOT_ID);
    prop_values.push_back(std::to_string(result.iceberg_snapshot_id));
  }

  // Build C arrays for FFI
  std::vector<const char*> c_keys, c_values;
  for (size_t i = 0; i < prop_keys.size(); ++i) {
    c_keys.push_back(prop_keys[i].c_str());
    c_values.push_back(prop_values[i].c_str());
  }

  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  // Step 3: Call loon_exttable_explore — discovers files via ARN role, writes manifest locally
  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;

  rc = loon_exttable_explore(columns_arr, 3, format.c_str(), local_base.c_str(), result.explore_dir.c_str(),
                             &loon_props, &out_num_files, &out_manifest_path);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_GT(out_num_files, 0u);
  ASSERT_NE(out_manifest_path, nullptr);

  std::cout << "[ARN Test] loon_exttable_explore: found " << out_num_files << " files, manifest=" << out_manifest_path
            << std::endl;

  // Step 4: Read manifest via FFI to get ColumnGroupFiles
  LoonManifest* out_manifest = nullptr;
  rc = loon_exttable_read_manifest(out_manifest_path, &loon_props, &out_manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(out_manifest, nullptr);
  ASSERT_EQ(out_manifest->column_groups.num_of_column_groups, 1u);

  auto* cg = &out_manifest->column_groups.column_group_array[0];
  ASSERT_EQ(cg->num_of_files, out_num_files);

  std::cout << "[ARN Test] manifest has " << cg->num_of_files << " files" << std::endl;

  // Step 5: Read data using FormatReader with ARN role
  const auto& read_props = ReadPropsFor(format);
  std::vector<std::string> columns = {"id", "name", "value"};

  int64_t total_rows = 0;
  for (uint64_t f = 0; f < cg->num_of_files; ++f) {
    auto& loon_file = cg->files[f];
    api::ColumnGroupFile cgfile;
    cgfile.path = loon_file.path;
    cgfile.start_index = loon_file.start_index;
    cgfile.end_index = loon_file.end_index;
    // Copy file properties (e.g., iceberg delete metadata)
    if (loon_file.property_keys != nullptr) {
      for (uint32_t p = 0; p < loon_file.num_properties; ++p) {
        cgfile.properties[loon_file.property_keys[p]] = loon_file.property_values[p];
      }
    }

    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(result.schema, format, cgfile, read_props, columns, nullptr));
    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());

    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total_rows += batch->num_rows();
    }
  }
  ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));
  std::cout << "[ARN Test] FormatReader read " << total_rows << " rows via ARN OK" << std::endl;

  // Cleanup
  loon_manifest_destroy(out_manifest);
  free(out_manifest_path);
  loon_properties_free(&loon_props);
}

INSTANTIATE_TEST_SUITE_P(ArnFormats,
                         ExternalTableArnTest,
                         ::testing::Values(LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE));

// ---------------------------------------------------------------------------
// Lance-only: verify credential refresh works with short load_frequency.
// Writes data with AKSK, reads twice with ARN role (load_frequency=2s),
// sleeping 3s between reads to force a credential refresh cycle.
// ---------------------------------------------------------------------------
TEST_F(ExternalTableArnTest, LanceCredentialRefresh) {
  GTEST_SKIP() << "This test sleeps 905s to verify credential refresh. Run manually when needed.";
  const uint64_t num_rows = 50;

  // Write test data using AKSK
  auto write_res = CreateTestTable(LOON_FORMAT_LANCE_TABLE, num_rows);
  ASSERT_STATUS_OK(write_res.status());
  auto result = std::move(write_res).ValueOrDie();

  // Build lance read props with load_frequency=900s (AWS STS minimum)
  api::Properties props = lance_read_props_;
  api::SetValue(props, PROPERTY_FS_LOAD_FREQUENCY, "900");

  std::vector<std::string> columns = {"id", "name", "value"};

  // --- First read ---
  {
    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(result.schema, LOON_FORMAT_LANCE_TABLE, result.cgfile, props,
                                                        columns, nullptr));
    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());

    int64_t total = 0;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total += batch->num_rows();
    }
    ASSERT_EQ(total, static_cast<int64_t>(num_rows));
    std::cout << "[Credential Refresh] first read: " << total << " rows OK" << std::endl;
  }

  // Sleep 905s — exceeds the 900s load_frequency, forces credential refresh
  std::cout << "[Credential Refresh] sleeping 905s to trigger credential refresh..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(905));

  // --- Second read (after credential refresh) ---
  {
    FilesystemCache::getInstance().clean();
    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(result.schema, LOON_FORMAT_LANCE_TABLE, result.cgfile, props,
                                                        columns, nullptr));
    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());

    int64_t total = 0;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total += batch->num_rows();
    }
    ASSERT_EQ(total, static_cast<int64_t>(num_rows));
    std::cout << "[Credential Refresh] second read: " << total << " rows OK" << std::endl;
  }
}

}  // namespace milvus_storage
