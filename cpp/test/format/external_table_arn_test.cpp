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
//
// Our-side cloud bucket (IAM-based, for writing manifest):
#define OUR_ENV_ADDRESS "OUR_TEST_ENV_ADDRESS"                // Endpoint (e.g., "s3.us-west-2.amazonaws.com")
#define OUR_ENV_BUCKET "OUR_TEST_ENV_BUCKET"                  // Our bucket (e.g., "zilliz-temp-back-uat")
#define OUR_ENV_REGION "OUR_TEST_ENV_REGION"                  // Region (e.g., "us-west-2")
#define OUR_ENV_CLOUD_PROVIDER "OUR_TEST_ENV_CLOUD_PROVIDER"  // Cloud provider (e.g., "aws", "gcp")
//
// Customer-side S3 bucket (ARN-based, for reading external data):
#define ARN_ENV_ADDRESS "ARN_TEST_ENV_ADDRESS"          // S3 endpoint (e.g., "s3.us-west-2.amazonaws.com")
#define ARN_ENV_REGION "ARN_TEST_ENV_REGION"            // AWS region (e.g., "us-west-2")
#define ARN_ENV_BUCKET_NAME "ARN_TEST_ENV_BUCKET_NAME"  // Target bucket (e.g., "file-transfering-bucket")
#define ARN_ENV_ACCESS_KEY "ARN_TEST_ENV_ACCESS_KEY"    // AWS access key with write access to the bucket
#define ARN_ENV_SECRET_KEY "ARN_TEST_ENV_SECRET_KEY"    // AWS secret key
#define ARN_ENV_ROLE_ARN "ARN_TEST_ENV_ROLE_ARN"        // IAM role ARN to assume for reading

#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <arrow/api.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format.h"
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

static arrow::Result<ArnWriteResult> WritePlainFormatFile(const std::string& format,
                                                          uint64_t num_rows,
                                                          const ArrowFileSystemPtr& write_fs,
                                                          const api::Properties& write_props,
                                                          const std::string& test_base,
                                                          const std::string& uri_scheme,
                                                          const std::string& address,
                                                          const std::string& bucket) {
  if (format != LOON_FORMAT_PARQUET && format != LOON_FORMAT_VORTEX) {
    return arrow::Status::Invalid("Expected a plain file format, got: ", format);
  }

  ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
  ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, 0, false, num_rows, 4, 50, {true, true, true, false}));
  auto dir = test_base + "/" + format;
  auto file_path = dir + "/data." + format;
  ARROW_ASSIGN_OR_RAISE(auto* format_impl, Format::get(format));
  ARROW_ASSIGN_OR_RAISE(auto writer, format_impl->create_writer(write_fs, schema, file_path, dir, write_props));
  ARROW_RETURN_NOT_OK(writer->Write(batch));
  ARROW_ASSIGN_OR_RAISE(auto cgfile, writer->Close());

  auto explore_dir = uri_scheme + "://" + address + "/" + bucket + "/" + dir + "/";
  return ArnWriteResult{std::move(cgfile), std::move(schema), num_rows, std::move(explore_dir), 0};
}

class ExternalTableArnTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    // Our-side bucket (IAM-based, for writing manifest)
    our_address_ = GetEnvVar(OUR_ENV_ADDRESS).ValueOr("");
    our_bucket_ = GetEnvVar(OUR_ENV_BUCKET).ValueOr("");
    our_region_ = GetEnvVar(OUR_ENV_REGION).ValueOr("");
    our_cloud_provider_ = GetEnvVar(OUR_ENV_CLOUD_PROVIDER).ValueOr("");

    // Customer-side S3 bucket (ARN-based)
    address_ = GetEnvVar(ARN_ENV_ADDRESS).ValueOr("");
    region_ = GetEnvVar(ARN_ENV_REGION).ValueOr("");
    arn_bucket_ = GetEnvVar(ARN_ENV_BUCKET_NAME).ValueOr("");
    arn_ak_ = GetEnvVar(ARN_ENV_ACCESS_KEY).ValueOr("");
    arn_sk_ = GetEnvVar(ARN_ENV_SECRET_KEY).ValueOr("");
    role_arn_ = GetEnvVar(ARN_ENV_ROLE_ARN).ValueOr("");

    if (our_address_.empty() || our_bucket_.empty() || our_cloud_provider_.empty() || address_.empty() ||
        region_.empty() || arn_bucket_.empty() || arn_ak_.empty() || arn_sk_.empty() || role_arn_.empty()) {
      GTEST_SKIP() << "ARN tests require all env vars: " << OUR_ENV_ADDRESS << ", " << OUR_ENV_BUCKET << ", "
                   << OUR_ENV_REGION << ", " << OUR_ENV_CLOUD_PROVIDER << ", " << ARN_ENV_ADDRESS << ", "
                   << ARN_ENV_REGION << ", " << ARN_ENV_BUCKET_NAME << ", " << ARN_ENV_ACCESS_KEY << ", "
                   << ARN_ENV_SECRET_KEY << ", " << ARN_ENV_ROLE_ARN;
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

    // --- Read properties: extfs.arn.* with role_arn ---
    api::SetValue(read_props_, "extfs.arn.storage_type", "remote");
    api::SetValue(read_props_, "extfs.arn.cloud_provider", "aws");
    api::SetValue(read_props_, "extfs.arn.address", address_.c_str());
    api::SetValue(read_props_, "extfs.arn.bucket_name", arn_bucket_.c_str());
    api::SetValue(read_props_, "extfs.arn.region", region_.c_str());
    api::SetValue(read_props_, "extfs.arn.use_ssl", "true");
    api::SetValue(read_props_, "extfs.arn.role_arn", role_arn_.c_str());

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

  arrow::Result<ArnWriteResult> CreateTestTable(const std::string& format, uint64_t num_rows) {
    if (format == LOON_FORMAT_LANCE_TABLE) {
      return CreateLanceTable(num_rows);
    } else if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return CreateIcebergTable(num_rows);
    } else if (format == LOON_FORMAT_PARQUET || format == LOON_FORMAT_VORTEX) {
      return WritePlainFormatFile(format, num_rows, write_fs_, write_props_, test_base_, "s3", address_, arn_bucket_);
    }
    return arrow::Status::Invalid("Unknown format: " + format);
  }

  // Our-side
  std::string our_address_;
  std::string our_bucket_;
  std::string our_region_;
  std::string our_cloud_provider_;
  // Customer-side
  std::string address_;
  std::string region_;
  std::string arn_bucket_;
  std::string arn_ak_;
  std::string arn_sk_;
  std::string role_arn_;

  api::Properties write_props_;
  api::Properties read_props_;
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
    std::cout << "[ARN Test] Lance cgfile: " << cgfile.ToString() << std::endl;
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
    std::cout << "[ARN Test] Iceberg cgfile: " << cg_file.ToString() << std::endl;
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
  //   - fs.*: our-side S3 bucket (IAM) for writing manifest
  //   - extfs.arn.*: ARN role for reading external data (explore_dir)
  auto manifest_base = test_base_ + "/manifest";

  std::vector<std::pair<std::string, std::string>> props = {
      // Default fs: our-side bucket with IAM for manifest storage
      {PROPERTY_FS_STORAGE_TYPE, "remote"},
      {PROPERTY_FS_CLOUD_PROVIDER, our_cloud_provider_},
      {PROPERTY_FS_ADDRESS, our_address_},
      {PROPERTY_FS_BUCKET_NAME, our_bucket_},
      {PROPERTY_FS_REGION, our_region_},
      {PROPERTY_FS_USE_SSL, "true"},
      {PROPERTY_FS_USE_IAM, "true"},
      // extfs.arn: ARN role for external data access
      {"extfs.arn.storage_type", "remote"},
      {"extfs.arn.cloud_provider", "aws"},
      {"extfs.arn.address", address_},
      {"extfs.arn.bucket_name", arn_bucket_},
      {"extfs.arn.region", region_},
      {"extfs.arn.use_ssl", "true"},
      {"extfs.arn.role_arn", role_arn_},
  };
  if (format == LOON_FORMAT_ICEBERG_TABLE) {
    props.emplace_back(PROPERTY_ICEBERG_SNAPSHOT_ID, std::to_string(result.iceberg_snapshot_id));
  }

  std::vector<const char*> c_keys, c_values;
  c_keys.reserve(props.size());
  c_values.reserve(props.size());
  for (const auto& [k, v] : props) {
    c_keys.push_back(k.c_str());
    c_values.push_back(v.c_str());
  }

  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  // Step 3: Call loon_exttable_explore — discovers files via ARN role, writes manifest to our S3
  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;

  rc = loon_exttable_explore(columns_arr, 3, format.c_str(), manifest_base.c_str(), result.explore_dir.c_str(),
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

    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(result.schema, format, cgfile, read_props_, columns, nullptr));
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

INSTANTIATE_TEST_SUITE_P(
    ArnFormats,
    ExternalTableArnTest,
    ::testing::Values(LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE, LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX));

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

  // Build read props with load_frequency=900s (AWS STS minimum)
  api::Properties props = read_props_;
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

// ===========================================================================
// Integration tests for reading external tables via GCP Service Account
// Impersonation.
//
// These tests verify that the storage layer can use a target SA email to
// access data in an external GCS bucket. Test data is written with HMAC
// credentials, then read back using only the target SA email.
//
// Required environment variables (all must be set, otherwise tests are skipped):
// ===========================================================================
#define GCP_IMP_ENV_ADDRESS "GCP_IMP_TEST_ENV_ADDRESS"        // GCS S3-compat endpoint (e.g., "storage.googleapis.com")
#define GCP_IMP_ENV_BUCKET "GCP_IMP_TEST_ENV_BUCKET"          // GCS bucket name
#define GCP_IMP_ENV_ACCESS_KEY "GCP_IMP_TEST_ENV_ACCESS_KEY"  // HMAC access key (for write)
#define GCP_IMP_ENV_SECRET_KEY "GCP_IMP_TEST_ENV_SECRET_KEY"  // HMAC secret key (for write)
#define GCP_IMP_ENV_TARGET_SA "GCP_IMP_TEST_ENV_TARGET_SA"    // Target SA email (for read via impersonation)

class ExternalTableGcpImpersonationTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    // Our-side bucket (IAM-based, for writing manifest) — shared with AWS test
    our_address_ = GetEnvVar(OUR_ENV_ADDRESS).ValueOr("");
    our_bucket_ = GetEnvVar(OUR_ENV_BUCKET).ValueOr("");
    our_region_ = GetEnvVar(OUR_ENV_REGION).ValueOr("");
    our_cloud_provider_ = GetEnvVar(OUR_ENV_CLOUD_PROVIDER).ValueOr("");

    // Customer-side GCS bucket
    address_ = GetEnvVar(GCP_IMP_ENV_ADDRESS).ValueOr("");
    bucket_ = GetEnvVar(GCP_IMP_ENV_BUCKET).ValueOr("");
    gcp_ak_ = GetEnvVar(GCP_IMP_ENV_ACCESS_KEY).ValueOr("");
    gcp_sk_ = GetEnvVar(GCP_IMP_ENV_SECRET_KEY).ValueOr("");
    target_sa_ = GetEnvVar(GCP_IMP_ENV_TARGET_SA).ValueOr("");

    if (our_address_.empty() || our_bucket_.empty() || our_cloud_provider_.empty() || address_.empty() ||
        bucket_.empty() || gcp_ak_.empty() || gcp_sk_.empty() || target_sa_.empty()) {
      GTEST_SKIP() << "GCP impersonation tests require all env vars: " << OUR_ENV_ADDRESS << ", " << OUR_ENV_BUCKET
                   << ", " << OUR_ENV_REGION << ", " << OUR_ENV_CLOUD_PROVIDER << ", " << GCP_IMP_ENV_ADDRESS << ", "
                   << GCP_IMP_ENV_BUCKET << ", " << GCP_IMP_ENV_ACCESS_KEY << ", " << GCP_IMP_ENV_SECRET_KEY << ", "
                   << GCP_IMP_ENV_TARGET_SA;
    }

    // --- Write properties: S3-compat mode to write test data to GCS.
    //     opendal's GCS backend doesn't accept HMAC AK/SK (only SA JSON or
    //     impersonation), so we write via cloud_provider=aws pointing at the
    //     GCS S3-compat endpoint — same physical objects, different API door.
    //     Read side (below) still uses native GCS + SA impersonation. ---
    api::SetValue(write_props_, PROPERTY_FS_STORAGE_TYPE, "remote");
    api::SetValue(write_props_, PROPERTY_FS_CLOUD_PROVIDER, "aws");
    api::SetValue(write_props_, PROPERTY_FS_ADDRESS, address_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_BUCKET_NAME, bucket_.c_str());
    // Region is only used as a SigV4 signing input — GCS S3-compat ignores its
    // value but opendal's S3 Builder rejects empty strings, which breaks the
    // iceberg write path (opendal-backed). Use "auto" to satisfy opendal; the
    // AWS SDK (Lance write path) accepts any non-empty region against GCS too.
    api::SetValue(write_props_, PROPERTY_FS_REGION, "auto");
    api::SetValue(write_props_, PROPERTY_FS_ACCESS_KEY_ID, gcp_ak_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_ACCESS_KEY_VALUE, gcp_sk_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_USE_SSL, "true");

    // Parquet and Vortex use the native C++ filesystem. Configure GCP HMAC
    // directly so the process installs the GCP HTTP factory rather than the
    // incompatible AWS factory used by the Rust S3-compatible write path.
    api::SetValue(plain_write_props_, PROPERTY_FS_STORAGE_TYPE, "remote");
    api::SetValue(plain_write_props_, PROPERTY_FS_CLOUD_PROVIDER, kCloudProviderGCP);
    api::SetValue(plain_write_props_, PROPERTY_FS_ADDRESS, address_.c_str());
    api::SetValue(plain_write_props_, PROPERTY_FS_BUCKET_NAME, bucket_.c_str());
    api::SetValue(plain_write_props_, PROPERTY_FS_REGION, "auto");
    api::SetValue(plain_write_props_, PROPERTY_FS_ACCESS_KEY_ID, gcp_ak_.c_str());
    api::SetValue(plain_write_props_, PROPERTY_FS_ACCESS_KEY_VALUE, gcp_sk_.c_str());
    api::SetValue(plain_write_props_, PROPERTY_FS_USE_SSL, "true");

    // --- Read properties: extfs.gcpsa.* with gcp_target_service_account ---
    // use_iam=true is required to reach IamImpersonateProvider on the C++
    // side (gcp_credential_provider.cpp:160); without it the producer falls
    // through to HmacProvider with empty AK/SK → no Authorization header,
    // ACCESS_DENIED on parquet data reads.
    api::SetValue(read_props_, "extfs.gcpsa.storage_type", "remote");
    api::SetValue(read_props_, "extfs.gcpsa.cloud_provider", "gcp");
    api::SetValue(read_props_, "extfs.gcpsa.address", address_.c_str());
    api::SetValue(read_props_, "extfs.gcpsa.bucket_name", bucket_.c_str());
    api::SetValue(read_props_, "extfs.gcpsa.use_ssl", "true");
    api::SetValue(read_props_, "extfs.gcpsa.use_iam", "true");
    api::SetValue(read_props_, "extfs.gcpsa.gcp_target_service_account", target_sa_.c_str());

    FilesystemCache::getInstance().clean();

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "zc/gcp-imp-test-" + std::to_string(ts);
  }

  // No TearDown cleanup: Lance and Iceberg write through the S3-compatible
  // endpoint, while Parquet and Vortex use the native GCP filesystem. Avoid
  // another filesystem lookup after the read path has installed its HTTP
  // factory. Test data is left in the customer bucket; rely on lifecycle rules.
  void TearDown() override { FilesystemCache::getInstance().clean(); }

  arrow::Result<ArnWriteResult> CreateTestTable(const std::string& format, uint64_t num_rows) {
    if (format == LOON_FORMAT_LANCE_TABLE) {
      return CreateLanceTable(num_rows);
    } else if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return CreateIcebergTable(num_rows);
    } else if (format == LOON_FORMAT_PARQUET || format == LOON_FORMAT_VORTEX) {
      ARROW_ASSIGN_OR_RAISE(auto write_fs, GetFileSystem(plain_write_props_));
      return WritePlainFormatFile(format, num_rows, write_fs, plain_write_props_, test_base_, "gs", address_, bucket_);
    }
    return arrow::Status::Invalid("Unknown format: " + format);
  }

  // Our-side
  std::string our_address_;
  std::string our_bucket_;
  std::string our_region_;
  std::string our_cloud_provider_;
  // Customer-side
  std::string address_;
  std::string bucket_;
  std::string gcp_ak_;
  std::string gcp_sk_;
  std::string target_sa_;

  api::Properties write_props_;
  api::Properties plain_write_props_;
  api::Properties read_props_;
  std::string test_base_;

  private:
  arrow::Result<ArnWriteResult> CreateLanceTable(uint64_t num_rows) {
    ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
    ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, 0, false, num_rows, 4, 50, {true, true, true, false}));
    auto path = test_base_ + "/lance";
    lance::LanceTableWriter writer(path, schema, write_props_);
    ARROW_RETURN_NOT_OK(writer.Write(batch));
    ARROW_ASSIGN_OR_RAISE(auto cgfile, writer.Close());
    std::cout << "[GCP Imp Test] Lance cgfile: " << cgfile.ToString() << std::endl;
    // explore_dir: gs://address/bucket/path (with address for extfs matching)
    auto explore_dir = "gs://" + address_ + "/" + bucket_ + "/" + path;
    return ArnWriteResult{std::move(cgfile), schema, num_rows, explore_dir, 0};
  }

  arrow::Result<ArnWriteResult> CreateIcebergTable(uint64_t num_rows) {
    auto path = test_base_ + "/iceberg";
    // Physical write goes via S3-compat (opendal routes s3:// → S3 backend +
    // HMAC, the only HMAC-capable door to GCS). But pass record_scheme_override
    // = "gs" so the Rust side rewrites every embedded URI across the metadata
    // tree after writing — necessary because iceberg-rust bakes the write-time
    // scheme into manifest list / manifest / metadata.json and never swaps it
    // on read. Without this, the read path's `gs://` FileIO would reject the
    // embedded `s3://` manifest-list reference (DataInvalid).
    auto table_uri = "s3://" + bucket_ + "/" + path;

    ArrowFileSystemConfig write_config;
    ARROW_RETURN_NOT_OK(ArrowFileSystemConfig::create_file_system_config(write_props_, write_config));
    auto storage_options = iceberg::ToStorageOptions(write_config);

    auto table_info = iceberg::CreateTestTable(table_uri, num_rows, false, {}, storage_options, "gs");

    auto explore_dir = iceberg::ToMilvusUri(table_info.metadata_location, address_);
    auto milvus_path = iceberg::ToMilvusUri(table_info.data_file_uri, address_);
    api::ColumnGroupFile cg_file{milvus_path, 0, static_cast<int64_t>(num_rows), {}};
    std::cout << "[GCP Imp Test] Iceberg cgfile: " << cg_file.ToString() << std::endl;
    return ArnWriteResult{std::move(cg_file), nullptr, num_rows, explore_dir, table_info.snapshot_id};
  }
};

// End-to-end flow of this test — which props each step uses:
//
//   Step 1  write test data      write_props_           (cloud_provider=aws,
//                                                         HMAC AK/SK; S3-compat
//                                                         to customer bucket)
//   Step 2  build LoonProperties fs.* (our-side IAM) +   (cloud_provider=gcp,
//                                extfs.gcpsa.*            use_iam=true for
//                                                         manifest storage;
//                                                         extfs.gcpsa.* carries
//                                                         target_sa_ for the
//                                                         customer-bucket reads)
//   Step 3  loon_exttable_explore the LoonProperties     (reads customer bucket
//                                from Step 2              via Impersonating-
//                                                         GcsStoreProvider using
//                                                         target_sa_; writes
//                                                         manifest to our-side
//                                                         bucket via VM SA
//                                                         OAuth2 Bearer)
//   Step 4  read back manifest   same LoonProperties     (our-side bucket)
//   Step 5  read data rows       read_props_             (extfs.gcpsa.* only;
//                                                         customer bucket via
//                                                         target SA impersonation,
//                                                         token cached by the
//                                                         Rust credential
//                                                         provider across calls)
TEST_P(ExternalTableGcpImpersonationTest, ReadWithImpersonation) {
  const auto& format = GetParam();
  const uint64_t num_rows = 100;

  // Step 1: Write test data using HMAC credentials
  ASSERT_AND_ASSIGN(auto result, CreateTestTable(format, num_rows));

  std::cout << "[GCP Imp Test] Format: " << format << std::endl;
  std::cout << "[GCP Imp Test] Written to: " << result.cgfile.path << std::endl;
  std::cout << "[GCP Imp Test] Explore dir: " << result.explore_dir << std::endl;
  std::cout << "[GCP Imp Test] Target SA: " << target_sa_ << std::endl;

  // Step 2: Build properties for loon_exttable_explore
  //   - fs.*: write filesystem for writing manifest (base_path)
  //   - extfs.gcpsa.*: SA impersonation for reading external data (explore_dir)
  auto manifest_base = test_base_ + "/manifest";

  std::vector<std::pair<std::string, std::string>> props = {
      // Default fs: our-side bucket with IAM for manifest storage
      {PROPERTY_FS_STORAGE_TYPE, "remote"},
      {PROPERTY_FS_CLOUD_PROVIDER, our_cloud_provider_},
      {PROPERTY_FS_ADDRESS, our_address_},
      {PROPERTY_FS_BUCKET_NAME, our_bucket_},
      {PROPERTY_FS_REGION, our_region_},
      {PROPERTY_FS_USE_SSL, "true"},
      {PROPERTY_FS_USE_IAM, "true"},
      // extfs.gcpsa: SA impersonation for external data access
      {"extfs.gcpsa.storage_type", "remote"},
      {"extfs.gcpsa.cloud_provider", "gcp"},
      {"extfs.gcpsa.address", address_},
      {"extfs.gcpsa.bucket_name", bucket_},
      {"extfs.gcpsa.use_ssl", "true"},
      {"extfs.gcpsa.use_iam", "true"},
      {"extfs.gcpsa.gcp_target_service_account", target_sa_},
  };
  if (format == LOON_FORMAT_ICEBERG_TABLE) {
    props.emplace_back(PROPERTY_ICEBERG_SNAPSHOT_ID, std::to_string(result.iceberg_snapshot_id));
  }

  std::vector<const char*> c_keys, c_values;
  c_keys.reserve(props.size());
  c_values.reserve(props.size());
  for (const auto& [k, v] : props) {
    c_keys.push_back(k.c_str());
    c_values.push_back(v.c_str());
  }

  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  // Step 3: Call loon_exttable_explore — discovers files via SA impersonation, writes manifest to GCS
  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;

  rc = loon_exttable_explore(columns_arr, 3, format.c_str(), manifest_base.c_str(), result.explore_dir.c_str(),
                             &loon_props, &out_num_files, &out_manifest_path);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_GT(out_num_files, 0u);
  ASSERT_NE(out_manifest_path, nullptr);

  std::cout << "[GCP Imp Test] loon_exttable_explore: found " << out_num_files
            << " files, manifest=" << out_manifest_path << std::endl;

  // Step 4: Read manifest via FFI to get ColumnGroupFiles
  LoonManifest* out_manifest = nullptr;
  rc = loon_exttable_read_manifest(out_manifest_path, &loon_props, &out_manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(out_manifest, nullptr);
  ASSERT_EQ(out_manifest->column_groups.num_of_column_groups, 1u);

  auto* cg = &out_manifest->column_groups.column_group_array[0];
  ASSERT_EQ(cg->num_of_files, out_num_files);

  std::cout << "[GCP Imp Test] manifest has " << cg->num_of_files << " files" << std::endl;

  // Step 5: Read data using FormatReader with SA impersonation
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
    std::cout << "[GCP Imp Test] reading cgfile[" << f << "]: " << cgfile.ToString() << std::endl;

    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(result.schema, format, cgfile, read_props_, columns, nullptr));
    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());

    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total_rows += batch->num_rows();
    }
  }
  ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));
  std::cout << "[GCP Imp Test] FormatReader read " << total_rows << " rows via SA impersonation OK" << std::endl;

  // Cleanup
  loon_manifest_destroy(out_manifest);
  free(out_manifest_path);
  loon_properties_free(&loon_props);
}

INSTANTIATE_TEST_SUITE_P(
    GcpImpersonationFormats,
    ExternalTableGcpImpersonationTest,
    ::testing::Values(LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE, LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX));

// ===========================================================================
// Integration tests for reading external Azure tables via brokered SAS tokens.
//
// Test data is written, discovered, and read using only the Azure broker
// configuration. All four external-table formats are covered, exercising both
// the Rust broker path and the C++ Azure token policy.
//
// Required environment variables (all must be set, otherwise tests are skipped):
// ===========================================================================
#define AZURE_ARN_ENV_ADDRESS "AZURE_ARN_TEST_ENV_ADDRESS"  // Endpoint suffix (e.g. "core.windows.net")
#define AZURE_ARN_ENV_REGION "AZURE_ARN_TEST_ENV_REGION"
#define AZURE_ARN_ENV_CONTAINER "AZURE_ARN_TEST_ENV_CONTAINER"
#define AZURE_ARN_ENV_ACCOUNT_NAME "AZURE_ARN_TEST_ENV_ACCOUNT_NAME"
#define AZURE_ARN_ENV_CLIENT_ID "AZURE_ARN_TEST_ENV_CLIENT_ID"
#define AZURE_ARN_ENV_TENANT_ID "AZURE_ARN_TEST_ENV_TENANT_ID"
#define AZURE_ARN_ENV_CREDENTIAL_ENDPOINT "AZURE_ARN_TEST_ENV_CREDENTIAL_ENDPOINT"

class ExternalTableAzureArnTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    // Customer-side Azure storage account.
    address_ = GetEnvVar(AZURE_ARN_ENV_ADDRESS).ValueOr("");
    region_ = GetEnvVar(AZURE_ARN_ENV_REGION).ValueOr("");
    container_ = GetEnvVar(AZURE_ARN_ENV_CONTAINER).ValueOr("");
    account_name_ = GetEnvVar(AZURE_ARN_ENV_ACCOUNT_NAME).ValueOr("");
    client_id_ = GetEnvVar(AZURE_ARN_ENV_CLIENT_ID).ValueOr("");
    tenant_id_ = GetEnvVar(AZURE_ARN_ENV_TENANT_ID).ValueOr("");
    credential_endpoint_ = GetEnvVar(AZURE_ARN_ENV_CREDENTIAL_ENDPOINT).ValueOr("");

    if (address_.empty() || region_.empty() || container_.empty() || account_name_.empty() || client_id_.empty() ||
        tenant_id_.empty() || credential_endpoint_.empty()) {
      GTEST_SKIP() << "Azure ARN tests require all env vars: " << AZURE_ARN_ENV_ADDRESS << ", " << AZURE_ARN_ENV_REGION
                   << ", " << AZURE_ARN_ENV_CONTAINER << ", " << AZURE_ARN_ENV_ACCOUNT_NAME << ", "
                   << AZURE_ARN_ENV_CLIENT_ID << ", " << AZURE_ARN_ENV_TENANT_ID << ", "
                   << AZURE_ARN_ENV_CREDENTIAL_ENDPOINT;
    }

    // The broker-issued SAS includes write/delete permissions, so test setup and
    // cleanup exercise the same authentication path as the read under test.
    api::SetValue(write_props_, PROPERTY_FS_STORAGE_TYPE, "remote");
    api::SetValue(write_props_, PROPERTY_FS_CLOUD_PROVIDER, kCloudProviderAzure);
    api::SetValue(write_props_, PROPERTY_FS_ADDRESS, address_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_BUCKET_NAME, container_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_REGION, region_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_ACCESS_KEY_ID, account_name_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_AZURE_CLIENT_ID, client_id_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_AZURE_TENANT_ID, tenant_id_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_AZURE_CREDENTIAL_ENDPOINT, credential_endpoint_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_LOAD_FREQUENCY, "3600");
    api::SetValue(write_props_, PROPERTY_FS_REQUEST_TIMEOUT_MS, "10000");
    api::SetValue(write_props_, PROPERTY_FS_USE_SSL, "true");

    // Read properties: broker configuration only. No account key is forwarded.
    api::SetValue(read_props_, "extfs.azsas.storage_type", "remote");
    api::SetValue(read_props_, "extfs.azsas.cloud_provider", kCloudProviderAzure);
    api::SetValue(read_props_, "extfs.azsas.address", address_.c_str());
    api::SetValue(read_props_, "extfs.azsas.bucket_name", container_.c_str());
    api::SetValue(read_props_, "extfs.azsas.region", region_.c_str());
    api::SetValue(read_props_, "extfs.azsas.access_key_id", account_name_.c_str());
    api::SetValue(read_props_, "extfs.azsas.azure_client_id", client_id_.c_str());
    api::SetValue(read_props_, "extfs.azsas.azure_tenant_id", tenant_id_.c_str());
    api::SetValue(read_props_, "extfs.azsas.azure_credential_endpoint", credential_endpoint_.c_str());
    api::SetValue(read_props_, "extfs.azsas.load_frequency", "3600");
    api::SetValue(read_props_, "extfs.azsas.request_timeout_ms", "10000");
    api::SetValue(read_props_, "extfs.azsas.use_ssl", "true");

    FilesystemCache::getInstance().clean();

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "zc/azure-arn-test-" + std::to_string(ts);
    manifest_root_ = "/tmp/milvus-storage-azure-arn-test";

    api::SetValue(manifest_props_, PROPERTY_FS_STORAGE_TYPE, "local");
    api::SetValue(manifest_props_, PROPERTY_FS_ROOT_PATH, manifest_root_.c_str());

    ASSERT_AND_ASSIGN(write_fs_, GetFileSystem(write_props_));
    ASSERT_AND_ASSIGN(manifest_fs_, GetFileSystem(manifest_props_));
  }

  void TearDown() override {
    if (write_fs_) {
      (void)DeleteTestDir(write_fs_, test_base_);
    }
    if (manifest_fs_) {
      (void)DeleteTestDir(manifest_fs_, test_base_);
    }
    FilesystemCache::getInstance().clean();
  }

  arrow::Result<ArnWriteResult> CreateTestTable(const std::string& format, uint64_t num_rows) {
    if (format == LOON_FORMAT_LANCE_TABLE) {
      return CreateLanceTable(num_rows);
    } else if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return CreateIcebergTable(num_rows);
    } else if (format == LOON_FORMAT_PARQUET || format == LOON_FORMAT_VORTEX) {
      return WritePlainFormatFile(format, num_rows, write_fs_, write_props_, test_base_, "az", address_, container_);
    }
    return arrow::Status::Invalid("Unknown format: " + format);
  }

  std::string address_;
  std::string region_;
  std::string container_;
  std::string account_name_;
  std::string client_id_;
  std::string tenant_id_;
  std::string credential_endpoint_;

  api::Properties write_props_;
  api::Properties read_props_;
  api::Properties manifest_props_;
  ArrowFileSystemPtr write_fs_;
  ArrowFileSystemPtr manifest_fs_;
  std::string test_base_;
  std::string manifest_root_;

  private:
  arrow::Result<ArnWriteResult> CreateLanceTable(uint64_t num_rows) {
    ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
    ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, 0, false, num_rows, 4, 50, {true, true, true, false}));
    auto path = test_base_ + "/lance";
    lance::LanceTableWriter writer(path, schema, write_props_);
    ARROW_RETURN_NOT_OK(writer.Write(batch));
    ARROW_ASSIGN_OR_RAISE(auto cgfile, writer.Close());
    std::cout << "[Azure ARN Test] Lance cgfile: " << cgfile.ToString() << std::endl;
    auto explore_dir = "az://" + address_ + "/" + container_ + "/" + path;
    return ArnWriteResult{std::move(cgfile), schema, num_rows, explore_dir, 0};
  }

  arrow::Result<ArnWriteResult> CreateIcebergTable(uint64_t num_rows) {
    auto path = test_base_ + "/iceberg";
    auto table_uri = "abfss://" + container_ + "/" + path;

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
    std::cout << "[Azure ARN Test] Iceberg cgfile: " << cg_file.ToString() << std::endl;
    auto explore_dir = iceberg::ToMilvusUri(table_info.metadata_location, address_);
    return ArnWriteResult{std::move(cg_file), nullptr, num_rows, explore_dir, table_info.snapshot_id};
  }
};

TEST_P(ExternalTableAzureArnTest, ReadWithBrokeredSas) {
  const auto& format = GetParam();
  const uint64_t num_rows = 100;

  // Step 1: Write test data using a brokered SAS token.
  ASSERT_AND_ASSIGN(auto result, CreateTestTable(format, num_rows));

  std::cout << "[Azure ARN Test] Format: " << format << std::endl;
  std::cout << "[Azure ARN Test] Written to: " << result.cgfile.path << std::endl;
  std::cout << "[Azure ARN Test] Explore dir: " << result.explore_dir << std::endl;

  // Step 2: Build properties for loon_exttable_explore.
  auto manifest_base = test_base_ + "/manifest";
  std::vector<std::pair<std::string, std::string>> props = {
      // The manifest is local; only external table access goes through Azure.
      {PROPERTY_FS_STORAGE_TYPE, "local"},
      {PROPERTY_FS_ROOT_PATH, manifest_root_},
      // extfs.azsas: brokered SAS access to the customer container.
      {"extfs.azsas.storage_type", "remote"},
      {"extfs.azsas.cloud_provider", kCloudProviderAzure},
      {"extfs.azsas.address", address_},
      {"extfs.azsas.bucket_name", container_},
      {"extfs.azsas.region", region_},
      {"extfs.azsas.access_key_id", account_name_},
      {"extfs.azsas.azure_client_id", client_id_},
      {"extfs.azsas.azure_tenant_id", tenant_id_},
      {"extfs.azsas.azure_credential_endpoint", credential_endpoint_},
      {"extfs.azsas.load_frequency", "3600"},
      {"extfs.azsas.request_timeout_ms", "10000"},
      {"extfs.azsas.use_ssl", "true"},
  };
  if (format == LOON_FORMAT_ICEBERG_TABLE) {
    props.emplace_back(PROPERTY_ICEBERG_SNAPSHOT_ID, std::to_string(result.iceberg_snapshot_id));
  }

  std::vector<const char*> c_keys, c_values;
  c_keys.reserve(props.size());
  c_values.reserve(props.size());
  for (const auto& [k, v] : props) {
    c_keys.push_back(k.c_str());
    c_values.push_back(v.c_str());
  }

  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  // Step 3: Discover files through the brokered SAS and write the manifest.
  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;

  rc = loon_exttable_explore(columns_arr, 3, format.c_str(), manifest_base.c_str(), result.explore_dir.c_str(),
                             &loon_props, &out_num_files, &out_manifest_path);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_GT(out_num_files, 0u);
  ASSERT_NE(out_manifest_path, nullptr);

  std::cout << "[Azure ARN Test] loon_exttable_explore: found " << out_num_files
            << " files, manifest=" << out_manifest_path << std::endl;

  // Step 4: Read the generated manifest.
  LoonManifest* out_manifest = nullptr;
  rc = loon_exttable_read_manifest(out_manifest_path, &loon_props, &out_manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(out_manifest, nullptr);
  ASSERT_EQ(out_manifest->column_groups.num_of_column_groups, 1u);

  auto* cg = &out_manifest->column_groups.column_group_array[0];
  ASSERT_EQ(cg->num_of_files, out_num_files);

  // Step 5: Read the discovered data using the broker configuration.
  std::vector<std::string> columns = {"id", "name", "value"};
  int64_t total_rows = 0;
  for (uint64_t f = 0; f < cg->num_of_files; ++f) {
    auto& loon_file = cg->files[f];
    api::ColumnGroupFile cgfile;
    cgfile.path = loon_file.path;
    cgfile.start_index = loon_file.start_index;
    cgfile.end_index = loon_file.end_index;
    if (loon_file.property_keys != nullptr) {
      for (uint32_t p = 0; p < loon_file.num_properties; ++p) {
        cgfile.properties[loon_file.property_keys[p]] = loon_file.property_values[p];
      }
    }

    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(result.schema, format, cgfile, read_props_, columns, nullptr));
    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total_rows += batch->num_rows();
    }
  }
  ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));
  std::cout << "[Azure ARN Test] FormatReader read " << total_rows << " rows via brokered SAS OK" << std::endl;

  loon_manifest_destroy(out_manifest);
  free(out_manifest_path);
  loon_properties_free(&loon_props);
}

INSTANTIATE_TEST_SUITE_P(
    AzureArnFormats,
    ExternalTableAzureArnTest,
    ::testing::Values(LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE, LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX));

// ===========================================================================
// Integration test for reading external OSS tables via Aliyun
// AssumeRoleWithOIDC.
//
// Exercises both halves of the Aliyun role_arn feature end-to-end:
//   * C++ native S3FS path — our-side manifest storage through
//     S3FileSystemProducer's Aliyun ARN dispatch
//     (s3_filesystem_producer.cpp), only reached if our_cloud_provider=aliyun.
//   * Rust Lance bridge path — customer-side Lance data read through
//     AliyunOssStoreProvider + opendal + reqsign
//     (aliyun_oss_provider.rs + lance_common.cpp's role_arn branch).
//
// Test data is written with explicit AKSK (cloud_provider=aliyun + AK/SK),
// then read back using only the role_arn. This mirrors the AWS ARN test.
//
// Iceberg now works through the bridge-local `AliyunOssStorageFactory`
// registered in `iceberg_bridgeimpl.rs::upstream_opendal_factory` — stock
// iceberg-storage-opendal's `OssConfig` still drops `role_arn` /
// `security_token` / `oidc-*`, which is why we route `oss://` to our own
// `Storage` impl instead of `OpenDalStorageFactory::Oss`.
//
// Required environment variables (all must be set; test is skipped otherwise):
//
// Our-side OSS bucket (for writing manifest). Unlike the AWS fixture, this
// uses static AK/SK rather than `use_iam=true`: our Aliyun credential provider
// implements AssumeRoleWithOIDC (the K8s RAM-for-Service-Account flow), not
// ECS instance RAM role. A vanilla ECS host without the RAM-for-SA env vars
// cannot use `use_iam=true` for the our-side bucket, so the test uses AK/SK
// there. Only the *customer-side* read path exercises the role_arn flow.
//   OUR_TEST_ENV_ADDRESS, OUR_TEST_ENV_BUCKET, OUR_TEST_ENV_REGION,
//   OUR_TEST_ENV_CLOUD_PROVIDER (set to "aliyun"),
//   OUR_TEST_ENV_ACCESS_KEY, OUR_TEST_ENV_SECRET_KEY
//
// Customer-side OSS bucket (ARN-based, for reading external data):
// ===========================================================================
#define OUR_ENV_ACCESS_KEY "OUR_TEST_ENV_ACCESS_KEY"                // AK for our-side bucket (Aliyun only)
#define OUR_ENV_SECRET_KEY "OUR_TEST_ENV_SECRET_KEY"                // SK for our-side bucket (Aliyun only)
#define ALIYUN_ARN_ENV_ADDRESS "ALIYUN_ARN_TEST_ENV_ADDRESS"        // e.g. "oss-cn-hangzhou.aliyuncs.com"
#define ALIYUN_ARN_ENV_REGION "ALIYUN_ARN_TEST_ENV_REGION"          // e.g. "cn-hangzhou"
#define ALIYUN_ARN_ENV_BUCKET "ALIYUN_ARN_TEST_ENV_BUCKET"          // target bucket
#define ALIYUN_ARN_ENV_ACCESS_KEY "ALIYUN_ARN_TEST_ENV_ACCESS_KEY"  // AK with write to bucket
#define ALIYUN_ARN_ENV_SECRET_KEY "ALIYUN_ARN_TEST_ENV_SECRET_KEY"  // SK
#define ALIYUN_ARN_ENV_ROLE_ARN "ALIYUN_ARN_TEST_ENV_ROLE_ARN"      // acs:ram::xxx:role/... to assume

// Machine identity (pod-level, not per-test): the process env MUST also carry
// ALIBABA_CLOUD_OIDC_TOKEN_FILE and ALIBABA_CLOUD_OIDC_PROVIDER_ARN. On an
// Aliyun RAM-for-Service-Account pod these are K8s-injected; we do NOT set
// them here — the dispatch in s3_filesystem_producer.cpp fails fast if they're
// missing, which is the intended safety net.

class ExternalTableAliyunArnTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // Our-side bucket (AK/SK-based; see fixture-level comment for why not IAM).
    our_address_ = GetEnvVar(OUR_ENV_ADDRESS).ValueOr("");
    our_bucket_ = GetEnvVar(OUR_ENV_BUCKET).ValueOr("");
    our_region_ = GetEnvVar(OUR_ENV_REGION).ValueOr("");
    our_cloud_provider_ = GetEnvVar(OUR_ENV_CLOUD_PROVIDER).ValueOr("");
    our_ak_ = GetEnvVar(OUR_ENV_ACCESS_KEY).ValueOr("");
    our_sk_ = GetEnvVar(OUR_ENV_SECRET_KEY).ValueOr("");

    // Customer-side OSS bucket (ARN-based)
    address_ = GetEnvVar(ALIYUN_ARN_ENV_ADDRESS).ValueOr("");
    region_ = GetEnvVar(ALIYUN_ARN_ENV_REGION).ValueOr("");
    arn_bucket_ = GetEnvVar(ALIYUN_ARN_ENV_BUCKET).ValueOr("");
    arn_ak_ = GetEnvVar(ALIYUN_ARN_ENV_ACCESS_KEY).ValueOr("");
    arn_sk_ = GetEnvVar(ALIYUN_ARN_ENV_SECRET_KEY).ValueOr("");
    role_arn_ = GetEnvVar(ALIYUN_ARN_ENV_ROLE_ARN).ValueOr("");

    if (our_address_.empty() || our_bucket_.empty() || our_cloud_provider_.empty() || our_ak_.empty() ||
        our_sk_.empty() || address_.empty() || region_.empty() || arn_bucket_.empty() || arn_ak_.empty() ||
        arn_sk_.empty() || role_arn_.empty()) {
      GTEST_SKIP() << "Aliyun ARN test requires env vars: " << OUR_ENV_ADDRESS << ", " << OUR_ENV_BUCKET << ", "
                   << OUR_ENV_REGION << ", " << OUR_ENV_CLOUD_PROVIDER << ", " << OUR_ENV_ACCESS_KEY << ", "
                   << OUR_ENV_SECRET_KEY << ", " << ALIYUN_ARN_ENV_ADDRESS << ", " << ALIYUN_ARN_ENV_REGION << ", "
                   << ALIYUN_ARN_ENV_BUCKET << ", " << ALIYUN_ARN_ENV_ACCESS_KEY << ", " << ALIYUN_ARN_ENV_SECRET_KEY
                   << ", " << ALIYUN_ARN_ENV_ROLE_ARN;
    }

    // Two machine-identity modes are supported; which one we need depends on
    // ALIYUN_ROLE_ARN_AUTH_MODE (kept in lockstep with the
    // dispatch in s3_filesystem_producer.cpp):
    //   - "ram":  ECS IMDS → sts:AssumeRole. No env vars required; the
    //             metadata service supplies the caller identity. Only runs
    //             on an ECS with a RAM role attached.
    //   - default / "oidc": legacy AssumeRoleWithOIDC. Requires
    //             ALIBABA_CLOUD_OIDC_TOKEN_FILE + _PROVIDER_ARN in process
    //             env; without them the Rust Lance path would surface a
    //             generic OSS 401 instead of a clear misconfig.
    const char* auth_mode_env = std::getenv("ALIYUN_ROLE_ARN_AUTH_MODE");
    const bool ram_mode = auth_mode_env != nullptr && std::string(auth_mode_env) == "ram";
    if (!ram_mode) {
      if (std::getenv("ALIBABA_CLOUD_OIDC_TOKEN_FILE") == nullptr ||
          std::getenv("ALIBABA_CLOUD_OIDC_PROVIDER_ARN") == nullptr) {
        GTEST_SKIP() << "Aliyun ARN test requires ALIBABA_CLOUD_OIDC_TOKEN_FILE and "
                        "ALIBABA_CLOUD_OIDC_PROVIDER_ARN in process env (pod-level machine identity), "
                        "or set ALIYUN_ROLE_ARN_AUTH_MODE=ram to use the ECS IMDS path";
      }
    }

    // Write properties: AKSK to the customer bucket. opendal's Oss service
    // accepts static AK/SK directly via reqsign's load_via_static — no
    // indirection needed.
    api::SetValue(write_props_, PROPERTY_FS_STORAGE_TYPE, "remote");
    api::SetValue(write_props_, PROPERTY_FS_CLOUD_PROVIDER, kCloudProviderAliyun);
    api::SetValue(write_props_, PROPERTY_FS_ADDRESS, address_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_BUCKET_NAME, arn_bucket_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_REGION, region_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_ACCESS_KEY_ID, arn_ak_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_ACCESS_KEY_VALUE, arn_sk_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_USE_SSL, "true");

    // Read properties: role_arn to trigger AssumeRoleWithOIDC. No AKSK on this
    // branch — see lance_common.cpp's role_arn emission: AK/SK alongside
    // role_arn would make reqsign's load_via_static win over
    // load_via_assume_role_with_oidc, silently bypassing the OIDC flow.
    api::SetValue(read_props_, "extfs.arn.storage_type", "remote");
    api::SetValue(read_props_, "extfs.arn.cloud_provider", kCloudProviderAliyun);
    api::SetValue(read_props_, "extfs.arn.address", address_.c_str());
    api::SetValue(read_props_, "extfs.arn.bucket_name", arn_bucket_.c_str());
    api::SetValue(read_props_, "extfs.arn.region", region_.c_str());
    api::SetValue(read_props_, "extfs.arn.use_ssl", "true");
    api::SetValue(read_props_, "extfs.arn.role_arn", role_arn_.c_str());

    // Create write filesystem for cleanup.
    ASSERT_AND_ASSIGN(write_fs_, GetFileSystem(write_props_));

    FilesystemCache::getInstance().clean();

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "zc/aliyun-arn-test-" + std::to_string(ts);
  }

  void TearDown() override {
    if (write_fs_) {
      (void)DeleteTestDir(write_fs_, test_base_);
    }
    FilesystemCache::getInstance().clean();
  }

  arrow::Result<ArnWriteResult> CreateLanceTable(uint64_t num_rows) {
    ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
    ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, 0, false, num_rows, 4, 50, {true, true, true, false}));
    auto path = test_base_ + "/lance";
    lance::LanceTableWriter writer(path, schema, write_props_);
    ARROW_RETURN_NOT_OK(writer.Write(batch));
    ARROW_ASSIGN_OR_RAISE(auto cgfile, writer.Close());
    std::cout << "[Aliyun ARN Test] Lance cgfile: " << cgfile.ToString() << std::endl;
    // explore_dir: oss://address/bucket/path (cloud provider URI scheme is oss)
    auto explore_dir = "oss://" + address_ + "/" + arn_bucket_ + "/" + path;
    return ArnWriteResult{std::move(cgfile), schema, num_rows, explore_dir, 0};
  }

  // Writes an Iceberg v1 table to the customer bucket using the AK/SK-backed
  // write_props_ (so the write goes through reqsign's static-creds path, not
  // the AssumeRole path). Returns the metadata.json URI in Milvus-URI shape
  // for loon_exttable_explore, plus the snapshot id needed as a prop.
  //
  // Shape-matches the AWS ExternalTableArnTest iceberg helper, minus the
  // cloud-provider hard-code — CreateTestTable takes whatever storage options
  // we hand it, and the Rust bridge picks the factory off the `oss://` scheme
  // baked into the metadata_location.
  arrow::Result<ArnWriteResult> CreateIcebergTable(uint64_t num_rows) {
    auto path = test_base_ + "/iceberg";
    auto table_uri = "oss://" + arn_bucket_ + "/" + path;

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
    std::cout << "[Aliyun ARN Test] Iceberg cgfile: " << cg_file.ToString() << std::endl;
    auto explore_dir = iceberg::ToMilvusUri(table_info.metadata_location, address_);
    return ArnWriteResult{std::move(cg_file), nullptr, num_rows, explore_dir, table_info.snapshot_id};
  }

  // Writes `num_files` parquet files of `rows_per_file` rows each into a fresh
  // subdirectory under test_base_, using the AK/SK-backed write_fs_ (so the
  // write itself never touches the role_arn path; that's exercised by the
  // explore/read steps below). Returns the (full URI) explore directory the
  // caller passes to loon_exttable_explore, plus the schema for FormatReader.
  struct ArnParquetWriteResult {
    std::shared_ptr<arrow::Schema> schema;
    uint64_t num_files;
    uint64_t rows_per_file;
    std::string explore_dir;  // oss://address/bucket/<dir>/   (trailing slash)
  };

  arrow::Result<ArnParquetWriteResult> CreateParquetFiles(uint64_t num_files, uint64_t rows_per_file) {
    ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
    auto dir = test_base_ + "/parquet";

    auto writer_props = ::parquet::WriterProperties::Builder().compression(::parquet::Compression::ZSTD)->build();

    for (uint64_t i = 0; i < num_files; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, /*start_offset=*/static_cast<int64_t>(i * rows_per_file),
                                                       /*randdata=*/false, rows_per_file, /*vector_dim=*/4,
                                                       /*str_length=*/50, {true, true, true, false}));
      auto file_path = dir + "/file_" + std::to_string(i) + ".parquet";
      ARROW_ASSIGN_OR_RAISE(auto sink, write_fs_->OpenOutputStream(file_path));
      ARROW_ASSIGN_OR_RAISE(auto pq_writer, ::parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(),
                                                                               sink, writer_props));
      ARROW_RETURN_NOT_OK(pq_writer->NewBufferedRowGroup());
      ARROW_RETURN_NOT_OK(pq_writer->WriteRecordBatch(*batch));
      ARROW_RETURN_NOT_OK(pq_writer->Close());
      ARROW_RETURN_NOT_OK(sink->Close());
      std::cout << "[Aliyun ARN Test] Parquet file written: " << file_path << std::endl;
    }

    // Trailing slash matches what the existing FFI tests pass to explore for
    // a "directory of parquet" — explore lists the prefix and treats every
    // matching object as a parquet file in a single column group.
    auto explore_dir = "oss://" + address_ + "/" + arn_bucket_ + "/" + dir + "/";
    return ArnParquetWriteResult{schema, num_files, rows_per_file, explore_dir};
  }

  // Our-side
  std::string our_address_;
  std::string our_bucket_;
  std::string our_region_;
  std::string our_cloud_provider_;
  std::string our_ak_;
  std::string our_sk_;
  // Customer-side
  std::string address_;
  std::string region_;
  std::string arn_bucket_;
  std::string arn_ak_;
  std::string arn_sk_;
  std::string role_arn_;

  api::Properties write_props_;
  api::Properties read_props_;
  ArrowFileSystemPtr write_fs_;
  std::string test_base_;
};

TEST_F(ExternalTableAliyunArnTest, ReadLanceWithArnRole) {
  const uint64_t num_rows = 100;

  // Step 1: Write test data using AKSK credentials (native OSS write via
  // opendal's Oss service; reqsign::load_via_static picks up AK/SK).
  ASSERT_AND_ASSIGN(auto result, CreateLanceTable(num_rows));

  std::cout << "[Aliyun ARN Test] Written to: " << result.cgfile.path << std::endl;
  std::cout << "[Aliyun ARN Test] Explore dir: " << result.explore_dir << std::endl;
  std::cout << "[Aliyun ARN Test] Role ARN: " << role_arn_ << std::endl;

  // Step 2: Build properties for loon_exttable_explore.
  //   - fs.*: our-side Aliyun OSS bucket with static AK/SK for manifest
  //     storage (see fixture-level comment: ECS RAM-role is not supported).
  //   - extfs.arn.*: Aliyun OSS + role_arn for reading the customer bucket.
  auto manifest_base = test_base_ + "/manifest";

  std::vector<std::pair<std::string, std::string>> props = {
      {PROPERTY_FS_STORAGE_TYPE, "remote"},    {PROPERTY_FS_CLOUD_PROVIDER, our_cloud_provider_},
      {PROPERTY_FS_ADDRESS, our_address_},     {PROPERTY_FS_BUCKET_NAME, our_bucket_},
      {PROPERTY_FS_REGION, our_region_},       {PROPERTY_FS_ACCESS_KEY_ID, our_ak_},
      {PROPERTY_FS_ACCESS_KEY_VALUE, our_sk_}, {PROPERTY_FS_USE_SSL, "true"},
      {"extfs.arn.storage_type", "remote"},    {"extfs.arn.cloud_provider", kCloudProviderAliyun},
      {"extfs.arn.address", address_},         {"extfs.arn.bucket_name", arn_bucket_},
      {"extfs.arn.region", region_},           {"extfs.arn.use_ssl", "true"},
      {"extfs.arn.role_arn", role_arn_},
  };

  std::vector<const char*> c_keys, c_values;
  c_keys.reserve(props.size());
  c_values.reserve(props.size());
  for (const auto& [k, v] : props) {
    c_keys.push_back(k.c_str());
    c_values.push_back(v.c_str());
  }

  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  // Step 3: Discover files via role_arn, write manifest to our-side bucket.
  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;

  rc = loon_exttable_explore(columns_arr, 3, LOON_FORMAT_LANCE_TABLE, manifest_base.c_str(), result.explore_dir.c_str(),
                             &loon_props, &out_num_files, &out_manifest_path);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_GT(out_num_files, 0u);
  ASSERT_NE(out_manifest_path, nullptr);

  std::cout << "[Aliyun ARN Test] loon_exttable_explore: found " << out_num_files
            << " files, manifest=" << out_manifest_path << std::endl;

  // Step 4: Read manifest.
  LoonManifest* out_manifest = nullptr;
  rc = loon_exttable_read_manifest(out_manifest_path, &loon_props, &out_manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(out_manifest, nullptr);
  ASSERT_EQ(out_manifest->column_groups.num_of_column_groups, 1u);

  auto* cg = &out_manifest->column_groups.column_group_array[0];
  ASSERT_EQ(cg->num_of_files, out_num_files);

  // Step 5: Read data using FormatReader with role_arn (AliyunOssStoreProvider
  // + reqsign AssumeRoleWithOIDC).
  std::vector<std::string> columns = {"id", "name", "value"};
  int64_t total_rows = 0;
  for (uint64_t f = 0; f < cg->num_of_files; ++f) {
    auto& loon_file = cg->files[f];
    api::ColumnGroupFile cgfile;
    cgfile.path = loon_file.path;
    cgfile.start_index = loon_file.start_index;
    cgfile.end_index = loon_file.end_index;
    if (loon_file.property_keys != nullptr) {
      for (uint32_t p = 0; p < loon_file.num_properties; ++p) {
        cgfile.properties[loon_file.property_keys[p]] = loon_file.property_values[p];
      }
    }

    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(result.schema, LOON_FORMAT_LANCE_TABLE, cgfile, read_props_,
                                                        columns, nullptr));
    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total_rows += batch->num_rows();
    }
  }
  ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));
  std::cout << "[Aliyun ARN Test] FormatReader read " << total_rows << " rows via Aliyun ARN OK" << std::endl;

  loon_manifest_destroy(out_manifest);
  free(out_manifest_path);
  loon_properties_free(&loon_props);
}

// Iceberg variant: writes an iceberg v1 table with AK/SK to the customer
// bucket, then exercises loon_exttable_explore + FormatReader via role_arn.
// End-to-end path:
//   write  — reqsign load_via_static through `AliyunOssStorage::create_operator`
//   plan   — iceberg::PlanFiles with role_arn (pre-write verification that
//            AssumeRoleWithOIDC resolves against the table's metadata.json)
//   read   — FormatReader drives `AliyunOssStorage` for each data file.
//
// Parallels ReadLanceWithArnRole step-for-step; the only format-specific bit
// is passing PROPERTY_ICEBERG_SNAPSHOT_ID so loon_exttable_explore pins the
// scan to the snapshot we just wrote.
TEST_F(ExternalTableAliyunArnTest, ReadIcebergWithArnRole) {
  const uint64_t num_rows = 100;

  // Step 1: Write iceberg table using AKSK credentials.
  ASSERT_AND_ASSIGN(auto result, CreateIcebergTable(num_rows));

  std::cout << "[Aliyun ARN Test] Iceberg metadata: " << result.explore_dir << std::endl;
  std::cout << "[Aliyun ARN Test] Role ARN: " << role_arn_ << std::endl;

  // Step 2: Build properties — our-side AKSK for manifest storage, customer-side
  // role_arn for reading. Iceberg also needs PROPERTY_ICEBERG_SNAPSHOT_ID so the
  // Rust bridge pins the scan to our freshly-written snapshot.
  auto manifest_base = test_base_ + "/manifest";

  std::vector<std::pair<std::string, std::string>> props = {
      {PROPERTY_FS_STORAGE_TYPE, "remote"},
      {PROPERTY_FS_CLOUD_PROVIDER, our_cloud_provider_},
      {PROPERTY_FS_ADDRESS, our_address_},
      {PROPERTY_FS_BUCKET_NAME, our_bucket_},
      {PROPERTY_FS_REGION, our_region_},
      {PROPERTY_FS_ACCESS_KEY_ID, our_ak_},
      {PROPERTY_FS_ACCESS_KEY_VALUE, our_sk_},
      {PROPERTY_FS_USE_SSL, "true"},
      {"extfs.arn.storage_type", "remote"},
      {"extfs.arn.cloud_provider", kCloudProviderAliyun},
      {"extfs.arn.address", address_},
      {"extfs.arn.bucket_name", arn_bucket_},
      {"extfs.arn.region", region_},
      {"extfs.arn.use_ssl", "true"},
      {"extfs.arn.role_arn", role_arn_},
      {PROPERTY_ICEBERG_SNAPSHOT_ID, std::to_string(result.iceberg_snapshot_id)},
  };

  std::vector<const char*> c_keys, c_values;
  c_keys.reserve(props.size());
  c_values.reserve(props.size());
  for (const auto& [k, v] : props) {
    c_keys.push_back(k.c_str());
    c_values.push_back(v.c_str());
  }

  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  // Step 3: explore (iceberg metadata read) via role_arn.
  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;

  rc = loon_exttable_explore(columns_arr, 3, LOON_FORMAT_ICEBERG_TABLE, manifest_base.c_str(),
                             result.explore_dir.c_str(), &loon_props, &out_num_files, &out_manifest_path);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_GT(out_num_files, 0u);
  ASSERT_NE(out_manifest_path, nullptr);

  std::cout << "[Aliyun ARN Test] loon_exttable_explore(iceberg): found " << out_num_files
            << " files, manifest=" << out_manifest_path << std::endl;

  // Step 4: Read manifest.
  LoonManifest* out_manifest = nullptr;
  rc = loon_exttable_read_manifest(out_manifest_path, &loon_props, &out_manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(out_manifest, nullptr);
  ASSERT_EQ(out_manifest->column_groups.num_of_column_groups, 1u);

  auto* cg = &out_manifest->column_groups.column_group_array[0];
  ASSERT_EQ(cg->num_of_files, out_num_files);

  // Step 5: Read data via FormatReader (each data file re-enters
  // AliyunOssStorage::read through the role_arn path). iceberg_test_table
  // writes a schema the read_props_ have to pick up at FormatReader::create
  // time — we ask the bridge to provide it via loon_exttable_get_schema so
  // the test doesn't hard-code the schema shape.
  std::vector<std::string> columns = {"id", "name", "value"};
  int64_t total_rows = 0;
  for (uint64_t f = 0; f < cg->num_of_files; ++f) {
    auto& loon_file = cg->files[f];
    api::ColumnGroupFile cgfile;
    cgfile.path = loon_file.path;
    cgfile.start_index = loon_file.start_index;
    cgfile.end_index = loon_file.end_index;
    if (loon_file.property_keys != nullptr) {
      for (uint32_t p = 0; p < loon_file.num_properties; ++p) {
        cgfile.properties[loon_file.property_keys[p]] = loon_file.property_values[p];
      }
    }

    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(result.schema, LOON_FORMAT_ICEBERG_TABLE, cgfile, read_props_,
                                                        columns, nullptr));
    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total_rows += batch->num_rows();
    }
  }
  ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));
  std::cout << "[Aliyun ARN Test] FormatReader read " << total_rows << " rows from iceberg via Aliyun ARN OK"
            << std::endl;

  loon_manifest_destroy(out_manifest);
  free(out_manifest_path);
  loon_properties_free(&loon_props);
}

// Parquet variant: writes 2 parquet files with AK/SK to the customer bucket,
// then exercises loon_exttable_explore + loon_exttable_get_file_info via the
// role_arn. Verifies the per-file row count from get_file_info before falling
// through to the same manifest-read + FormatReader pipeline as the Lance test.
TEST_F(ExternalTableAliyunArnTest, ReadTwoParquetFilesWithArnRole) {
  const uint64_t num_files = 2;
  const uint64_t rows_per_file = 50;
  const uint64_t total_rows = num_files * rows_per_file;

  // Step 1: Write parquet files with AK/SK (no role_arn involved on this leg).
  ASSERT_AND_ASSIGN(auto result, CreateParquetFiles(num_files, rows_per_file));

  std::cout << "[Aliyun ARN Test] Explore dir: " << result.explore_dir << std::endl;
  std::cout << "[Aliyun ARN Test] Role ARN: " << role_arn_ << std::endl;

  // Step 2: Build properties for loon_exttable_explore — same structure as
  // the Lance variant so the role_arn dispatch is identical.
  auto manifest_base = test_base_ + "/manifest";

  std::vector<std::pair<std::string, std::string>> props = {
      {PROPERTY_FS_STORAGE_TYPE, "remote"},    {PROPERTY_FS_CLOUD_PROVIDER, our_cloud_provider_},
      {PROPERTY_FS_ADDRESS, our_address_},     {PROPERTY_FS_BUCKET_NAME, our_bucket_},
      {PROPERTY_FS_REGION, our_region_},       {PROPERTY_FS_ACCESS_KEY_ID, our_ak_},
      {PROPERTY_FS_ACCESS_KEY_VALUE, our_sk_}, {PROPERTY_FS_USE_SSL, "true"},
      {"extfs.arn.storage_type", "remote"},    {"extfs.arn.cloud_provider", kCloudProviderAliyun},
      {"extfs.arn.address", address_},         {"extfs.arn.bucket_name", arn_bucket_},
      {"extfs.arn.region", region_},           {"extfs.arn.use_ssl", "true"},
      {"extfs.arn.role_arn", role_arn_},
  };

  std::vector<const char*> c_keys, c_values;
  c_keys.reserve(props.size());
  c_values.reserve(props.size());
  for (const auto& [k, v] : props) {
    c_keys.push_back(k.c_str());
    c_values.push_back(v.c_str());
  }

  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  // Step 3: Discover the 2 parquet files via role_arn, write manifest to our-side bucket.
  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;

  rc = loon_exttable_explore(columns_arr, 3, LOON_FORMAT_PARQUET, manifest_base.c_str(), result.explore_dir.c_str(),
                             &loon_props, &out_num_files, &out_manifest_path);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_EQ(out_num_files, num_files);
  ASSERT_NE(out_manifest_path, nullptr);

  std::cout << "[Aliyun ARN Test] loon_exttable_explore: found " << out_num_files
            << " parquet files, manifest=" << out_manifest_path << std::endl;

  // Step 4: Read manifest.
  LoonManifest* out_manifest = nullptr;
  rc = loon_exttable_read_manifest(out_manifest_path, &loon_props, &out_manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(out_manifest, nullptr);
  ASSERT_EQ(out_manifest->column_groups.num_of_column_groups, 1u);

  auto* cg = &out_manifest->column_groups.column_group_array[0];
  ASSERT_EQ(cg->num_of_files, num_files);

  // Step 5: For each discovered file call loon_exttable_get_file_info via the
  // same loon_props (so the read goes through the role_arn path). Each file
  // must report exactly rows_per_file rows.
  for (uint64_t f = 0; f < cg->num_of_files; ++f) {
    auto& loon_file = cg->files[f];
    uint64_t per_file_rows = 0;
    rc = loon_exttable_get_file_info(LOON_FORMAT_PARQUET, loon_file.path, &loon_props, &per_file_rows);
    ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
    ASSERT_EQ(per_file_rows, rows_per_file) << "File " << f << " (" << loon_file.path << ") row count mismatch";
    std::cout << "[Aliyun ARN Test] get_file_info[" << f << "]: " << loon_file.path << " rows=" << per_file_rows
              << std::endl;
  }

  // Step 6: Read data with FormatReader (same as the Lance variant).
  std::vector<std::string> columns = {"id", "name", "value"};
  int64_t total_rows_read = 0;
  for (uint64_t f = 0; f < cg->num_of_files; ++f) {
    auto& loon_file = cg->files[f];
    api::ColumnGroupFile cgfile;
    cgfile.path = loon_file.path;
    cgfile.start_index = loon_file.start_index;
    cgfile.end_index = loon_file.end_index;
    if (loon_file.property_keys != nullptr) {
      for (uint32_t p = 0; p < loon_file.num_properties; ++p) {
        cgfile.properties[loon_file.property_keys[p]] = loon_file.property_values[p];
      }
    }

    ASSERT_AND_ASSIGN(auto reader,
                      FormatReader::create(result.schema, LOON_FORMAT_PARQUET, cgfile, read_props_, columns, nullptr));
    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total_rows_read += batch->num_rows();
    }
  }
  ASSERT_EQ(total_rows_read, static_cast<int64_t>(total_rows));
  std::cout << "[Aliyun ARN Test] FormatReader read " << total_rows_read << " rows from " << num_files
            << " parquet files via Aliyun ARN OK" << std::endl;

  loon_manifest_destroy(out_manifest);
  free(out_manifest_path);
  loon_properties_free(&loon_props);
}

// Vortex follows the same plain-format path as Parquet. This verifies that
// discovery, metadata loading, and data reads all use the customer role ARN.
TEST_F(ExternalTableAliyunArnTest, ReadVortexWithArnRole) {
  const uint64_t num_rows = 100;

  ASSERT_AND_ASSIGN(auto result, WritePlainFormatFile(LOON_FORMAT_VORTEX, num_rows, write_fs_, write_props_, test_base_,
                                                      "oss", address_, arn_bucket_));

  std::cout << "[Aliyun ARN Test] Vortex explore dir: " << result.explore_dir << std::endl;
  std::cout << "[Aliyun ARN Test] Role ARN: " << role_arn_ << std::endl;

  auto manifest_base = test_base_ + "/manifest";
  std::vector<std::pair<std::string, std::string>> props = {
      {PROPERTY_FS_STORAGE_TYPE, "remote"},    {PROPERTY_FS_CLOUD_PROVIDER, our_cloud_provider_},
      {PROPERTY_FS_ADDRESS, our_address_},     {PROPERTY_FS_BUCKET_NAME, our_bucket_},
      {PROPERTY_FS_REGION, our_region_},       {PROPERTY_FS_ACCESS_KEY_ID, our_ak_},
      {PROPERTY_FS_ACCESS_KEY_VALUE, our_sk_}, {PROPERTY_FS_USE_SSL, "true"},
      {"extfs.arn.storage_type", "remote"},    {"extfs.arn.cloud_provider", kCloudProviderAliyun},
      {"extfs.arn.address", address_},         {"extfs.arn.bucket_name", arn_bucket_},
      {"extfs.arn.region", region_},           {"extfs.arn.use_ssl", "true"},
      {"extfs.arn.role_arn", role_arn_},
  };

  std::vector<const char*> c_keys, c_values;
  c_keys.reserve(props.size());
  c_values.reserve(props.size());
  for (const auto& [k, v] : props) {
    c_keys.push_back(k.c_str());
    c_values.push_back(v.c_str());
  }

  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;
  rc = loon_exttable_explore(columns_arr, 3, LOON_FORMAT_VORTEX, manifest_base.c_str(), result.explore_dir.c_str(),
                             &loon_props, &out_num_files, &out_manifest_path);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_EQ(out_num_files, 1u);
  ASSERT_NE(out_manifest_path, nullptr);

  LoonManifest* out_manifest = nullptr;
  rc = loon_exttable_read_manifest(out_manifest_path, &loon_props, &out_manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(out_manifest, nullptr);
  ASSERT_EQ(out_manifest->column_groups.num_of_column_groups, 1u);

  auto* cg = &out_manifest->column_groups.column_group_array[0];
  ASSERT_EQ(cg->num_of_files, 1u);

  auto& loon_file = cg->files[0];
  uint64_t file_rows = 0;
  rc = loon_exttable_get_file_info(LOON_FORMAT_VORTEX, loon_file.path, &loon_props, &file_rows);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_EQ(file_rows, num_rows);

  api::ColumnGroupFile cgfile;
  cgfile.path = loon_file.path;
  cgfile.start_index = loon_file.start_index;
  cgfile.end_index = loon_file.end_index;
  if (loon_file.property_keys != nullptr) {
    for (uint32_t p = 0; p < loon_file.num_properties; ++p) {
      cgfile.properties[loon_file.property_keys[p]] = loon_file.property_values[p];
    }
  }

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, LOON_FORMAT_VORTEX, cgfile, read_props_, columns, nullptr));
  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  int64_t total_rows_read = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    total_rows_read += batch->num_rows();
  }
  ASSERT_EQ(total_rows_read, static_cast<int64_t>(num_rows));

  loon_manifest_destroy(out_manifest);
  free(out_manifest_path);
  loon_properties_free(&loon_props);
}

// ===========================================================================
// Aliyun OIDC chain integration test (`AliyunOIDCAssumeRoleChainProvider` on
// the C++ side, `apply_oidc_chain_if_requested` on the Rust side).
//
// RAM mode is intentionally NOT covered here; that path stays under
// `ExternalTableAliyunArnTest` above which exercises both modes via env.
//
// Required process env (the three RRSA-style ALIBABA_CLOUD_* triple, same
// shape K8s would inject inside an ACK pod):
//   ALIBABA_CLOUD_ROLE_ARN          machine identity role (zilliz-side)
//   ALIBABA_CLOUD_OIDC_PROVIDER_ARN machine identity OIDC provider
//   ALIBABA_CLOUD_OIDC_TOKEN_FILE   absolute path to a JWT pulled from a
//                                   live pod. Tokens expire ~1h after issue.
//
// `aliyun-config/oidc-config.env` in the repo root is a `source`-able
// template — fill in the token absolute path, then `source` it before
// running the test. The fixture itself doesn't read any files, just env.
//
// Optional `ALIYUN_ARN_TEST_ENV_EXTERNAL_ID` — set only if the customer's
// role trust policy carries a matching `sts:ExternalId` condition. Empty
// means no `ExternalId` parameter is sent on the step-2 AssumeRole.
// ===========================================================================
#define ALIYUN_ARN_ENV_EXTERNAL_ID "ALIYUN_ARN_TEST_ENV_EXTERNAL_ID"

class ExternalTableAliyunOIDCArnTest : public ::testing::Test {
  protected:
  void SetUp() override {
    // 1. Machine identity must be in process env. The dispatch in
    //    s3_filesystem_producer.cpp fails fast if missing, but skipping
    //    here gives a clearer diagnostic when the user just forgot to
    //    `source aliyun-config/oidc-config.env`.
    if (std::getenv("ALIBABA_CLOUD_ROLE_ARN") == nullptr || std::getenv("ALIBABA_CLOUD_OIDC_PROVIDER_ARN") == nullptr ||
        std::getenv("ALIBABA_CLOUD_OIDC_TOKEN_FILE") == nullptr) {
      GTEST_SKIP() << "Aliyun OIDC chain test requires ALIBABA_CLOUD_ROLE_ARN, "
                      "ALIBABA_CLOUD_OIDC_PROVIDER_ARN and ALIBABA_CLOUD_OIDC_TOKEN_FILE "
                      "in process env. Try `source aliyun-config/oidc-config.env`.";
    }

    // 2. Per-tenant test config — same env vars as ExternalTableAliyunArnTest
    //    so the same CI secrets can drive both fixtures.
    our_address_ = GetEnvVar(OUR_ENV_ADDRESS).ValueOr("");
    our_bucket_ = GetEnvVar(OUR_ENV_BUCKET).ValueOr("");
    our_region_ = GetEnvVar(OUR_ENV_REGION).ValueOr("");
    our_cloud_provider_ = GetEnvVar(OUR_ENV_CLOUD_PROVIDER).ValueOr("");
    our_ak_ = GetEnvVar(OUR_ENV_ACCESS_KEY).ValueOr("");
    our_sk_ = GetEnvVar(OUR_ENV_SECRET_KEY).ValueOr("");

    address_ = GetEnvVar(ALIYUN_ARN_ENV_ADDRESS).ValueOr("");
    region_ = GetEnvVar(ALIYUN_ARN_ENV_REGION).ValueOr("");
    arn_bucket_ = GetEnvVar(ALIYUN_ARN_ENV_BUCKET).ValueOr("");
    arn_ak_ = GetEnvVar(ALIYUN_ARN_ENV_ACCESS_KEY).ValueOr("");
    arn_sk_ = GetEnvVar(ALIYUN_ARN_ENV_SECRET_KEY).ValueOr("");
    role_arn_ = GetEnvVar(ALIYUN_ARN_ENV_ROLE_ARN).ValueOr("");
    // ExternalId is optional. Empty == not sent (the chain provider's
    // step-2 body omits the parameter entirely; see
    // TestAliyunOIDCChainProviderForwardsExternalIdToStep2Only).
    external_id_ = GetEnvVar(ALIYUN_ARN_ENV_EXTERNAL_ID).ValueOr("");

    if (our_address_.empty() || our_bucket_.empty() || our_cloud_provider_.empty() || our_ak_.empty() ||
        our_sk_.empty() || address_.empty() || region_.empty() || arn_bucket_.empty() || arn_ak_.empty() ||
        arn_sk_.empty() || role_arn_.empty()) {
      GTEST_SKIP() << "Aliyun OIDC chain test requires env vars: " << OUR_ENV_ADDRESS << ", " << OUR_ENV_BUCKET << ", "
                   << OUR_ENV_REGION << ", " << OUR_ENV_CLOUD_PROVIDER << ", " << OUR_ENV_ACCESS_KEY << ", "
                   << OUR_ENV_SECRET_KEY << ", " << ALIYUN_ARN_ENV_ADDRESS << ", " << ALIYUN_ARN_ENV_REGION << ", "
                   << ALIYUN_ARN_ENV_BUCKET << ", " << ALIYUN_ARN_ENV_ACCESS_KEY << ", " << ALIYUN_ARN_ENV_SECRET_KEY
                   << ", " << ALIYUN_ARN_ENV_ROLE_ARN << " (plus optional " << ALIYUN_ARN_ENV_EXTERNAL_ID << ")";
    }

    // 4. write_props_ — AKSK against the customer bucket. Same as
    //    ExternalTableAliyunArnTest: the role_arn flow is exercised on the
    //    read leg only.
    api::SetValue(write_props_, PROPERTY_FS_STORAGE_TYPE, "remote");
    api::SetValue(write_props_, PROPERTY_FS_CLOUD_PROVIDER, kCloudProviderAliyun);
    api::SetValue(write_props_, PROPERTY_FS_ADDRESS, address_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_BUCKET_NAME, arn_bucket_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_REGION, region_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_ACCESS_KEY_ID, arn_ak_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_ACCESS_KEY_VALUE, arn_sk_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_USE_SSL, "true");

    // 5. read_props_ — extfs.arn.* with role_arn (and optional external_id)
    //    routed through the OIDC chain provider.
    api::SetValue(read_props_, "extfs.arn.storage_type", "remote");
    api::SetValue(read_props_, "extfs.arn.cloud_provider", kCloudProviderAliyun);
    api::SetValue(read_props_, "extfs.arn.address", address_.c_str());
    api::SetValue(read_props_, "extfs.arn.bucket_name", arn_bucket_.c_str());
    api::SetValue(read_props_, "extfs.arn.region", region_.c_str());
    api::SetValue(read_props_, "extfs.arn.use_ssl", "true");
    api::SetValue(read_props_, "extfs.arn.role_arn", role_arn_.c_str());
    if (!external_id_.empty()) {
      api::SetValue(read_props_, "extfs.arn.external_id", external_id_.c_str());
    }

    ASSERT_AND_ASSIGN(write_fs_, GetFileSystem(write_props_));
    FilesystemCache::getInstance().clean();

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "zc/aliyun-oidc-test-" + std::to_string(ts);
  }

  void TearDown() override {
    if (write_fs_) {
      (void)DeleteTestDir(write_fs_, test_base_);
    }
    FilesystemCache::getInstance().clean();
  }

  // FFI properties payload shared by all four formats. Iceberg additionally
  // needs PROPERTY_ICEBERG_SNAPSHOT_ID; callers append.
  std::vector<std::pair<std::string, std::string>> BaseProps() const {
    std::vector<std::pair<std::string, std::string>> props = {
        {PROPERTY_FS_STORAGE_TYPE, "remote"},    {PROPERTY_FS_CLOUD_PROVIDER, our_cloud_provider_},
        {PROPERTY_FS_ADDRESS, our_address_},     {PROPERTY_FS_BUCKET_NAME, our_bucket_},
        {PROPERTY_FS_REGION, our_region_},       {PROPERTY_FS_ACCESS_KEY_ID, our_ak_},
        {PROPERTY_FS_ACCESS_KEY_VALUE, our_sk_}, {PROPERTY_FS_USE_SSL, "true"},
        {"extfs.arn.storage_type", "remote"},    {"extfs.arn.cloud_provider", kCloudProviderAliyun},
        {"extfs.arn.address", address_},         {"extfs.arn.bucket_name", arn_bucket_},
        {"extfs.arn.region", region_},           {"extfs.arn.use_ssl", "true"},
        {"extfs.arn.role_arn", role_arn_},
    };
    if (!external_id_.empty()) {
      props.emplace_back("extfs.arn.external_id", external_id_);
    }
    return props;
  }

  arrow::Result<ArnWriteResult> CreateLanceTable(uint64_t num_rows) {
    ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
    ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, 0, false, num_rows, 4, 50, {true, true, true, false}));
    auto path = test_base_ + "/lance";
    lance::LanceTableWriter writer(path, schema, write_props_);
    ARROW_RETURN_NOT_OK(writer.Write(batch));
    ARROW_ASSIGN_OR_RAISE(auto cgfile, writer.Close());
    auto explore_dir = "oss://" + address_ + "/" + arn_bucket_ + "/" + path;
    return ArnWriteResult{std::move(cgfile), schema, num_rows, explore_dir, 0};
  }

  arrow::Result<ArnWriteResult> CreateIcebergTable(uint64_t num_rows) {
    auto path = test_base_ + "/iceberg";
    auto table_uri = "oss://" + arn_bucket_ + "/" + path;
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
    auto explore_dir = iceberg::ToMilvusUri(table_info.metadata_location, address_);
    return ArnWriteResult{std::move(cg_file), nullptr, num_rows, explore_dir, table_info.snapshot_id};
  }

  struct ParquetWriteResult {
    std::shared_ptr<arrow::Schema> schema;
    uint64_t num_files;
    uint64_t rows_per_file;
    std::string explore_dir;
  };

  arrow::Result<ParquetWriteResult> CreateParquetFiles(uint64_t num_files, uint64_t rows_per_file) {
    ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
    auto dir = test_base_ + "/parquet";
    auto writer_props = ::parquet::WriterProperties::Builder().compression(::parquet::Compression::ZSTD)->build();

    for (uint64_t i = 0; i < num_files; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, /*start_offset=*/static_cast<int64_t>(i * rows_per_file),
                                                       /*randdata=*/false, rows_per_file, /*vector_dim=*/4,
                                                       /*str_length=*/50, {true, true, true, false}));
      auto file_path = dir + "/file_" + std::to_string(i) + ".parquet";
      ARROW_ASSIGN_OR_RAISE(auto sink, write_fs_->OpenOutputStream(file_path));
      ARROW_ASSIGN_OR_RAISE(auto pq_writer, ::parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(),
                                                                               sink, writer_props));
      ARROW_RETURN_NOT_OK(pq_writer->NewBufferedRowGroup());
      ARROW_RETURN_NOT_OK(pq_writer->WriteRecordBatch(*batch));
      ARROW_RETURN_NOT_OK(pq_writer->Close());
      ARROW_RETURN_NOT_OK(sink->Close());
    }
    auto explore_dir = "oss://" + address_ + "/" + arn_bucket_ + "/" + dir + "/";
    return ParquetWriteResult{schema, num_files, rows_per_file, explore_dir};
  }

  // Our-side
  std::string our_address_, our_bucket_, our_region_, our_cloud_provider_, our_ak_, our_sk_;
  // Customer-side
  std::string address_, region_, arn_bucket_, arn_ak_, arn_sk_, role_arn_, external_id_;

  api::Properties write_props_;
  api::Properties read_props_;
  ArrowFileSystemPtr write_fs_;
  std::string test_base_;
};

// Lance + OIDC chain end-to-end. Writes with AKSK, reads back through
// loon_exttable_explore (Rust AliyunOssStoreProvider on step 2 of the
// chain) + FormatReader.
TEST_F(ExternalTableAliyunOIDCArnTest, ReadLanceWithOIDCChain) {
  const uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateLanceTable(num_rows));

  std::cout << "[Aliyun OIDC Test] Lance written to: " << result.cgfile.path << std::endl;
  std::cout << "[Aliyun OIDC Test] Role ARN: " << role_arn_ << (external_id_.empty() ? "" : " (with external_id)")
            << std::endl;

  auto manifest_base = test_base_ + "/manifest";
  auto props = BaseProps();

  std::vector<const char*> c_keys, c_values;
  c_keys.reserve(props.size());
  c_values.reserve(props.size());
  for (const auto& [k, v] : props) {
    c_keys.push_back(k.c_str());
    c_values.push_back(v.c_str());
  }
  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;
  rc = loon_exttable_explore(columns_arr, 3, LOON_FORMAT_LANCE_TABLE, manifest_base.c_str(), result.explore_dir.c_str(),
                             &loon_props, &out_num_files, &out_manifest_path);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_GT(out_num_files, 0u);

  LoonManifest* out_manifest = nullptr;
  rc = loon_exttable_read_manifest(out_manifest_path, &loon_props, &out_manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(out_manifest, nullptr);
  ASSERT_EQ(out_manifest->column_groups.num_of_column_groups, 1u);
  auto* cg = &out_manifest->column_groups.column_group_array[0];
  ASSERT_EQ(cg->num_of_files, out_num_files);

  std::vector<std::string> columns = {"id", "name", "value"};
  int64_t total_rows = 0;
  for (uint64_t f = 0; f < cg->num_of_files; ++f) {
    auto& loon_file = cg->files[f];
    api::ColumnGroupFile cgfile;
    cgfile.path = loon_file.path;
    cgfile.start_index = loon_file.start_index;
    cgfile.end_index = loon_file.end_index;
    if (loon_file.property_keys != nullptr) {
      for (uint32_t p = 0; p < loon_file.num_properties; ++p) {
        cgfile.properties[loon_file.property_keys[p]] = loon_file.property_values[p];
      }
    }
    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(result.schema, LOON_FORMAT_LANCE_TABLE, cgfile, read_props_,
                                                        columns, nullptr));
    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total_rows += batch->num_rows();
    }
  }
  ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));

  loon_manifest_destroy(out_manifest);
  free(out_manifest_path);
  loon_properties_free(&loon_props);
}

// Iceberg variant for the same OIDC chain. This reuses BaseProps() for the
// our-side manifest bucket plus extfs.arn.* read credentials, and appends the
// snapshot id required by the Iceberg bridge.
TEST_F(ExternalTableAliyunOIDCArnTest, ReadIcebergWithOIDCChain) {
  const uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateIcebergTable(num_rows));

  std::cout << "[Aliyun OIDC Test] Iceberg metadata: " << result.explore_dir << std::endl;
  std::cout << "[Aliyun OIDC Test] Role ARN: " << role_arn_ << (external_id_.empty() ? "" : " (with external_id)")
            << std::endl;

  auto manifest_base = test_base_ + "/manifest";
  auto props = BaseProps();
  props.emplace_back(PROPERTY_ICEBERG_SNAPSHOT_ID, std::to_string(result.iceberg_snapshot_id));

  std::vector<const char*> c_keys, c_values;
  c_keys.reserve(props.size());
  c_values.reserve(props.size());
  for (const auto& [k, v] : props) {
    c_keys.push_back(k.c_str());
    c_values.push_back(v.c_str());
  }
  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;
  rc = loon_exttable_explore(columns_arr, 3, LOON_FORMAT_ICEBERG_TABLE, manifest_base.c_str(),
                             result.explore_dir.c_str(), &loon_props, &out_num_files, &out_manifest_path);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_GT(out_num_files, 0u);
  ASSERT_NE(out_manifest_path, nullptr);

  LoonManifest* out_manifest = nullptr;
  rc = loon_exttable_read_manifest(out_manifest_path, &loon_props, &out_manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(out_manifest, nullptr);
  ASSERT_EQ(out_manifest->column_groups.num_of_column_groups, 1u);
  auto* cg = &out_manifest->column_groups.column_group_array[0];
  ASSERT_EQ(cg->num_of_files, out_num_files);

  std::vector<std::string> columns = {"id", "name", "value"};
  int64_t total_rows = 0;
  for (uint64_t f = 0; f < cg->num_of_files; ++f) {
    auto& loon_file = cg->files[f];
    api::ColumnGroupFile cgfile;
    cgfile.path = loon_file.path;
    cgfile.start_index = loon_file.start_index;
    cgfile.end_index = loon_file.end_index;
    if (loon_file.property_keys != nullptr) {
      for (uint32_t p = 0; p < loon_file.num_properties; ++p) {
        cgfile.properties[loon_file.property_keys[p]] = loon_file.property_values[p];
      }
    }
    ASSERT_AND_ASSIGN(auto reader, FormatReader::create(result.schema, LOON_FORMAT_ICEBERG_TABLE, cgfile, read_props_,
                                                        columns, nullptr));
    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total_rows += batch->num_rows();
    }
  }
  ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));

  loon_manifest_destroy(out_manifest);
  free(out_manifest_path);
  loon_properties_free(&loon_props);
}

// Parquet variant — the format that surfaced the original prod bug
// (PlainFormat::explore goes through the C++ aws-sdk-cpp filesystem, i.e.
// `AliyunOIDCAssumeRoleChainProvider`, NOT the Rust opendal path that
// Lance/Iceberg take). A dedicated parquet test makes it impossible for a
// future refactor to "fix Lance/Iceberg" while leaving the parquet path
// regressed.
TEST_F(ExternalTableAliyunOIDCArnTest, ReadTwoParquetFilesWithOIDCChain) {
  const uint64_t num_files = 2;
  const uint64_t rows_per_file = 50;
  const uint64_t total_rows = num_files * rows_per_file;

  ASSERT_AND_ASSIGN(auto result, CreateParquetFiles(num_files, rows_per_file));

  std::cout << "[Aliyun OIDC Test] Parquet explore dir: " << result.explore_dir << std::endl;

  auto manifest_base = test_base_ + "/manifest";
  auto props = BaseProps();

  std::vector<const char*> c_keys, c_values;
  c_keys.reserve(props.size());
  c_values.reserve(props.size());
  for (const auto& [k, v] : props) {
    c_keys.push_back(k.c_str());
    c_values.push_back(v.c_str());
  }
  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;
  rc = loon_exttable_explore(columns_arr, 3, LOON_FORMAT_PARQUET, manifest_base.c_str(), result.explore_dir.c_str(),
                             &loon_props, &out_num_files, &out_manifest_path);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_EQ(out_num_files, num_files);

  LoonManifest* out_manifest = nullptr;
  rc = loon_exttable_read_manifest(out_manifest_path, &loon_props, &out_manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  auto* cg = &out_manifest->column_groups.column_group_array[0];
  ASSERT_EQ(cg->num_of_files, num_files);

  // get_file_info verifies the per-file row count via the same role_arn flow.
  for (uint64_t f = 0; f < cg->num_of_files; ++f) {
    auto& loon_file = cg->files[f];
    uint64_t per_file_rows = 0;
    rc = loon_exttable_get_file_info(LOON_FORMAT_PARQUET, loon_file.path, &loon_props, &per_file_rows);
    ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
    ASSERT_EQ(per_file_rows, rows_per_file);
  }

  std::vector<std::string> columns = {"id", "name", "value"};
  int64_t total_rows_read = 0;
  for (uint64_t f = 0; f < cg->num_of_files; ++f) {
    auto& loon_file = cg->files[f];
    api::ColumnGroupFile cgfile;
    cgfile.path = loon_file.path;
    cgfile.start_index = loon_file.start_index;
    cgfile.end_index = loon_file.end_index;
    if (loon_file.property_keys != nullptr) {
      for (uint32_t p = 0; p < loon_file.num_properties; ++p) {
        cgfile.properties[loon_file.property_keys[p]] = loon_file.property_values[p];
      }
    }
    ASSERT_AND_ASSIGN(auto reader,
                      FormatReader::create(result.schema, LOON_FORMAT_PARQUET, cgfile, read_props_, columns, nullptr));
    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total_rows_read += batch->num_rows();
    }
  }
  ASSERT_EQ(total_rows_read, static_cast<int64_t>(total_rows));

  loon_manifest_destroy(out_manifest);
  free(out_manifest_path);
  loon_properties_free(&loon_props);
}

// Vortex also uses the native C++ filesystem, so it must exercise the OIDC
// chain independently of the Rust-backed Lance and Iceberg paths.
TEST_F(ExternalTableAliyunOIDCArnTest, ReadVortexWithOIDCChain) {
  const uint64_t num_rows = 100;

  ASSERT_AND_ASSIGN(auto result, WritePlainFormatFile(LOON_FORMAT_VORTEX, num_rows, write_fs_, write_props_, test_base_,
                                                      "oss", address_, arn_bucket_));

  std::cout << "[Aliyun OIDC Test] Vortex explore dir: " << result.explore_dir << std::endl;

  auto manifest_base = test_base_ + "/manifest";
  auto props = BaseProps();

  std::vector<const char*> c_keys, c_values;
  c_keys.reserve(props.size());
  c_values.reserve(props.size());
  for (const auto& [k, v] : props) {
    c_keys.push_back(k.c_str());
    c_values.push_back(v.c_str());
  }

  LoonProperties loon_props = {};
  auto rc = loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), &loon_props);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  const char* columns_arr[] = {"id", "name", "value"};
  uint64_t out_num_files = 0;
  char* out_manifest_path = nullptr;
  rc = loon_exttable_explore(columns_arr, 3, LOON_FORMAT_VORTEX, manifest_base.c_str(), result.explore_dir.c_str(),
                             &loon_props, &out_num_files, &out_manifest_path);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_EQ(out_num_files, 1u);
  ASSERT_NE(out_manifest_path, nullptr);

  LoonManifest* out_manifest = nullptr;
  rc = loon_exttable_read_manifest(out_manifest_path, &loon_props, &out_manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(out_manifest, nullptr);
  ASSERT_EQ(out_manifest->column_groups.num_of_column_groups, 1u);

  auto* cg = &out_manifest->column_groups.column_group_array[0];
  ASSERT_EQ(cg->num_of_files, 1u);

  auto& loon_file = cg->files[0];
  uint64_t file_rows = 0;
  rc = loon_exttable_get_file_info(LOON_FORMAT_VORTEX, loon_file.path, &loon_props, &file_rows);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_EQ(file_rows, num_rows);

  api::ColumnGroupFile cgfile;
  cgfile.path = loon_file.path;
  cgfile.start_index = loon_file.start_index;
  cgfile.end_index = loon_file.end_index;
  if (loon_file.property_keys != nullptr) {
    for (uint32_t p = 0; p < loon_file.num_properties; ++p) {
      cgfile.properties[loon_file.property_keys[p]] = loon_file.property_values[p];
    }
  }

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, LOON_FORMAT_VORTEX, cgfile, read_props_, columns, nullptr));
  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  int64_t total_rows_read = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    total_rows_read += batch->num_rows();
  }
  ASSERT_EQ(total_rows_read, static_cast<int64_t>(num_rows));

  loon_manifest_destroy(out_manifest);
  free(out_manifest_path);
  loon_properties_free(&loon_props);
}

}  // namespace milvus_storage
