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

// External-table integration tests with independent authentication fixtures.
// Shared data builders and read scenarios below exercise the same formats
// through AWS roles, GCP impersonation, Azure/S3 brokers, and OSS/COS roles.
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
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <arrow/api.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <curl/curl.h>
#include <folly/ScopeGuard.h>
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
#include "lance/lance_bridge.h"
#include "iceberg/iceberg_bridge.h"
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
                                                          const std::shared_ptr<const api::Properties>& write_props,
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

static arrow::Result<ArnWriteResult> WriteLanceTable(const uint64_t num_rows,
                                                     const std::shared_ptr<const api::Properties>& write_props,
                                                     const std::string& path,
                                                     const std::string& explore_dir) {
  ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
  ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, 0, false, num_rows, 4, 50, {true, true, true, false}));
  lance::LanceTableWriter writer(path, schema, write_props);
  ARROW_RETURN_NOT_OK(writer.Write(batch));
  ARROW_ASSIGN_OR_RAISE(auto cgfile, writer.Close());
  return ArnWriteResult{std::move(cgfile), std::move(schema), num_rows, explore_dir, 0};
}

static arrow::Result<ArnWriteResult> WriteIcebergTable(const uint64_t num_rows,
                                                       const std::shared_ptr<const api::Properties>& write_props,
                                                       const ArrowFileSystemPtr& write_fs,
                                                       const std::string& table_uri,
                                                       const std::string& address,
                                                       const std::string& record_scheme = "",
                                                       const std::string& explore_scheme = "") {
  ArrowFileSystemConfig write_config;
  ARROW_RETURN_NOT_OK(ArrowFileSystemConfig::create_file_system_config(write_props, write_config));
  ARROW_ASSIGN_OR_RAISE(auto storage_options, iceberg::ToWriterOptions(write_config));
  ARROW_ASSIGN_OR_RAISE(auto table_info,
                        iceberg::CreateTestTable(table_uri, num_rows, false, {}, storage_options, record_scheme));

  api::ColumnGroupFile cgfile;
  if (!record_scheme.empty()) {
    // GCP writes through S3 HMAC but records gs:// throughout the metadata.
    // Avoid constructing an AWS filesystem before the native GCP read path.
    cgfile = {iceberg::ToMilvusUri(table_info.data_file_uri, address), 0, static_cast<int64_t>(num_rows), {}};
  } else {
    ARROW_ASSIGN_OR_RAISE(auto files, iceberg::PlanFiles(table_info.metadata_location, table_info.snapshot_id, write_fs,
                                                         iceberg::ToReaderOptions(write_config)));
    if (files.empty()) {
      return arrow::Status::Invalid("PlanFiles returned no files");
    }
    cgfile = {
        iceberg::ToMilvusUri(files[0].data_file_path, address), 0, static_cast<int64_t>(files[0].record_count), {}};
  }

  auto explore_dir = iceberg::ToMilvusUri(table_info.metadata_location, address);
  if (!explore_scheme.empty()) {
    // Azure's external-source URI uses the azure:// alias for abfss:// metadata.
    ARROW_ASSIGN_OR_RAISE(auto uri, StorageUri::Parse(table_info.metadata_location, false));
    uri.scheme = explore_scheme;
    uri.address = address;
    ARROW_ASSIGN_OR_RAISE(explore_dir, StorageUri::Make(uri));
  }
  return ArnWriteResult{std::move(cgfile), nullptr, num_rows, std::move(explore_dir), table_info.snapshot_id};
}

static arrow::Result<ArnWriteResult> WriteParquetFiles(const uint64_t num_files,
                                                       const uint64_t rows_per_file,
                                                       const ArrowFileSystemPtr& write_fs,
                                                       const std::string& dir,
                                                       const std::string& explore_dir) {
  ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
  const auto writer_props = ::parquet::WriterProperties::Builder().compression(::parquet::Compression::ZSTD)->build();
  for (uint64_t i = 0; i < num_files; ++i) {
    ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, static_cast<int64_t>(i * rows_per_file), false,
                                                     rows_per_file, 4, 50, {true, true, true, false}));
    const auto path = dir + "/file_" + std::to_string(i) + ".parquet";
    ARROW_ASSIGN_OR_RAISE(auto sink, write_fs->OpenOutputStream(path));
    ARROW_ASSIGN_OR_RAISE(
        auto writer, ::parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), sink, writer_props));
    ARROW_RETURN_NOT_OK(writer->NewBufferedRowGroup());
    ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
    ARROW_RETURN_NOT_OK(writer->Close());
    ARROW_RETURN_NOT_OK(sink->Close());
  }
  return ArnWriteResult{{}, std::move(schema), num_files * rows_per_file, explore_dir, 0};
}

// Shared data path. Authentication fixtures supply properties and URIs; this
// scenario owns the FFI resources and checks discovery, metadata, and reads.
using TestPropertyPairs = std::vector<std::pair<std::string, std::string>>;

class ExternalTableReadScenario {
  public:
  explicit ExternalTableReadScenario(std::string format) : format_(std::move(format)) {}
  ExternalTableReadScenario(const ExternalTableReadScenario&) = delete;
  ExternalTableReadScenario& operator=(const ExternalTableReadScenario&) = delete;

  ~ExternalTableReadScenario() {
    if (manifest_) {
      loon_manifest_destroy(manifest_);
    }
    std::free(manifest_path_);
    loon_properties_free(&properties_);
  }

  void Explore(const ArnWriteResult& data, const std::string& manifest_base, TestPropertyPairs props) {
    if (format_ == LOON_FORMAT_ICEBERG_TABLE) {
      props.emplace_back(PROPERTY_READER_EXTTABLE_SNAPSHOT_ID, std::to_string(data.iceberg_snapshot_id));
    }
    std::vector<const char*> keys, values;
    for (const auto& [key, value] : props) {
      keys.push_back(key.c_str());
      values.push_back(value.c_str());
    }
    auto rc = loon_properties_create(keys.data(), values.data(), keys.size(), &properties_);
    auto release_result = folly::makeGuard([&] { loon_ffi_free_result(&rc); });
    ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
    loon_ffi_free_result(&rc);

    const char* columns[] = {"id", "name", "value"};
    uint64_t num_files = 0;
    rc = loon_exttable_explore(columns, 3, format_.c_str(), manifest_base.c_str(), data.explore_dir.c_str(),
                               &properties_, &num_files, &manifest_path_);
    ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
    ASSERT_GT(num_files, 0u);
    ASSERT_NE(manifest_path_, nullptr);
    loon_ffi_free_result(&rc);
    rc = loon_exttable_read_manifest(manifest_path_, &properties_, &manifest_);
    ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
    ASSERT_NE(manifest_, nullptr);
    ASSERT_EQ(manifest_->column_groups.num_of_column_groups, 1u);
    const auto& group = manifest_->column_groups.column_group_array[0];
    ASSERT_EQ(group.num_of_files, num_files);
    for (uint64_t i = 0; i < num_files; ++i) {
      const auto& file = group.files[i];
      api::ColumnGroupFile cgfile{file.path, file.start_index, file.end_index, {}};
      for (uint32_t p = 0; p < file.num_properties; ++p) {
        cgfile.properties[file.property_keys[p]] = file.property_values[p];
      }
      files_.push_back(std::move(cgfile));
    }
  }

  void CheckFileRows(const uint64_t expected_rows) const {
    for (const auto& file : files_) {
      SCOPED_TRACE(file.path);
      uint64_t rows = 0;
      auto rc = loon_exttable_get_file_info(format_.c_str(), file.path.c_str(), &properties_, &rows);
      auto release_result = folly::makeGuard([&] { loon_ffi_free_result(&rc); });
      ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
      ASSERT_EQ(rows, expected_rows);
    }
  }

  void ReadAndCheck(const std::shared_ptr<arrow::Schema>& schema,
                    const std::vector<api::ColumnGroupFile>& files,
                    const std::shared_ptr<const api::Properties>& read_props,
                    const uint64_t expected_rows) {
    batches_.clear();
    int64_t total_rows = 0;
    for (const auto& file : files) {
      SCOPED_TRACE(file.path);
      ASSERT_AND_ASSIGN(auto reader,
                        FormatReader::create(schema, format_, file, read_props, {"id", "name", "value"}, nullptr));
      ASSERT_AND_ASSIGN(auto row_groups, reader->get_row_group_infos());
      for (size_t i = 0; i < row_groups.size(); ++i) {
        ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
        total_rows += batch->num_rows();
        batches_.push_back(std::move(batch));
      }
    }
    ASSERT_EQ(total_rows, static_cast<int64_t>(expected_rows));
  }

  const std::vector<api::ColumnGroupFile>& files() const { return files_; }
  const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches() const { return batches_; }

  private:
  const std::string format_;
  LoonProperties properties_{};
  char* manifest_path_ = nullptr;
  LoonManifest* manifest_ = nullptr;
  std::vector<api::ColumnGroupFile> files_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
};

// Authentication fixtures configure setup, external-read, and manifest credentials.
// Data tests below each fixture compose the shared scenario with auth-specific checks.
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
    ASSERT_AND_ASSIGN(write_fs_, GetFileSystem(std::make_shared<const api::Properties>(write_props_)));

    FilesystemCache::getInstance().clean();

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "zc/arn-test-" + std::to_string(ts);
  }

  TestPropertyPairs BaseProps() const {
    return {
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
      return WritePlainFormatFile(format, num_rows, write_fs_, std::make_shared<const api::Properties>(write_props_),
                                  test_base_, "s3", address_, arn_bucket_);
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
    const auto path = test_base_ + "/lance";
    return WriteLanceTable(num_rows, std::make_shared<const api::Properties>(write_props_), path,
                           "s3://" + address_ + "/" + arn_bucket_ + "/" + path);
  }

  arrow::Result<ArnWriteResult> CreateIcebergTable(uint64_t num_rows) {
    ARROW_ASSIGN_OR_RAISE(
        const auto provider,
        api::GetValue<std::string>(std::make_shared<const api::Properties>(write_props_), PROPERTY_FS_CLOUD_PROVIDER));
    const std::string scheme = provider == kCloudProviderAliyun ? "oss://" : "s3://";
    return WriteIcebergTable(num_rows, std::make_shared<const api::Properties>(write_props_), write_fs_,
                             scheme + arn_bucket_ + "/" + test_base_ + "/iceberg", address_);
  }
};

TEST_P(ExternalTableArnTest, ReadWithArnRole) {
  const auto& format = GetParam();
  constexpr uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateTestTable(format, num_rows));
  ExternalTableReadScenario data(format);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", BaseProps()));
  ASSERT_NO_FATAL_FAILURE(
      data.ReadAndCheck(result.schema, data.files(), std::make_shared<const api::Properties>(read_props_), num_rows));
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
    ASSERT_AND_ASSIGN(auto reader,
                      FormatReader::create(result.schema, LOON_FORMAT_LANCE_TABLE, result.cgfile,
                                           std::make_shared<const api::Properties>(props), columns, nullptr));
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
    ASSERT_AND_ASSIGN(auto reader,
                      FormatReader::create(result.schema, LOON_FORMAT_LANCE_TABLE, result.cgfile,
                                           std::make_shared<const api::Properties>(props), columns, nullptr));
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
    // The target alone selects IamImpersonateProvider in
    // BuildGcpProviderFromConfig; use_iam=true is accepted but not required.
    // It is kept here so the read slot cannot be mistaken for a Broker
    // configuration (credential_endpoint without use_iam) and to pin the
    // source identity to this VM's service account.
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

  TestPropertyPairs BaseProps() const {
    return {
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
      ARROW_ASSIGN_OR_RAISE(auto write_fs, GetFileSystem(std::make_shared<const api::Properties>(plain_write_props_)));
      return WritePlainFormatFile(format, num_rows, write_fs,
                                  std::make_shared<const api::Properties>(plain_write_props_), test_base_, "gs",
                                  address_, bucket_);
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
    const auto path = test_base_ + "/lance";
    return WriteLanceTable(num_rows, std::make_shared<const api::Properties>(write_props_), path,
                           "gs://" + address_ + "/" + bucket_ + "/" + path);
  }

  arrow::Result<ArnWriteResult> CreateIcebergTable(uint64_t num_rows) {
    return WriteIcebergTable(num_rows, std::make_shared<const api::Properties>(write_props_), nullptr,
                             "s3://" + bucket_ + "/" + test_base_ + "/iceberg", address_, "gs");
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
  constexpr uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateTestTable(format, num_rows));
  ExternalTableReadScenario data(format);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", BaseProps()));
  ASSERT_NO_FATAL_FAILURE(
      data.ReadAndCheck(result.schema, data.files(), std::make_shared<const api::Properties>(read_props_), num_rows));
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

class OneShotHttpPostRelay {
  public:
  explicit OneShotHttpPostRelay(std::string upstream_endpoint)
      : upstream_endpoint_(std::move(upstream_endpoint)),
        acceptor_(io_context_, boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address_v4("127.0.0.1"), 0)) {
    endpoint_ = "http://127.0.0.1:" + std::to_string(acceptor_.local_endpoint().port());
    Accept();
    worker_ = std::thread([this] { io_context_.run(); });
  }

  ~OneShotHttpPostRelay() {
    io_context_.stop();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  const std::string& endpoint() const { return endpoint_; }

  size_t request_count() const { return request_count_.load(std::memory_order_relaxed); }

  std::string error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
  }

  private:
  void Accept() {
    acceptor_.async_accept([this](const boost::system::error_code& error, boost::asio::ip::tcp::socket socket) {
      if (error) {
        return;
      }

      // Keep accepting after the first response so failed refresh attempts are
      // counted even when the provider falls back to its unexpired credentials.
      Accept();

      try {
        boost::beast::flat_buffer buffer;
        boost::beast::http::request<boost::beast::http::string_body> request;
        boost::beast::http::read(socket, buffer, request);
        if (request_count_.fetch_add(1, std::memory_order_relaxed) != 0) {
          boost::beast::http::response<boost::beast::http::string_body> response(
              boost::beast::http::status::service_unavailable, request.version());
          response.keep_alive(false);
          response.body() = "Only the first credential request is forwarded";
          response.prepare_payload();
          boost::beast::http::write(socket, response);
          return;
        }

        std::string response_body;
        char curl_error[CURL_ERROR_SIZE] = {};
        auto* curl = curl_easy_init();
        if (curl == nullptr) {
          throw std::runtime_error("curl_easy_init failed");
        }
        auto* headers = curl_slist_append(nullptr, "Content-Type: application/json");
        headers = curl_slist_append(headers, "Expect:");
        curl_easy_setopt(curl, CURLOPT_URL, upstream_endpoint_.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.body().data());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(request.body().size()));
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 10000L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 30000L);
        curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_error);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &OneShotHttpPostRelay::AppendResponseBody);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);

        auto curl_status = curl_easy_perform(curl);
        long response_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (curl_status != CURLE_OK) {
          throw std::runtime_error(curl_error[0] == '\0' ? curl_easy_strerror(curl_status) : curl_error);
        }
        if (response_code < 100 || response_code > 599) {
          throw std::runtime_error("upstream returned an invalid HTTP status");
        }

        boost::beast::http::response<boost::beast::http::string_body> response(
            static_cast<boost::beast::http::status>(response_code), request.version());
        response.set(boost::beast::http::field::content_type, "application/json");
        response.keep_alive(false);
        response.body() = std::move(response_body);
        response.prepare_payload();
        boost::beast::http::write(socket, response);
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = error.what();
      }

      boost::system::error_code shutdown_error;
      socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both, shutdown_error);
    });
  }

  static size_t AppendResponseBody(char* data, size_t size, size_t count, void* output) {
    auto bytes = size * count;
    static_cast<std::string*>(output)->append(data, bytes);
    return bytes;
  }

  std::string upstream_endpoint_;
  boost::asio::io_context io_context_;
  boost::asio::ip::tcp::acceptor acceptor_;
  std::thread worker_;
  std::string endpoint_;
  std::atomic<size_t> request_count_{0};
  mutable std::mutex mutex_;
  std::string error_;
};

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
    api::SetValue(write_props_, PROPERTY_FS_CREDENTIAL_ENDPOINT, credential_endpoint_.c_str());
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
    api::SetValue(read_props_, "extfs.azsas.credential_endpoint", credential_endpoint_.c_str());
    api::SetValue(read_props_, "extfs.azsas.load_frequency", "3600");
    api::SetValue(read_props_, "extfs.azsas.request_timeout_ms", "10000");
    api::SetValue(read_props_, "extfs.azsas.use_ssl", "true");

    FilesystemCache::getInstance().clean();

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "zc/azure-arn-test-" + std::to_string(ts);
    manifest_root_ = "/tmp/milvus-storage-azure-arn-test";

    api::SetValue(manifest_props_, PROPERTY_FS_STORAGE_TYPE, "local");
    api::SetValue(manifest_props_, PROPERTY_FS_ROOT_PATH, manifest_root_.c_str());

    ASSERT_AND_ASSIGN(write_fs_, GetFileSystem(std::make_shared<const api::Properties>(write_props_)));
    ASSERT_AND_ASSIGN(manifest_fs_, GetFileSystem(std::make_shared<const api::Properties>(manifest_props_)));
  }

  TestPropertyPairs BaseProps() const {
    return {
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
        {"extfs.azsas.credential_endpoint", credential_endpoint_},
        {"extfs.azsas.load_frequency", "3600"},
        {"extfs.azsas.request_timeout_ms", "10000"},
        {"extfs.azsas.use_ssl", "true"},
    };
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
      return WritePlainFormatFile(format, num_rows, write_fs_, std::make_shared<const api::Properties>(write_props_),
                                  test_base_, "az", address_, container_);
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
    const auto path = test_base_ + "/lance";
    return WriteLanceTable(num_rows, std::make_shared<const api::Properties>(write_props_), path,
                           "az://" + address_ + "/" + container_ + "/" + path);
  }

  arrow::Result<ArnWriteResult> CreateIcebergTable(uint64_t num_rows) {
    return WriteIcebergTable(num_rows, std::make_shared<const api::Properties>(write_props_), write_fs_,
                             "abfss://" + container_ + "/" + test_base_ + "/iceberg", address_, "", "azure");
  }
};

TEST_P(ExternalTableAzureArnTest, ReadWithBrokeredSas) {
  const auto& format = GetParam();
  constexpr uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateTestTable(format, num_rows));
  ExternalTableReadScenario data(format);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", BaseProps()));
  ASSERT_NO_FATAL_FAILURE(
      data.ReadAndCheck(result.schema, data.files(), std::make_shared<const api::Properties>(read_props_), num_rows));
}

TEST_P(ExternalTableAzureArnTest, CachedReadsFetchBrokeredSasOnce) {
  const auto& format = GetParam();
  if (format != LOON_FORMAT_LANCE_TABLE && format != LOON_FORMAT_ICEBERG_TABLE) {
    GTEST_SKIP() << "Cache reuse is covered only by Lance and Iceberg";
  }

  const uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateTestTable(format, num_rows));

  OneShotHttpPostRelay broker_relay(credential_endpoint_);
  auto cache_read_props = read_props_;
  api::SetValue(cache_read_props, "extfs.azsas.credential_endpoint", broker_relay.endpoint().c_str());
  ASSERT_AND_ASSIGN(auto fs_config,
                    FilesystemCache::resolve_config(std::make_shared<const api::Properties>(cache_read_props),
                                                    result.explore_dir.c_str()));

  constexpr size_t cache_hit_iterations = 10;
  if (format == LOON_FORMAT_LANCE_TABLE) {
    auto storage_options = lance::ToReaderOptions(fs_config);
    auto cache_key = storage_options.find("milvus_fs_cache_key");
    ASSERT_NE(cache_key, storage_options.end());
    ASSERT_EQ(cache_key->second, fs_config.GetCacheKey());

    std::vector<std::string> columns = {"id", "name", "value"};
    std::shared_ptr<FormatReader> reader;
    for (size_t iteration = 0; iteration < cache_hit_iterations; ++iteration) {
      SCOPED_TRACE(::testing::Message() << "cache hit iteration " << iteration);
      ASSERT_AND_ASSIGN(
          reader, FormatReader::create(result.schema, format, result.cgfile,
                                       std::make_shared<const api::Properties>(cache_read_props), columns, nullptr));
    }

    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
    ValidateRowGroupInfos(rg_infos, num_rows);
    int64_t total_rows = 0;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total_rows += batch->num_rows();
    }
    ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));
  } else {
    auto snapshot_id = std::to_string(result.iceberg_snapshot_id);
    api::SetValue(cache_read_props, PROPERTY_READER_EXTTABLE_SNAPSHOT_ID, snapshot_id.c_str());

    auto reader_options = iceberg::ToReaderOptions(fs_config);
    EXPECT_EQ(reader_options, (std::unordered_map<std::string, std::string>{
                                  {"milvus_fs_is_local", "false"}, {"milvus_fs_bucket", fs_config.bucket_name}}));

    ASSERT_AND_ASSIGN(auto* iceberg_format, Format::get(format));
    for (size_t iteration = 0; iteration < cache_hit_iterations; ++iteration) {
      SCOPED_TRACE(::testing::Message() << "cache hit iteration " << iteration);
      ASSERT_AND_ASSIGN(auto files, iceberg_format->explore(result.explore_dir,
                                                            std::make_shared<const api::Properties>(cache_read_props)));
      ASSERT_FALSE(files.empty());
    }
  }

  ASSERT_EQ(broker_relay.request_count(), 1u);
  ASSERT_TRUE(broker_relay.error().empty()) << broker_relay.error();
}

INSTANTIATE_TEST_SUITE_P(
    AzureArnFormats,
    ExternalTableAzureArnTest,
    ::testing::Values(LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE, LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX));

// ===========================================================================
// External-table reads through an S3-compatible temporary-credential broker.
// The cloud provider is supplied by the environment (e.g. aws, aliyun, tencent,
// huawei, volcengine); the fixture does not restrict it to AWS. The configured
// storage provider and broker must support the requested AssumeRole contract.
// Setup credentials are used only to prepare/clean up data. Manifest storage is local.
// ===========================================================================
#define S3_BROKER_ENV_CLOUD_PROVIDER "S3_BROKER_TEST_ENV_CLOUD_PROVIDER"
#define S3_BROKER_ENV_ADDRESS "S3_BROKER_TEST_ENV_ADDRESS"
#define S3_BROKER_ENV_REGION "S3_BROKER_TEST_ENV_REGION"
#define S3_BROKER_ENV_BUCKET "S3_BROKER_TEST_ENV_BUCKET"
#define S3_BROKER_ENV_ACCESS_KEY "S3_BROKER_TEST_ENV_ACCESS_KEY"
#define S3_BROKER_ENV_SECRET_KEY "S3_BROKER_TEST_ENV_SECRET_KEY"
#define S3_BROKER_ENV_ROLE_ARN "S3_BROKER_TEST_ENV_ROLE_ARN"
#define S3_BROKER_ENV_CREDENTIAL_ENDPOINT "S3_BROKER_TEST_ENV_CREDENTIAL_ENDPOINT"
// Optional: use the default credential chain (including session tokens) for setup.
#define S3_BROKER_ENV_USE_IAM "S3_BROKER_TEST_ENV_USE_IAM"
// Optional: trust-policy external ID and HTTP storage endpoints for local fixtures.
#define S3_BROKER_ENV_EXTERNAL_ID "S3_BROKER_TEST_ENV_EXTERNAL_ID"
#define S3_BROKER_ENV_USE_SSL "S3_BROKER_TEST_ENV_USE_SSL"

class ExternalTableS3BrokerTest : public ExternalTableArnTest {
  protected:
  void SetUp() override {
    cloud_provider_ = GetEnvVar(S3_BROKER_ENV_CLOUD_PROVIDER).ValueOr("");
    address_ = GetEnvVar(S3_BROKER_ENV_ADDRESS).ValueOr("");
    region_ = GetEnvVar(S3_BROKER_ENV_REGION).ValueOr("");
    arn_bucket_ = GetEnvVar(S3_BROKER_ENV_BUCKET).ValueOr("");
    arn_ak_ = GetEnvVar(S3_BROKER_ENV_ACCESS_KEY).ValueOr("");
    arn_sk_ = GetEnvVar(S3_BROKER_ENV_SECRET_KEY).ValueOr("");
    role_arn_ = GetEnvVar(S3_BROKER_ENV_ROLE_ARN).ValueOr("");
    const auto endpoint = GetEnvVar(S3_BROKER_ENV_CREDENTIAL_ENDPOINT).ValueOr("");
    const auto external_id = GetEnvVar(S3_BROKER_ENV_EXTERNAL_ID).ValueOr("");
    const auto use_ssl = GetEnvVar(S3_BROKER_ENV_USE_SSL).ValueOr("true");
    const auto use_iam = GetEnvVar(S3_BROKER_ENV_USE_IAM).ValueOr("false");
    if (cloud_provider_.empty() || address_.empty() || region_.empty() || arn_bucket_.empty() ||
        (use_iam != "true" && (arn_ak_.empty() || arn_sk_.empty())) || role_arn_.empty() || endpoint.empty()) {
      GTEST_SKIP() << "S3 broker tests require S3_BROKER_TEST_ENV_{CLOUD_PROVIDER,ADDRESS,REGION,BUCKET,"
                      "ROLE_ARN,CREDENTIAL_ENDPOINT} and either ACCESS_KEY/SECRET_KEY or USE_IAM=true";
    }

    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "zc/s3-broker-test-" + std::to_string(ts);
    const auto manifest_root = (std::filesystem::temp_directory_path() / "milvus-storage-s3-broker-test").string();
    FilesystemCache::getInstance().clean();

    // Native Lance/Iceberg writers use the S3-compatible backend for these
    // providers. Reads retain the selected provider to exercise its broker path.
    const auto write_provider = (cloud_provider_ == kCloudProviderTencent || cloud_provider_ == kCloudProviderHuawei ||
                                 cloud_provider_ == kCloudProviderVolcengine)
                                    ? kCloudProviderAWS
                                    : cloud_provider_;
    if (GetParam() == LOON_FORMAT_LANCE_TABLE && write_provider != cloud_provider_) {
      // Lance's URI builder also rejects these providers on the read path.
      GTEST_SKIP() << "Lance reads do not support cloud provider: " << cloud_provider_;
    }
    const std::vector<std::pair<std::string, std::string>> write_properties = {
        {PROPERTY_FS_STORAGE_TYPE, "remote"},    {PROPERTY_FS_CLOUD_PROVIDER, write_provider},
        {PROPERTY_FS_ADDRESS, address_},         {PROPERTY_FS_REGION, region_},
        {PROPERTY_FS_BUCKET_NAME, arn_bucket_},  {PROPERTY_FS_ACCESS_KEY_ID, arn_ak_},
        {PROPERTY_FS_ACCESS_KEY_VALUE, arn_sk_}, {PROPERTY_FS_USE_SSL, use_ssl},
        {PROPERTY_FS_USE_IAM, use_iam},
    };
    for (const auto& [key, value] : write_properties) {
      const auto error = api::SetValue(write_props_, key.c_str(), value.c_str());
      ASSERT_FALSE(error.has_value()) << error.value_or("");
    }
    ASSERT_AND_ASSIGN(write_fs_, GetFileSystem(std::make_shared<const api::Properties>(write_props_)));

    // Only one broker response is available. Repeated reads must reuse it,
    // and the request count detects an accidental fallback to another auth path.
    broker_relay_ = std::make_unique<OneShotHttpPostRelay>(endpoint);
    property_pairs_ = {
        {PROPERTY_FS_STORAGE_TYPE, "local"},
        {PROPERTY_FS_ROOT_PATH, manifest_root},
        {"extfs.s3broker.storage_type", "remote"},
        {"extfs.s3broker.cloud_provider", cloud_provider_},
        {"extfs.s3broker.address", address_},
        {"extfs.s3broker.region", region_},
        {"extfs.s3broker.bucket_name", arn_bucket_},
        {"extfs.s3broker.use_ssl", use_ssl},
        {"extfs.s3broker.role_arn", role_arn_},
        {"extfs.s3broker.external_id", external_id},
        {"extfs.s3broker.credential_endpoint", broker_relay_->endpoint()},
        {"extfs.s3broker.load_frequency", "3600"},
        {"extfs.s3broker.request_timeout_ms", "75000"},
    };
    for (const auto& [key, value] : property_pairs_) {
      const auto error = api::SetValue(read_props_, key.c_str(), value.c_str());
      ASSERT_FALSE(error.has_value()) << error.value_or("");
    }
    ASSERT_AND_ASSIGN(manifest_fs_, GetFileSystem(std::make_shared<const api::Properties>(read_props_)));
  }

  void TearDown() override {
    if (manifest_fs_) {
      (void)DeleteTestDir(manifest_fs_, test_base_);
    }
    ExternalTableArnTest::TearDown();
  }

  std::string cloud_provider_;
  std::vector<std::pair<std::string, std::string>> property_pairs_;
  std::unique_ptr<OneShotHttpPostRelay> broker_relay_;
  ArrowFileSystemPtr manifest_fs_;
};

TEST_P(ExternalTableS3BrokerTest, ReadWithBrokeredCredentials) {
  const auto& format = GetParam();
  constexpr uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateTestTable(format, num_rows));
  if (format == LOON_FORMAT_ICEBERG_TABLE) {
    api::SetValue(read_props_, PROPERTY_READER_EXTTABLE_SNAPSHOT_ID,
                  std::to_string(result.iceberg_snapshot_id).c_str());
  }
  ExternalTableReadScenario data(format);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", property_pairs_));
  ASSERT_NO_FATAL_FAILURE(
      data.ReadAndCheck(result.schema, data.files(), std::make_shared<const api::Properties>(read_props_), num_rows));
  EXPECT_EQ(broker_relay_->request_count(), 1u) << "External reads must use the configured credential broker";
  EXPECT_TRUE(broker_relay_->error().empty()) << broker_relay_->error();
}

TEST_P(ExternalTableS3BrokerTest, CachedReadsFetchBrokerCredentialsOnce) {
  const auto& format = GetParam();
  constexpr uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateTestTable(format, num_rows));
  if (format == LOON_FORMAT_ICEBERG_TABLE) {
    const auto snapshot = std::to_string(result.iceberg_snapshot_id);
    api::SetValue(read_props_, PROPERTY_READER_EXTTABLE_SNAPSHOT_ID, snapshot.c_str());
  }
  ASSERT_AND_ASSIGN(auto* format_impl, Format::get(format));
  ASSERT_AND_ASSIGN(auto cached_fs, FilesystemCache::getInstance().get(
                                        std::make_shared<const api::Properties>(read_props_), result.explore_dir));
  for (size_t iteration = 0; iteration < 3; ++iteration) {
    SCOPED_TRACE(iteration);
    ASSERT_AND_ASSIGN(auto fs, FilesystemCache::getInstance().get(std::make_shared<const api::Properties>(read_props_),
                                                                  result.explore_dir));
    EXPECT_EQ(fs.get(), cached_fs.get());
    ASSERT_AND_ASSIGN(auto files,
                      format_impl->explore(result.explore_dir, std::make_shared<const api::Properties>(read_props_)));
    ASSERT_FALSE(files.empty());
    ExternalTableReadScenario data(format);
    ASSERT_NO_FATAL_FAILURE(
        data.ReadAndCheck(result.schema, files, std::make_shared<const api::Properties>(read_props_), num_rows));
    EXPECT_EQ(broker_relay_->request_count(), 1u);
    EXPECT_TRUE(broker_relay_->error().empty()) << broker_relay_->error();
  }
}

INSTANTIATE_TEST_SUITE_P(
    S3BrokerFormats,
    ExternalTableS3BrokerTest,
    ::testing::Values(LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE, LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX));

// ===========================================================================
// Integration tests for reading external OSS/COS tables via role ARN.
//
// Exercises both halves of the Aliyun role_arn feature end-to-end:
//   * C++ native S3FS path — our-side manifest storage through
//     S3FileSystemProducer's Aliyun ARN dispatch
//     (s3_filesystem_producer.cpp), only reached if our_cloud_provider=aliyun.
//   * C++ filesystem path — customer-side Lance reads call the same
//     S3FileSystemProducer-owned Aliyun credentials through filesystem_c.
//     Rust adapts those reads to Lance's ObjectStore interface only.
//
// Test data is written with explicit AKSK (cloud_provider=aliyun + AK/SK),
// then read back using only the role_arn. This mirrors the AWS ARN test.
//
// Iceberg planning now reads through the same cached C++ filesystem as the
// other filesystem-backed formats. The role and credential refresh lifecycle
// therefore remains owned by the C++ Aliyun filesystem provider.
//
// Required environment variables (all must be set; test is skipped otherwise):
//
// Our-side bucket (for writing manifest). Unlike the AWS fixture, this
// uses static AK/SK rather than `use_iam=true`: our Aliyun credential provider
// implements AssumeRoleWithOIDC (the K8s RAM-for-Service-Account flow), not
// ECS instance RAM role. A vanilla ECS host without the RAM-for-SA env vars
// cannot use `use_iam=true` for the our-side bucket, so the test uses AK/SK
// there. Only the *customer-side* read path exercises the role_arn flow.
//   OUR_TEST_ENV_ADDRESS, OUR_TEST_ENV_BUCKET, OUR_TEST_ENV_REGION,
//   OUR_TEST_ENV_CLOUD_PROVIDER (set to "aliyun" or "tencent"),
//   OUR_TEST_ENV_ACCESS_KEY, OUR_TEST_ENV_SECRET_KEY
//
// Customer-side OSS/COS bucket (ARN-based, for reading external data):
// ===========================================================================
#define OUR_ENV_ACCESS_KEY "OUR_TEST_ENV_ACCESS_KEY"                          // AK for our-side bucket
#define OUR_ENV_SECRET_KEY "OUR_TEST_ENV_SECRET_KEY"                          // SK for our-side bucket
#define S3_COMP_ARN_ENV_CLOUD_PROVIDER "S3_COMP_ARN_TEST_ENV_CLOUD_PROVIDER"  // "aliyun" or "tencent"
#define S3_COMP_ARN_ENV_ADDRESS "S3_COMP_ARN_TEST_ENV_ADDRESS"                // endpoint without scheme or bucket
#define S3_COMP_ARN_ENV_REGION "S3_COMP_ARN_TEST_ENV_REGION"                  // bucket region
#define S3_COMP_ARN_ENV_BUCKET "S3_COMP_ARN_TEST_ENV_BUCKET"                  // target bucket
#define S3_COMP_ARN_ENV_ACCESS_KEY "S3_COMP_ARN_TEST_ENV_ACCESS_KEY"          // AK with write to bucket
#define S3_COMP_ARN_ENV_SECRET_KEY "S3_COMP_ARN_TEST_ENV_SECRET_KEY"          // SK
#define S3_COMP_ARN_ENV_ROLE_ARN "S3_COMP_ARN_TEST_ENV_ROLE_ARN"              // target role to assume

// Optional sts:ExternalId for the target role's trust policy.
#define S3_COMP_ARN_ENV_EXTERNAL_ID "S3_COMP_ARN_TEST_ENV_EXTERNAL_ID"

// Aliyun OIDC machine identity (pod-level, not per-test): the process env must
// carry ALIBABA_CLOUD_OIDC_TOKEN_FILE and ALIBABA_CLOUD_OIDC_PROVIDER_ARN. On an
// Aliyun RAM-for-Service-Account pod these are K8s-injected; we do NOT set
// them here — the dispatch in s3_filesystem_producer.cpp fails fast if they're
// missing, which is the intended safety net.

// Shared role-ARN fixture. Tencent runs the Parquet case; the other formats
// below retain their existing Aliyun-only coverage.
// Both providers use S3_COMP_ARN_TEST_ENV_* for the customer-side bucket and
// OUR_TEST_ENV_* for the manifest bucket. Select the fixture's provider with
// S3_COMP_ARN_TEST_ENV_CLOUD_PROVIDER; other providers' fixtures are skipped.
// Tencent additionally requires TKE_{REGION,ROLE_ARN,WEB_IDENTITY_TOKEN_FILE,
// PROVIDER_ID}. The target role must differ from the TKE workload role.
class ExternalTableRoleArnTest : public ::testing::Test {
  protected:
  explicit ExternalTableRoleArnTest(const char* cloud_provider) : cloud_provider_(cloud_provider) {}

  void ReadTwoParquetFilesWithArnRole();
  void SetUp() override {
    if (GetEnvVar(S3_COMP_ARN_ENV_CLOUD_PROVIDER).ValueOr("") != cloud_provider_) {
      GTEST_SKIP() << "Set " << S3_COMP_ARN_ENV_CLOUD_PROVIDER << "=" << cloud_provider_ << " to run this fixture";
    }

    // Our-side bucket (AK/SK-based; see fixture-level comment for why not IAM).
    our_address_ = GetEnvVar(OUR_ENV_ADDRESS).ValueOr("");
    our_bucket_ = GetEnvVar(OUR_ENV_BUCKET).ValueOr("");
    our_region_ = GetEnvVar(OUR_ENV_REGION).ValueOr("");
    our_cloud_provider_ = GetEnvVar(OUR_ENV_CLOUD_PROVIDER).ValueOr("");
    our_ak_ = GetEnvVar(OUR_ENV_ACCESS_KEY).ValueOr("");
    our_sk_ = GetEnvVar(OUR_ENV_SECRET_KEY).ValueOr("");

    address_ = GetEnvVar(S3_COMP_ARN_ENV_ADDRESS).ValueOr("");
    region_ = GetEnvVar(S3_COMP_ARN_ENV_REGION).ValueOr("");
    arn_bucket_ = GetEnvVar(S3_COMP_ARN_ENV_BUCKET).ValueOr("");
    arn_ak_ = GetEnvVar(S3_COMP_ARN_ENV_ACCESS_KEY).ValueOr("");
    arn_sk_ = GetEnvVar(S3_COMP_ARN_ENV_SECRET_KEY).ValueOr("");
    role_arn_ = GetEnvVar(S3_COMP_ARN_ENV_ROLE_ARN).ValueOr("");
    external_id_ = GetEnvVar(S3_COMP_ARN_ENV_EXTERNAL_ID).ValueOr("");

    if (our_address_.empty() || our_bucket_.empty() || our_cloud_provider_.empty() || our_ak_.empty() ||
        our_sk_.empty() || address_.empty() || region_.empty() || arn_bucket_.empty() || arn_ak_.empty() ||
        arn_sk_.empty() || role_arn_.empty()) {
      GTEST_SKIP() << cloud_provider_
                   << " ARN test requires OUR_TEST_ENV_{ADDRESS,BUCKET,REGION,"
                      "CLOUD_PROVIDER,ACCESS_KEY,SECRET_KEY} and "
                      "S3_COMP_ARN_TEST_ENV_{ADDRESS,REGION,BUCKET,ACCESS_KEY,SECRET_KEY,ROLE_ARN}";
    }

    if (cloud_provider_ == kCloudProviderTencent) {
      for (const char* name : {"TKE_REGION", "TKE_ROLE_ARN", "TKE_WEB_IDENTITY_TOKEN_FILE", "TKE_PROVIDER_ID"}) {
        if (GetEnvVar(name).ValueOr("").empty()) {
          GTEST_SKIP() << "Tencent ARN test requires " << name << " from the TKE workload identity";
        }
      }
      ASSERT_NE(role_arn_, GetEnvVar("TKE_ROLE_ARN").ValueOr(""))
          << "Use a distinct target role to exercise the second STS hop";
    } else {
      // Two machine-identity modes are supported; which one we need depends on
      // ALIYUN_ROLE_ARN_AUTH_MODE (kept in lockstep with the
      // dispatch in s3_filesystem_producer.cpp):
      //   - "ram":  ECS IMDS → sts:AssumeRole. No env vars required; the
      //             metadata service supplies the caller identity. Only runs
      //             on an ECS with a RAM role attached.
      //   - default / "oidc": legacy AssumeRoleWithOIDC. Requires
      //             ALIBABA_CLOUD_OIDC_TOKEN_FILE + _PROVIDER_ARN in process
      //             env; without them the C++ filesystem path fails fast with
      //             a clear configuration error.
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
    }

    // Static credentials are used only to prepare and clean up test data.
    api::SetValue(write_props_, PROPERTY_FS_STORAGE_TYPE, "remote");
    api::SetValue(write_props_, PROPERTY_FS_CLOUD_PROVIDER, cloud_provider_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_ADDRESS, address_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_BUCKET_NAME, arn_bucket_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_REGION, region_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_ACCESS_KEY_ID, arn_ak_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_ACCESS_KEY_VALUE, arn_sk_.c_str());
    api::SetValue(write_props_, PROPERTY_FS_USE_SSL, "true");

    // Read through the workload identity -> target role chain, without AK/SK.
    api::SetValue(read_props_, "extfs.arn.storage_type", "remote");
    api::SetValue(read_props_, "extfs.arn.cloud_provider", cloud_provider_.c_str());
    api::SetValue(read_props_, "extfs.arn.address", address_.c_str());
    api::SetValue(read_props_, "extfs.arn.bucket_name", arn_bucket_.c_str());
    api::SetValue(read_props_, "extfs.arn.region", region_.c_str());
    api::SetValue(read_props_, "extfs.arn.use_ssl", "true");
    api::SetValue(read_props_, "extfs.arn.role_arn", role_arn_.c_str());
    if (!external_id_.empty()) {
      api::SetValue(read_props_, "extfs.arn.external_id", external_id_.c_str());
    }

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "zc/" + cloud_provider_ + "-arn-test-" + std::to_string(ts);

    // Create write filesystem for cleanup.
    ASSERT_AND_ASSIGN(write_fs_, GetFileSystem(std::make_shared<const api::Properties>(write_props_)));

    FilesystemCache::getInstance().clean();
  }

  void TearDown() override {
    if (manifest_fs_) {
      EXPECT_TRUE(DeleteTestDir(manifest_fs_, test_base_ + "/manifest").ok());
    }
    if (write_fs_) {
      (void)DeleteTestDir(write_fs_, test_base_);
    }
    FilesystemCache::getInstance().clean();
  }

  // FFI properties payload shared by all four formats. Iceberg additionally
  // needs PROPERTY_READER_EXTTABLE_SNAPSHOT_ID; the shared scenario appends it.
  std::vector<std::pair<std::string, std::string>> BaseProps() const {
    std::vector<std::pair<std::string, std::string>> props = {
        {PROPERTY_FS_STORAGE_TYPE, "remote"},    {PROPERTY_FS_CLOUD_PROVIDER, our_cloud_provider_},
        {PROPERTY_FS_ADDRESS, our_address_},     {PROPERTY_FS_BUCKET_NAME, our_bucket_},
        {PROPERTY_FS_REGION, our_region_},       {PROPERTY_FS_ACCESS_KEY_ID, our_ak_},
        {PROPERTY_FS_ACCESS_KEY_VALUE, our_sk_}, {PROPERTY_FS_USE_SSL, "true"},
        {"extfs.arn.storage_type", "remote"},    {"extfs.arn.cloud_provider", cloud_provider_},
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
    const auto path = test_base_ + "/lance";
    return WriteLanceTable(num_rows, std::make_shared<const api::Properties>(write_props_), path,
                           "oss://" + address_ + "/" + arn_bucket_ + "/" + path);
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
    return WriteIcebergTable(num_rows, std::make_shared<const api::Properties>(write_props_), write_fs_,
                             "oss://" + arn_bucket_ + "/" + test_base_ + "/iceberg", address_);
  }

  // Writes `num_files` parquet files of `rows_per_file` rows each into a fresh
  // subdirectory under test_base_, using the AK/SK-backed write_fs_ (so the
  // write itself never touches the role_arn path; that's exercised by the
  // explore/read steps below). Returns the (full URI) explore directory the
  // caller passes to loon_exttable_explore, plus the schema for FormatReader.

  arrow::Result<ArnWriteResult> CreateParquetFiles(uint64_t num_files, uint64_t rows_per_file) {
    const auto dir = test_base_ + "/parquet";
    const std::string scheme = cloud_provider_ == kCloudProviderTencent ? "s3://" : "oss://";
    return WriteParquetFiles(num_files, rows_per_file, write_fs_, dir,
                             scheme + address_ + "/" + arn_bucket_ + "/" + dir + "/");
  }

  const std::string cloud_provider_;

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
  std::string external_id_;

  api::Properties write_props_;
  api::Properties read_props_;
  ArrowFileSystemPtr write_fs_;
  ArrowFileSystemPtr manifest_fs_;
  std::string test_base_;
};

class ExternalTableAliyunArnTest : public ExternalTableRoleArnTest {
  protected:
  ExternalTableAliyunArnTest() : ExternalTableRoleArnTest(kCloudProviderAliyun) {}
};

class ExternalTableTencentArnTest : public ExternalTableRoleArnTest {
  protected:
  ExternalTableTencentArnTest() : ExternalTableRoleArnTest(kCloudProviderTencent) {}
};

TEST_F(ExternalTableAliyunArnTest, ReadLanceWithArnRole) {
  constexpr uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateLanceTable(num_rows));
  ExternalTableReadScenario data(LOON_FORMAT_LANCE_TABLE);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", BaseProps()));
  ASSERT_NO_FATAL_FAILURE(
      data.ReadAndCheck(result.schema, data.files(), std::make_shared<const api::Properties>(read_props_), num_rows));
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
// is passing PROPERTY_READER_EXTTABLE_SNAPSHOT_ID so loon_exttable_explore pins the
// scan to the snapshot we just wrote.
TEST_F(ExternalTableAliyunArnTest, ReadIcebergWithArnRole) {
  constexpr uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateIcebergTable(num_rows));
  ExternalTableReadScenario data(LOON_FORMAT_ICEBERG_TABLE);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", BaseProps()));
  ASSERT_NO_FATAL_FAILURE(
      data.ReadAndCheck(result.schema, data.files(), std::make_shared<const api::Properties>(read_props_), num_rows));
}

// Parquet variant: writes 2 parquet files with AK/SK to the customer bucket,
// then exercises loon_exttable_explore + loon_exttable_get_file_info via the
// role_arn. Verifies the per-file row count from get_file_info before falling
// through to the same manifest-read + FormatReader pipeline as the Lance test.
void ExternalTableRoleArnTest::ReadTwoParquetFilesWithArnRole() {
  constexpr uint64_t num_files = 2;
  constexpr uint64_t rows_per_file = 50;
  constexpr uint64_t total_rows = num_files * rows_per_file;
  ASSERT_AND_ASSIGN(auto result, CreateParquetFiles(num_files, rows_per_file));
  const auto props = BaseProps();
  api::Properties manifest_props;
  for (const auto& [key, value] : props) {
    api::SetValue(manifest_props, key.c_str(), value.c_str());
  }
  ASSERT_AND_ASSIGN(manifest_fs_, GetFileSystem(std::make_shared<const api::Properties>(manifest_props)));
  ExternalTableReadScenario data(LOON_FORMAT_PARQUET);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", props));
  ASSERT_EQ(data.files().size(), num_files);
  ASSERT_NO_FATAL_FAILURE(data.CheckFileRows(rows_per_file));
  ASSERT_NO_FATAL_FAILURE(
      data.ReadAndCheck(result.schema, data.files(), std::make_shared<const api::Properties>(read_props_), total_rows));
  std::vector<int64_t> ids;
  for (const auto& batch : data.batches()) {
    ASSERT_TRUE(batch->schema()->Equals(result.schema)) << batch->schema()->ToString();
    auto id_column = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("id"));
    auto name_column = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName("name"));
    auto value_column = std::static_pointer_cast<arrow::DoubleArray>(batch->GetColumnByName("value"));
    ASSERT_EQ(id_column->null_count(), 0);
    ASSERT_EQ(name_column->null_count(), 0);
    ASSERT_EQ(value_column->null_count(), 0);
    for (int64_t row = 0; row < batch->num_rows(); ++row) {
      const auto id = id_column->Value(row);
      ids.push_back(id);
      EXPECT_EQ(name_column->GetString(row), "name_" + std::to_string(id));
      EXPECT_DOUBLE_EQ(value_column->Value(row), id * 1.5);
    }
  }
  ASSERT_EQ(ids.size(), total_rows);
  std::sort(ids.begin(), ids.end());
  for (uint64_t i = 0; i < total_rows; ++i) {
    EXPECT_EQ(ids[i], static_cast<int64_t>(i));
  }
}

TEST_F(ExternalTableAliyunArnTest, ReadTwoParquetFilesWithArnRole) { ReadTwoParquetFilesWithArnRole(); }

TEST_F(ExternalTableTencentArnTest, ReadTwoParquetFilesWithArnRole) { ReadTwoParquetFilesWithArnRole(); }

// Vortex follows the same plain-format path as Parquet. This verifies that
// discovery, metadata loading, and data reads all use the customer role ARN.
TEST_F(ExternalTableAliyunArnTest, ReadVortexWithArnRole) {
  constexpr uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, WritePlainFormatFile(LOON_FORMAT_VORTEX, num_rows, write_fs_,
                                                      std::make_shared<const api::Properties>(write_props_), test_base_,
                                                      "oss", address_, arn_bucket_));
  ExternalTableReadScenario data(LOON_FORMAT_VORTEX);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", BaseProps()));
  ASSERT_EQ(data.files().size(), 1u);
  ASSERT_NO_FATAL_FAILURE(data.CheckFileRows(num_rows));
  ASSERT_NO_FATAL_FAILURE(
      data.ReadAndCheck(result.schema, data.files(), std::make_shared<const api::Properties>(read_props_), num_rows));
}

// ===========================================================================
// Aliyun OIDC chain integration test (`AliyunOIDCAssumeRoleChainProvider` for
// filesystem-backed formats, `apply_oidc_chain_if_requested` for Iceberg).
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
// Optional `S3_COMP_ARN_TEST_ENV_EXTERNAL_ID` — set only if the customer's
// role trust policy carries a matching `sts:ExternalId` condition. Empty
// means no `ExternalId` parameter is sent on the step-2 AssumeRole.
// ===========================================================================
class ExternalTableAliyunOIDCArnTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (GetEnvVar(S3_COMP_ARN_ENV_CLOUD_PROVIDER).ValueOr("") != kCloudProviderAliyun) {
      GTEST_SKIP() << "Set " << S3_COMP_ARN_ENV_CLOUD_PROVIDER << "=aliyun to run this fixture";
    }

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

    address_ = GetEnvVar(S3_COMP_ARN_ENV_ADDRESS).ValueOr("");
    region_ = GetEnvVar(S3_COMP_ARN_ENV_REGION).ValueOr("");
    arn_bucket_ = GetEnvVar(S3_COMP_ARN_ENV_BUCKET).ValueOr("");
    arn_ak_ = GetEnvVar(S3_COMP_ARN_ENV_ACCESS_KEY).ValueOr("");
    arn_sk_ = GetEnvVar(S3_COMP_ARN_ENV_SECRET_KEY).ValueOr("");
    role_arn_ = GetEnvVar(S3_COMP_ARN_ENV_ROLE_ARN).ValueOr("");
    // ExternalId is optional. Empty == not sent (the chain provider's
    // step-2 body omits the parameter entirely; see
    // TestAliyunOIDCChainProviderForwardsExternalIdToStep2Only).
    external_id_ = GetEnvVar(S3_COMP_ARN_ENV_EXTERNAL_ID).ValueOr("");

    if (our_address_.empty() || our_bucket_.empty() || our_cloud_provider_.empty() || our_ak_.empty() ||
        our_sk_.empty() || address_.empty() || region_.empty() || arn_bucket_.empty() || arn_ak_.empty() ||
        arn_sk_.empty() || role_arn_.empty()) {
      GTEST_SKIP() << "Aliyun OIDC chain test requires env vars: " << OUR_ENV_ADDRESS << ", " << OUR_ENV_BUCKET << ", "
                   << OUR_ENV_REGION << ", " << OUR_ENV_CLOUD_PROVIDER << ", " << OUR_ENV_ACCESS_KEY << ", "
                   << OUR_ENV_SECRET_KEY << ", " << S3_COMP_ARN_ENV_ADDRESS << ", " << S3_COMP_ARN_ENV_REGION << ", "
                   << S3_COMP_ARN_ENV_BUCKET << ", " << S3_COMP_ARN_ENV_ACCESS_KEY << ", " << S3_COMP_ARN_ENV_SECRET_KEY
                   << ", " << S3_COMP_ARN_ENV_ROLE_ARN << " (plus optional " << S3_COMP_ARN_ENV_EXTERNAL_ID << ")";
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

    ASSERT_AND_ASSIGN(write_fs_, GetFileSystem(std::make_shared<const api::Properties>(write_props_)));
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

  class ScopedEnvVarOverride {
 public:
    ScopedEnvVarOverride(const char* name, const std::string& value) : name_(name) {
      const char* old_value = std::getenv(name_.c_str());
      if (old_value != nullptr) {
        had_old_value_ = true;
        old_value_ = old_value;
      }
      setenv(name_.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVarOverride() {
      if (had_old_value_) {
        setenv(name_.c_str(), old_value_.c_str(), 1);
      } else {
        unsetenv(name_.c_str());
      }
    }

    ScopedEnvVarOverride(const ScopedEnvVarOverride&) = delete;
    ScopedEnvVarOverride& operator=(const ScopedEnvVarOverride&) = delete;

 private:
    std::string name_;
    std::string old_value_;
    bool had_old_value_ = false;
  };

  // FFI properties payload shared by all four formats. Iceberg additionally
  // needs PROPERTY_READER_EXTTABLE_SNAPSHOT_ID; the shared scenario appends it.
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
    const auto path = test_base_ + "/lance";
    return WriteLanceTable(num_rows, std::make_shared<const api::Properties>(write_props_), path,
                           "oss://" + address_ + "/" + arn_bucket_ + "/" + path);
  }

  arrow::Result<ArnWriteResult> CreateIcebergTable(uint64_t num_rows) {
    return WriteIcebergTable(num_rows, std::make_shared<const api::Properties>(write_props_), write_fs_,
                             "oss://" + arn_bucket_ + "/" + test_base_ + "/iceberg", address_);
  }

  arrow::Result<ArnWriteResult> CreateParquetFiles(uint64_t num_files, uint64_t rows_per_file) {
    const auto dir = test_base_ + "/parquet";
    const std::string scheme = "oss://";
    return WriteParquetFiles(num_files, rows_per_file, write_fs_, dir,
                             scheme + address_ + "/" + arn_bucket_ + "/" + dir + "/");
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

// Lance + OIDC chain end-to-end. Writes with AKSK, then reads back through
// loon_exttable_explore and FormatReader using the C++ filesystem's OIDC
// chain.
TEST_F(ExternalTableAliyunOIDCArnTest, ReadLanceWithOIDCChain) {
  constexpr uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateLanceTable(num_rows));
  auto rate_limited_read_props = read_props_;
  api::SetValue(rate_limited_read_props, "extfs.arn.iops_initial_rate", "1");
  api::SetValue(rate_limited_read_props, "extfs.arn.iops_max_rate", "1");
  ScopedEnvVarOverride burst_capacity("LANCE_AIMD_BURST_CAPACITY", "0");
  ScopedEnvVarOverride max_retries("LANCE_AIMD_MAX_RETRIES", "3");
  auto props = BaseProps();
  props.emplace_back("extfs.arn.iops_initial_rate", "1");
  props.emplace_back("extfs.arn.iops_max_rate", "1");
  ExternalTableReadScenario data(LOON_FORMAT_LANCE_TABLE);
  const auto read_start = std::chrono::steady_clock::now();
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", std::move(props)));
  ASSERT_NO_FATAL_FAILURE(data.ReadAndCheck(
      result.schema, data.files(), std::make_shared<const api::Properties>(rate_limited_read_props), num_rows));
  const auto read_elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - read_start).count();
  EXPECT_GE(read_elapsed_ms, 3000) << "Aliyun ARN reads bypassed the configured AIMD rate limit";
}

TEST_F(ExternalTableAliyunOIDCArnTest, LanceFilesystemCacheHitDoesNotReloadOidcToken) {
  const uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateLanceTable(num_rows));

  auto session_name = "lance-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  auto cache_read_props = read_props_;
  api::SetValue(cache_read_props, "extfs.arn.session_name", session_name.c_str());

  ASSERT_AND_ASSIGN(auto fs_config,
                    FilesystemCache::resolve_config(std::make_shared<const api::Properties>(cache_read_props),
                                                    result.explore_dir.c_str()));
  auto storage_options = lance::ToReaderOptions(fs_config);
  auto cache_key = storage_options.find("milvus_fs_cache_key");
  ASSERT_NE(cache_key, storage_options.end());
  ASSERT_EQ(cache_key->second, fs_config.GetCacheKey());

  auto props = BaseProps();
  props.emplace_back("extfs.arn.session_name", session_name);
  ExternalTableReadScenario data(LOON_FORMAT_LANCE_TABLE);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/lance-cache-manifest", std::move(props)));

  {
    ScopedEnvVarOverride invalid_token_file("ALIBABA_CLOUD_OIDC_TOKEN_FILE",
                                            "/path/that/does/not/exist/milvus-storage-oidc-token");
    std::vector<std::string> columns = {"id", "name", "value"};
    constexpr size_t cache_hit_iterations = 1000;
    std::shared_ptr<FormatReader> reader;
    std::cout << "[Aliyun OIDC Cache Test] Lance: starting " << cache_hit_iterations
              << " cache-hit opens with an invalid token file; no additional STS request logs are expected"
              << std::endl;
    for (size_t iteration = 0; iteration < cache_hit_iterations; ++iteration) {
      SCOPED_TRACE(::testing::Message() << "cache hit iteration " << iteration);
      ASSERT_AND_ASSIGN(
          reader, FormatReader::create(result.schema, LOON_FORMAT_LANCE_TABLE, result.cgfile,
                                       std::make_shared<const api::Properties>(cache_read_props), columns, nullptr));
    }
    std::cout << "[Aliyun OIDC Cache Test] Lance: completed " << cache_hit_iterations << " cache-hit opens"
              << std::endl;

    ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
    ValidateRowGroupInfos(rg_infos, num_rows);

    int64_t total_rows = 0;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      total_rows += batch->num_rows();
    }
    ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));
  }
}

// Iceberg variant for the same OIDC chain. This reuses BaseProps() for the
// our-side manifest bucket plus extfs.arn.* read credentials, and appends the
// snapshot id required by the Iceberg bridge.
TEST_F(ExternalTableAliyunOIDCArnTest, ReadIcebergWithOIDCChain) {
  constexpr uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateIcebergTable(num_rows));
  ExternalTableReadScenario data(LOON_FORMAT_ICEBERG_TABLE);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", BaseProps()));
  ASSERT_NO_FATAL_FAILURE(
      data.ReadAndCheck(result.schema, data.files(), std::make_shared<const api::Properties>(read_props_), num_rows));
}

TEST_F(ExternalTableAliyunOIDCArnTest, IcebergFilesystemCacheHitDoesNotReloadOidcToken) {
  const uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, CreateIcebergTable(num_rows));

  auto session_name = "iceberg-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  auto cache_read_props = read_props_;
  api::SetValue(cache_read_props, "extfs.arn.session_name", session_name.c_str());
  api::SetValue(cache_read_props, PROPERTY_READER_EXTTABLE_SNAPSHOT_ID,
                std::to_string(result.iceberg_snapshot_id).c_str());

  ASSERT_AND_ASSIGN(auto fs_config,
                    FilesystemCache::resolve_config(std::make_shared<const api::Properties>(cache_read_props),
                                                    result.explore_dir.c_str()));
  auto reader_options = iceberg::ToReaderOptions(fs_config);
  EXPECT_EQ(reader_options, (std::unordered_map<std::string, std::string>{
                                {"milvus_fs_is_local", "false"}, {"milvus_fs_bucket", fs_config.bucket_name}}));

  auto props = BaseProps();
  props.emplace_back("extfs.arn.session_name", session_name);
  ExternalTableReadScenario data(LOON_FORMAT_ICEBERG_TABLE);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/iceberg-cache-manifest-first", std::move(props)));

  {
    ScopedEnvVarOverride invalid_token_file("ALIBABA_CLOUD_OIDC_TOKEN_FILE",
                                            "/path/that/does/not/exist/milvus-storage-oidc-token");
    constexpr size_t cache_hit_iterations = 1000;
    ASSERT_AND_ASSIGN(auto* iceberg_format, Format::get(LOON_FORMAT_ICEBERG_TABLE));
    std::cout << "[Aliyun OIDC Cache Test] Iceberg: starting " << cache_hit_iterations
              << " cache-hit explores with an invalid token file; no additional STS request logs are expected"
              << std::endl;
    for (size_t iteration = 0; iteration < cache_hit_iterations; ++iteration) {
      SCOPED_TRACE(::testing::Message() << "cache hit iteration " << iteration);
      ASSERT_AND_ASSIGN(auto files, iceberg_format->explore(result.explore_dir,
                                                            std::make_shared<const api::Properties>(cache_read_props)));
      ASSERT_EQ(files.size(), data.files().size());
    }
    std::cout << "[Aliyun OIDC Cache Test] Iceberg: completed " << cache_hit_iterations << " cache-hit explores"
              << std::endl;
  }
}

// Parquet variant — the format that surfaced the original prod bug
// (PlainFormat::explore goes through the C++ aws-sdk-cpp filesystem, i.e.
// `AliyunOIDCAssumeRoleChainProvider`, as do filesystem-backed Lance and
// Iceberg planning reads. A dedicated parquet test makes it impossible for a
// future refactor to fix table formats while leaving the parquet path
// regressed.
TEST_F(ExternalTableAliyunOIDCArnTest, ReadTwoParquetFilesWithOIDCChain) {
  constexpr uint64_t num_files = 2;
  constexpr uint64_t rows_per_file = 50;
  ASSERT_AND_ASSIGN(auto result, CreateParquetFiles(num_files, rows_per_file));
  ExternalTableReadScenario data(LOON_FORMAT_PARQUET);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", BaseProps()));
  ASSERT_EQ(data.files().size(), num_files);
  ASSERT_NO_FATAL_FAILURE(data.CheckFileRows(rows_per_file));
  ASSERT_NO_FATAL_FAILURE(data.ReadAndCheck(
      result.schema, data.files(), std::make_shared<const api::Properties>(read_props_), num_files * rows_per_file));
}

// Vortex also uses the native C++ filesystem, so it must exercise the OIDC
// chain independently of the Lance and Iceberg adapters.
TEST_F(ExternalTableAliyunOIDCArnTest, ReadVortexWithOIDCChain) {
  constexpr uint64_t num_rows = 100;
  ASSERT_AND_ASSIGN(auto result, WritePlainFormatFile(LOON_FORMAT_VORTEX, num_rows, write_fs_,
                                                      std::make_shared<const api::Properties>(write_props_), test_base_,
                                                      "oss", address_, arn_bucket_));
  ExternalTableReadScenario data(LOON_FORMAT_VORTEX);
  ASSERT_NO_FATAL_FAILURE(data.Explore(result, test_base_ + "/manifest", BaseProps()));
  ASSERT_EQ(data.files().size(), 1u);
  ASSERT_NO_FATAL_FAILURE(data.CheckFileRows(num_rows));
  ASSERT_NO_FATAL_FAILURE(
      data.ReadAndCheck(result.schema, data.files(), std::make_shared<const api::Properties>(read_props_), num_rows));
}

}  // namespace milvus_storage
