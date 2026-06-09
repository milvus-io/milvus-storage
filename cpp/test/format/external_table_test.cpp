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

// Integration tests for reading external tables (Lance, Iceberg) on S3.
// These tests require:
//   STORAGE_TYPE=remote  CLOUD_PROVIDER=aws
//   ADDRESS=<s3-endpoint>  BUCKET_NAME=<bucket>  REGION=<region>

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>

#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <unordered_set>
#include <variant>

#include <arrow/api.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/lance/lance_table_writer.h"
#include "milvus-storage/format/iceberg/iceberg_common.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/transaction/transaction.h"
#include "lance_bridge.h"
#include "iceberg_bridge.h"
#include "test_env.h"

namespace milvus_storage {

// Validate row group infos: offsets are contiguous and sum to expected_logical_rows.
static void ValidateRowGroupInfos(const std::vector<RowGroupInfo>& rg_infos, uint64_t expected_logical_rows) {
  ASSERT_FALSE(rg_infos.empty());
  ASSERT_EQ(rg_infos.front().start_offset, 0u);
  for (size_t i = 1; i < rg_infos.size(); ++i) {
    ASSERT_EQ(rg_infos[i].start_offset, rg_infos[i - 1].end_offset)
        << "Row group " << i << " start_offset is not contiguous with previous end_offset";
  }
  ASSERT_EQ(rg_infos.back().end_offset, expected_logical_rows);
}

// Holds everything needed to read back a written table via FormatReader
struct WriteResult {
  api::ColumnGroupFile cgfile;
  std::shared_ptr<arrow::Schema> schema;  // nullptr for Iceberg
  uint64_t num_rows;
};

constexpr int64_t kSplitExternalTotalRows = 2'000'000;
constexpr int64_t kSplitExternalRowsPerFile = 1'000'000;
constexpr int64_t kSplitExternalWriteBatchRows = 100'000;

static std::vector<std::string> SplitExternalColumns() { return {"id", "name", "value"}; }

static std::string FormatSuffix(const std::string& format) {
  if (format == LOON_FORMAT_PARQUET) {
    return "parquet";
  }
  if (format == LOON_FORMAT_VORTEX) {
    return "vortex";
  }
  if (format == LOON_FORMAT_LANCE_TABLE) {
    return "lance";
  }
  if (format == LOON_FORMAT_ICEBERG_TABLE) {
    return "iceberg";
  }
  return format;
}

static std::string PropertyValueToString(const api::PropertyVariant& value) {
  return std::visit(
      [](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
          return v ? "true" : "false";
        } else if constexpr (std::is_same_v<T, std::nullptr_t>) {
          return "";
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          std::ostringstream oss;
          for (size_t i = 0; i < v.size(); ++i) {
            if (i != 0) {
              oss << ",";
            }
            oss << v[i];
          }
          return oss.str();
        } else if constexpr (std::is_arithmetic_v<T>) {
          return std::to_string(v);
        } else {
          return v;
        }
      },
      value);
}

static LoonFFIResult CreateLoonPropertiesFromApiProperties(const api::Properties& properties,
                                                           LoonProperties* loon_properties) {
  std::vector<std::string> keys;
  std::vector<std::string> values;
  keys.reserve(properties.size());
  values.reserve(properties.size());
  for (const auto& [key, value] : properties) {
    keys.push_back(key);
    values.push_back(PropertyValueToString(value));
  }

  std::vector<const char*> c_keys;
  std::vector<const char*> c_values;
  c_keys.reserve(keys.size());
  c_values.reserve(values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    c_keys.push_back(keys[i].c_str());
    c_values.push_back(values[i].c_str());
  }

  return loon_properties_create(c_keys.data(), c_values.data(), c_keys.size(), loon_properties);
}

struct LoonPropertiesGuard {
  LoonProperties properties{};
  ~LoonPropertiesGuard() { loon_properties_free(&properties); }
};

struct LoonTransactionGuard {
  LoonTransactionHandle handle = 0;
  ~LoonTransactionGuard() { loon_transaction_destroy(handle); }
};

struct LoonManifestGuard {
  LoonManifest* manifest = nullptr;
  ~LoonManifestGuard() { loon_manifest_destroy(manifest); }
};

class ExternalTableTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    if (!IsCloudEnv()) {
      GTEST_SKIP() << "External table tests require cloud environment (STORAGE_TYPE=remote)";
    }
    auto use_azurite = std::getenv("USE_AZURITE");
    if (use_azurite && std::string(use_azurite) == "true") {
      GTEST_SKIP() << "External table tests require real cloud storage, not Azurite";
    }
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    ASSERT_AND_ASSIGN(fs_config_, GetFileSystemConfig(properties_));

    auto address = GetEnvVar(ENV_VAR_ADDRESS).ValueOr("");
    auto bucket = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("");
    auto region = GetEnvVar(ENV_VAR_REGION).ValueOr("");
    auto cloud_provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER).ValueOr("aws");
    api::SetValue(properties_, "extfs.iam.storage_type", "remote");
    api::SetValue(properties_, "extfs.iam.cloud_provider", cloud_provider.c_str());
    api::SetValue(properties_, "extfs.iam.address", address.c_str());
    api::SetValue(properties_, "extfs.iam.bucket_name", bucket.c_str());
    api::SetValue(properties_, "extfs.iam.region", region.c_str());
    auto use_ssl_str = GetEnvVar(ENV_VAR_USE_SSL).ValueOr("true");
    api::SetValue(properties_, "extfs.iam.use_ssl", use_ssl_str.c_str());
    auto use_iam_str = GetEnvVar(ENV_VAR_USE_IAM).ValueOr("false");
    api::SetValue(properties_, "extfs.iam.use_iam", use_iam_str.c_str());
    auto ak = GetEnvVar(ENV_VAR_ACCESS_KEY_ID).ValueOr("minioadmin");
    if (use_iam_str != "true" && use_iam_str != "1") {
      auto sk = GetEnvVar(ENV_VAR_ACCESS_KEY_VALUE).ValueOr("minioadmin");
      api::SetValue(properties_, "extfs.iam.access_key_id", ak.c_str());
      api::SetValue(properties_, "extfs.iam.access_key_value", sk.c_str());
    } else {
      api::SetValue(properties_, "extfs.iam.access_key_id", ak.c_str());
    }
    FilesystemCache::getInstance().clean();

    // External table tests support S3-compatible (AWS, MinIO), Azure, and GCP storage.
    if (fs_config_.cloud_provider != kCloudProviderAWS && fs_config_.cloud_provider != kCloudProviderAzure &&
        fs_config_.cloud_provider != kCloudProviderGCP) {
      GTEST_SKIP() << "External table tests require S3, Azure, or GCP storage";
    }

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "exttable-test-" + std::to_string(ts);
  }

  void TearDown() override {
    if (fs_) {
      (void)DeleteTestDir(fs_, test_base_);
    }
    FilesystemCache::getInstance().clean();
  }

  // Create a table on S3 and return the ColumnGroupFile for reading
  arrow::Result<WriteResult> CreateTestTable(const std::string& format, uint64_t num_rows) {
    if (format == LOON_FORMAT_LANCE_TABLE) {
      return CreateLanceTable(num_rows);
    } else if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return CreateIcebergTable(num_rows);
    }
    return arrow::Status::Invalid("Unknown format: " + format);
  }

  // Create a table with some rows deleted, return ColumnGroupFile for reading
  arrow::Result<WriteResult> CreateTestTableWithDeletes(const std::string& format,
                                                        uint64_t num_rows,
                                                        const std::vector<int64_t>& deleted_ids) {
    if (format == LOON_FORMAT_LANCE_TABLE) {
      return CreateLanceTableWithDeletes(num_rows, deleted_ids);
    } else if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return CreateIcebergTableWithDeletes(num_rows, deleted_ids);
    }
    return arrow::Status::Invalid("Unknown format: " + format);
  }

  std::string MakeTableUri(const std::string& bucket, const std::string& path) {
    const auto& provider = fs_config_.cloud_provider;
    if (provider == kCloudProviderAzure) {
      // Use uniform scheme://container/path format for all providers.
      // The Rust bridge reconstructs the full container@account.dfs.endpoint
      // authority that opendal requires, using adls.account-name and
      // adls.endpoint-suffix from storage_options.
      return "abfss://" + bucket + "/" + path;
    } else if (provider == kCloudProviderGCP) {
      return "gs://" + bucket + "/" + path;
    }
    return "s3://" + bucket + "/" + path;
  }

  api::Properties properties_;
  ArrowFileSystemPtr fs_;
  ArrowFileSystemConfig fs_config_;
  std::string test_base_;

  private:
  arrow::Result<WriteResult> CreateLanceTable(uint64_t num_rows) {
    ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
    ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, 0, false, num_rows, 4, 50, {true, true, true, false}));
    auto path = test_base_ + "/lance";
    lance::LanceTableWriter writer(path, schema, properties_);
    ARROW_RETURN_NOT_OK(writer.Write(batch));
    ARROW_ASSIGN_OR_RAISE(auto cgfile, writer.Close());
    return WriteResult{std::move(cgfile), schema, num_rows};
  }

  arrow::Result<WriteResult> CreateLanceTableWithDeletes(uint64_t num_rows, const std::vector<int64_t>& deleted_ids) {
    // Write the full dataset first
    ARROW_ASSIGN_OR_RAISE(auto result, CreateLanceTable(num_rows));

    // Open the dataset and delete rows by predicate.
    // cgfile.path is Milvus format (scheme://address/bucket/key?fragment_id=N);
    // strip address before handing to Lance, which treats host as bucket.
    ARROW_ASSIGN_OR_RAISE(auto parsed, lance::ParseLanceUri(result.cgfile.path));
    auto lance_uri = lance::ToStandardLanceUri(parsed.first);

    ArrowFileSystemConfig fs_config;
    ARROW_RETURN_NOT_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
    auto storage_options = lance::ToStorageOptions(fs_config);

    auto dataset = lance::BlockingDataset::Open(lance_uri, storage_options);

    // Build predicate like "id in (3, 10, 25)"
    std::string predicate = "id in (";
    for (size_t i = 0; i < deleted_ids.size(); ++i) {
      if (i > 0)
        predicate += ", ";
      predicate += std::to_string(deleted_ids[i]);
    }
    predicate += ")";
    dataset->DeleteRows(predicate);

    return WriteResult{std::move(result.cgfile), result.schema, num_rows};
  }

  arrow::Result<WriteResult> CreateIcebergTable(uint64_t num_rows) {
    auto bucket = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("");
    if (bucket.empty()) {
      return arrow::Status::Invalid("BUCKET_NAME env var must be set");
    }
    auto path = test_base_ + "/iceberg";
    auto table_uri = MakeTableUri(bucket, path);
    auto storage_options = iceberg::ToStorageOptions(fs_config_);

    auto table_info = iceberg::CreateTestTable(table_uri, num_rows, false, {}, storage_options);
    auto file_infos = iceberg::PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
    if (file_infos.empty()) {
      return arrow::Status::Invalid("PlanFiles returned no files");
    }

    auto milvus_path = iceberg::ToMilvusUri(file_infos[0].data_file_path, fs_config_.address);
    api::ColumnGroupFile cg_file{milvus_path, 0, static_cast<int64_t>(file_infos[0].record_count), {}};
    return WriteResult{std::move(cg_file), nullptr, num_rows};
  }

  arrow::Result<WriteResult> CreateIcebergTableWithDeletes(uint64_t num_rows, const std::vector<int64_t>& deleted_ids) {
    auto bucket = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("");
    if (bucket.empty()) {
      return arrow::Status::Invalid("BUCKET_NAME env var must be set");
    }
    auto path = test_base_ + "/iceberg-deletes";
    auto table_uri = MakeTableUri(bucket, path);
    auto storage_options = iceberg::ToStorageOptions(fs_config_);

    auto table_info = iceberg::CreateTestTable(table_uri, num_rows, true, deleted_ids, storage_options);
    auto file_infos = iceberg::PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
    if (file_infos.empty()) {
      return arrow::Status::Invalid("PlanFiles returned no files");
    }

    auto milvus_path = iceberg::ToMilvusUri(file_infos[0].data_file_path, fs_config_.address);
    std::unordered_map<std::string, std::string> file_props;
    file_props[api::kPropertyMetadata] =
        iceberg::ConvertDeleteMetadataPaths(file_infos[0].delete_metadata_json, fs_config_.address);
    api::ColumnGroupFile cg_file{milvus_path, 0, static_cast<int64_t>(file_infos[0].record_count),
                                 std::move(file_props)};
    return WriteResult{std::move(cg_file), nullptr, num_rows};
  }
};

class ExternalSplitColumnGroupTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    api::Manifest::CleanCache();
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_config_, GetFileSystemConfig(properties_));
    ConfigureExternalFilesystemProperties();
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    ASSERT_AND_ASSIGN(schema_, CreateTestSchema({true, true, true, false}));

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "external-split-column-group-test-" + std::to_string(ts);
    ASSERT_STATUS_OK(DeleteTestDir(fs_, test_base_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, test_base_));
  }

  void TearDown() override {
    api::Manifest::CleanCache();
    if (fs_) {
      (void)DeleteTestDir(fs_, test_base_);
    }
    FilesystemCache::getInstance().clean();
  }

  void ConfigureExternalFilesystemProperties() {
    if (fs_config_.storage_type != "remote") {
      return;
    }

    auto set_extfs = [this](const std::string& key, const std::string& value) {
      api::SetValue(properties_, ("extfs.split." + key).c_str(), value.c_str());
    };

    set_extfs("storage_type", fs_config_.storage_type);
    set_extfs("cloud_provider", fs_config_.cloud_provider);
    set_extfs("address", fs_config_.address);
    set_extfs("bucket_name", fs_config_.bucket_name);
    set_extfs("region", fs_config_.region);
    set_extfs("root_path", fs_config_.root_path);
    set_extfs("use_ssl", fs_config_.use_ssl ? "true" : "false");
    set_extfs("use_iam", fs_config_.use_iam ? "true" : "false");
    if (!fs_config_.access_key_id.empty()) {
      set_extfs("access_key_id", fs_config_.access_key_id);
    }
    if (!fs_config_.access_key_value.empty()) {
      set_extfs("access_key_value", fs_config_.access_key_value);
    }
    if (!fs_config_.role_arn.empty()) {
      set_extfs("role_arn", fs_config_.role_arn);
    }
    if (!fs_config_.session_name.empty()) {
      set_extfs("session_name", fs_config_.session_name);
    }
    if (!fs_config_.external_id.empty()) {
      set_extfs("external_id", fs_config_.external_id);
    }
    if (fs_config_.load_frequency > 0) {
      set_extfs("load_frequency", std::to_string(fs_config_.load_frequency));
    }
    if (!fs_config_.gcp_target_service_account.empty()) {
      set_extfs("gcp_target_service_account", fs_config_.gcp_target_service_account);
    }
  }

  std::string AbsoluteLocalPath(const std::string& relative_path) const {
    return (LocalRootPath() / relative_path).lexically_normal().string();
  }

  std::filesystem::path LocalRootPath() const {
    std::filesystem::path root(fs_config_.root_path);
    if (root.is_relative()) {
      root = std::filesystem::absolute(root);
    }
    std::error_code error;
    auto canonical_root = std::filesystem::weakly_canonical(root, error);
    return error ? root.lexically_normal() : canonical_root;
  }

  std::string LocalReadablePath(const std::string& path) const {
    if (fs_config_.storage_type != "local") {
      return path;
    }

    auto root_str = LocalRootPath().string();
    std::filesystem::path local_path(path);
    std::error_code error;
    auto canonical_path = std::filesystem::weakly_canonical(local_path, error);
    auto normalized_path = (error ? local_path.lexically_normal() : canonical_path).string();
    auto prefix = root_str + "/";
    if (normalized_path.rfind(prefix, 0) == 0) {
      return normalized_path.substr(prefix.size());
    }
    return normalized_path;
  }

  arrow::Result<std::string> MakeIcebergTableUri(const std::string& relative_path) const {
    if (fs_config_.storage_type == "local") {
      return AbsoluteLocalPath(relative_path);
    }

    if (fs_config_.bucket_name.empty()) {
      return arrow::Status::Invalid("BUCKET_NAME env var must be set for remote Iceberg split test");
    }

    if (fs_config_.cloud_provider == kCloudProviderAzure) {
      return "abfss://" + fs_config_.bucket_name + "/" + relative_path;
    }
    if (fs_config_.cloud_provider == kCloudProviderGCP) {
      return "gs://" + fs_config_.bucket_name + "/" + relative_path;
    }
    return "s3://" + fs_config_.bucket_name + "/" + relative_path;
  }

  arrow::Result<std::vector<api::ColumnGroupFile>> WriteSplitColumnGroupFiles(const std::string& format) {
    if (format == LOON_FORMAT_LANCE_TABLE) {
      return WriteLanceSplitFiles();
    }

    ARROW_ASSIGN_OR_RAISE(auto file, WriteFormatFile(format));
    auto first_file = file;
    first_file.start_index = 0;
    first_file.end_index = kSplitExternalRowsPerFile;

    auto second_file = file;
    second_file.start_index = kSplitExternalRowsPerFile;
    second_file.end_index = kSplitExternalTotalRows;

    return std::vector<api::ColumnGroupFile>{std::move(first_file), std::move(second_file)};
  }

  arrow::Result<api::ColumnGroupFile> WriteFormatFile(const std::string& format) {
    if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return WriteIcebergFile();
    }

    ARROW_ASSIGN_OR_RAISE(auto* format_impl, Format::get(format));
    const auto suffix = FormatSuffix(format);
    const auto file_path = get_data_filepath(test_base_, "one_2m_rows." + suffix);
    const auto table_base_path = test_base_ + "/" + suffix;
    ARROW_ASSIGN_OR_RAISE(auto writer,
                          format_impl->create_writer(fs_, schema_, file_path, table_base_path, properties_));

    for (int64_t start = 0; start < kSplitExternalTotalRows; start += kSplitExternalWriteBatchRows) {
      const auto rows = std::min(kSplitExternalWriteBatchRows, kSplitExternalTotalRows - start);
      ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema_, start, false, rows, 4, 50, {true, true, true, false}));
      ARROW_RETURN_NOT_OK(writer->Write(batch));
      ARROW_RETURN_NOT_OK(writer->Flush());
    }

    return writer->Close();
  }

  arrow::Result<std::vector<api::ColumnGroupFile>> WriteLanceSplitFiles() {
    ARROW_ASSIGN_OR_RAISE(auto* format_impl, Format::get(LOON_FORMAT_LANCE_TABLE));
    std::vector<api::ColumnGroupFile> files;
    files.reserve(2);

    const auto table_base_path = test_base_ + "/lance";
    for (int64_t start = 0; start < kSplitExternalTotalRows; start += kSplitExternalRowsPerFile) {
      ARROW_ASSIGN_OR_RAISE(auto writer,
                            format_impl->create_writer(fs_, schema_, "" /* file_path */, table_base_path, properties_));
      ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema_, start, false, kSplitExternalRowsPerFile, 4, 50,
                                                       {true, true, true, false}));
      ARROW_RETURN_NOT_OK(writer->Write(batch));
      ARROW_ASSIGN_OR_RAISE(auto file, writer->Close());
      if (file.end_index != kSplitExternalRowsPerFile) {
        return arrow::Status::Invalid("Expected Lance fragment to contain ", kSplitExternalRowsPerFile, " rows, got ",
                                      file.end_index);
      }
      file.start_index = 0;
      file.end_index = kSplitExternalRowsPerFile;
      files.emplace_back(std::move(file));
    }

    return files;
  }

  arrow::Result<api::ColumnGroupFile> WriteIcebergFile() {
    ARROW_ASSIGN_OR_RAISE(auto table_uri, MakeIcebergTableUri(test_base_ + "/iceberg"));
    auto storage_options = iceberg::ToStorageOptions(fs_config_);

    auto table_info = iceberg::CreateTestTable(table_uri, kSplitExternalTotalRows, false, {}, storage_options);
    auto file_infos = iceberg::PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
    if (file_infos.size() != 1) {
      return arrow::Status::Invalid("Expected exactly one Iceberg data file, got ", file_infos.size());
    }

    auto path = file_infos[0].data_file_path;
    if (fs_config_.storage_type == "local") {
      path = LocalReadablePath(path);
    } else {
      path = iceberg::ToMilvusUri(path, fs_config_.address);
    }

    return api::ColumnGroupFile{std::move(path), 0, static_cast<int64_t>(file_infos[0].record_count), {}};
  }

  std::shared_ptr<api::ColumnGroup> CreateSplitColumnGroup(const std::string& format,
                                                           std::vector<api::ColumnGroupFile> files) const {
    auto column_group = std::make_shared<api::ColumnGroup>();
    column_group->columns = SplitExternalColumns();
    column_group->format = format;
    column_group->files = std::move(files);
    return column_group;
  }

  void VerifyBatch(const std::shared_ptr<arrow::RecordBatch>& batch, int64_t expected_start_id) const {
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->num_columns(), 3);
    ASSERT_GT(batch->num_rows(), 0);

    auto id_array = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    auto value_array = std::static_pointer_cast<arrow::DoubleArray>(batch->column(2));

    auto verify_row = [&](int64_t row_offset) {
      const auto expected_id = expected_start_id + row_offset;
      ASSERT_EQ(id_array->Value(row_offset), expected_id);
      ASSERT_DOUBLE_EQ(value_array->Value(row_offset), static_cast<double>(expected_id) * 1.5);
    };

    verify_row(0);
    if (batch->num_rows() > 2) {
      verify_row(batch->num_rows() / 2);
    }
    if (batch->num_rows() > 1) {
      verify_row(batch->num_rows() - 1);
    }
  }

  void VerifyChunks(const std::vector<std::shared_ptr<arrow::RecordBatch>>& chunks) const {
    int64_t expected_start_id = 0;
    for (const auto& chunk : chunks) {
      VerifyBatch(chunk, expected_start_id);
      expected_start_id += chunk->num_rows();
    }
    ASSERT_EQ(expected_start_id, kSplitExternalTotalRows);
  }

  api::Properties properties_;
  ArrowFileSystemPtr fs_;
  ArrowFileSystemConfig fs_config_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string test_base_;
};

TEST_P(ExternalSplitColumnGroupTest, TwoMillionRowsSplitIntoMillionRowFilesReadable) {
  const auto& format = GetParam();
  if (format == LOON_FORMAT_ICEBERG_TABLE && fs_config_.cloud_provider == kCloudProviderAzure) {
    GTEST_SKIP() << "Iceberg split column-group test is not supported on Azure";
  }

  ASSERT_AND_ASSIGN(auto split_files, WriteSplitColumnGroupFiles(format));
  ASSERT_EQ(split_files.size(), 2);
  for (const auto& file : split_files) {
    ASSERT_EQ(file.end_index - file.start_index, kSplitExternalRowsPerFile);
  }

  auto column_group = CreateSplitColumnGroup(format, std::move(split_files));
  auto columns = SplitExternalColumns();
  ASSERT_AND_ASSIGN(auto reader, api::ColumnGroupReader::create(schema_, column_group, columns, properties_,
                                                                nullptr /* key_retriever */));

  ASSERT_EQ(reader->total_rows(), kSplitExternalTotalRows);
  ASSERT_GT(reader->total_number_of_chunks(), 1);

  ASSERT_AND_ASSIGN(auto boundary_chunk_indices,
                    reader->get_chunk_indices(
                        {0, kSplitExternalRowsPerFile - 1, kSplitExternalRowsPerFile, kSplitExternalTotalRows - 1}));
  ASSERT_GE(boundary_chunk_indices.size(), 2);

  std::vector<int64_t> all_chunk_indices(reader->total_number_of_chunks());
  std::iota(all_chunk_indices.begin(), all_chunk_indices.end(), 0);

  std::vector<std::shared_ptr<arrow::RecordBatch>> direct_chunks;
  direct_chunks.reserve(all_chunk_indices.size());
  for (const auto chunk_index : all_chunk_indices) {
    ASSERT_AND_ASSIGN(auto chunk, reader->get_chunk(chunk_index));
    direct_chunks.emplace_back(std::move(chunk));
  }
  VerifyChunks(direct_chunks);

  ASSERT_AND_ASSIGN(auto batched_chunks, reader->get_chunks(all_chunk_indices, 1));
  VerifyChunks(batched_chunks);
}

INSTANTIATE_TEST_SUITE_P(
    ExternalSplitFormats,
    ExternalSplitColumnGroupTest,
    ::testing::Values(LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE, LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX));

TEST_P(ExternalTableTest, TransactionReadsManifestFromAbsoluteS3Uri) {
  if (GetParam() != LOON_FORMAT_LANCE_TABLE) {
    GTEST_SKIP() << "Transaction URI regression only needs one ExternalTableTest parameter";
  }
  if (fs_config_.cloud_provider != kCloudProviderAWS) {
    GTEST_SKIP() << "Transaction absolute URI regression is S3-specific";
  }

  const std::string table_base_path = test_base_ + "/path-a";
  ASSERT_STATUS_OK(DeleteTestDir(fs_, table_base_path));
  ASSERT_STATUS_OK(CreateTestDir(fs_, table_base_path));

  auto column_group = std::make_shared<api::ColumnGroup>();
  column_group->columns = SplitExternalColumns();
  column_group->format = LOON_FORMAT_PARQUET;
  column_group->files.push_back(
      {.path = get_data_filepath(table_base_path, "data.parquet"), .start_index = 0, .end_index = 10});

  {
    ASSERT_AND_ASSIGN(auto writer_txn, api::transaction::Transaction::Open(fs_, table_base_path));
    writer_txn->AppendFiles({column_group});
    ASSERT_AND_ASSIGN(auto committed_version, writer_txn->Commit());
    ASSERT_EQ(committed_version, 1);
  }

  StorageUri table_uri;
  table_uri.scheme = "s3";
  table_uri.address = fs_config_.address;
  table_uri.bucket_name = fs_config_.bucket_name;
  table_uri.key = table_base_path;
  ASSERT_AND_ASSIGN(auto absolute_table_uri, StorageUri::Make(table_uri));

  ASSERT_AND_ASSIGN(auto uri_fs, FilesystemCache::getInstance().get(properties_, absolute_table_uri));
  ASSERT_AND_ASSIGN(auto reader_txn,
                    api::transaction::Transaction::Open(uri_fs, absolute_table_uri, api::transaction::LATEST));
  ASSERT_EQ(reader_txn->GetReadVersion(), 1);
  ASSERT_AND_ASSIGN(auto manifest, reader_txn->GetManifest());
  ASSERT_EQ(manifest->columnGroups().size(), 1);
  ASSERT_EQ(manifest->columnGroups()[0]->files.size(), 1);
  ASSERT_EQ(manifest->columnGroups()[0]->files[0].path, get_data_filepath(table_base_path, "data.parquet"));

  ASSERT_AND_ASSIGN(auto relative_reader_txn,
                    api::transaction::Transaction::Open(fs_, table_base_path, api::transaction::LATEST));
  ASSERT_EQ(relative_reader_txn->GetReadVersion(), 1);
  ASSERT_AND_ASSIGN(auto relative_manifest, relative_reader_txn->GetManifest());
  ASSERT_EQ(relative_manifest->columnGroups().size(), 1);
  ASSERT_EQ(relative_manifest->columnGroups()[0]->files.size(), 1);
  ASSERT_EQ(relative_manifest->columnGroups()[0]->files[0].path, get_data_filepath(table_base_path, "data.parquet"));

  ASSERT_AND_ASSIGN(auto parsed_uri, StorageUri::Parse(absolute_table_uri));
  ASSERT_AND_ASSIGN(auto direct_manifest,
                    api::Manifest::ReadFrom(uri_fs, get_manifest_filepath(parsed_uri.ToRelativePath(), 1)));
  ASSERT_EQ(direct_manifest->columnGroups().size(), 1);

  LoonPropertiesGuard loon_props;
  auto rc = CreateLoonPropertiesFromApiProperties(properties_, &loon_props.properties);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  LoonTransactionGuard relative_ffi_txn;
  rc = loon_transaction_begin(table_base_path.c_str(), &loon_props.properties, -1, LOON_TRANSACTION_RESOLVE_FAIL, 1,
                              &relative_ffi_txn.handle);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  int64_t relative_ffi_read_version = 0;
  rc = loon_transaction_get_read_version(relative_ffi_txn.handle, &relative_ffi_read_version);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_EQ(relative_ffi_read_version, 1);

  LoonManifestGuard relative_ffi_manifest;
  rc = loon_transaction_get_manifest(relative_ffi_txn.handle, &relative_ffi_manifest.manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(relative_ffi_manifest.manifest, nullptr);
  ASSERT_EQ(relative_ffi_manifest.manifest->column_groups.num_of_column_groups, 1u);

  LoonTransactionGuard absolute_ffi_txn;
  rc = loon_transaction_begin(absolute_table_uri.c_str(), &loon_props.properties, -1, LOON_TRANSACTION_RESOLVE_FAIL, 1,
                              &absolute_ffi_txn.handle);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);

  int64_t absolute_ffi_read_version = 0;
  rc = loon_transaction_get_read_version(absolute_ffi_txn.handle, &absolute_ffi_read_version);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_EQ(absolute_ffi_read_version, 1);

  LoonManifestGuard absolute_ffi_manifest;
  rc = loon_transaction_get_manifest(absolute_ffi_txn.handle, &absolute_ffi_manifest.manifest);
  ASSERT_TRUE(loon_ffi_is_success(&rc)) << loon_ffi_get_errmsg(&rc);
  ASSERT_NE(absolute_ffi_manifest.manifest, nullptr);
  ASSERT_EQ(absolute_ffi_manifest.manifest->column_groups.num_of_column_groups, 1u);
}

// ---------------------------------------------------------------------------
// Parameterized: write table to S3, read back via FormatReader, verify data
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, WriteAndRead) {
  const auto& format = GetParam();
  const uint64_t num_rows = 100;

  ASSERT_AND_ASSIGN(auto result, CreateTestTable(format, num_rows));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  // Read all row groups and count total rows
  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, num_rows);

  int64_t total_rows = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    ASSERT_GT(batch->num_rows(), 0);
    ASSERT_EQ(batch->num_rows(), static_cast<int64_t>(rg_infos[i].end_offset - rg_infos[i].start_offset));
    total_rows += batch->num_rows();
  }
  ASSERT_EQ(total_rows, static_cast<int64_t>(num_rows));

  // Verify data content from the first chunk
  ASSERT_AND_ASSIGN(auto first_batch, reader->get_chunk(0));
  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(first_batch->column(0));
  auto value_array = std::dynamic_pointer_cast<arrow::DoubleArray>(first_batch->column(2));
  ASSERT_NE(id_array, nullptr);
  ASSERT_NE(value_array, nullptr);

  ASSERT_EQ(id_array->Value(0), 0);
  ASSERT_DOUBLE_EQ(value_array->Value(0), 0.0);
  if (first_batch->num_rows() > 1) {
    ASSERT_EQ(id_array->Value(1), 1);
    ASSERT_DOUBLE_EQ(value_array->Value(1), 1.5);
  }
}

INSTANTIATE_TEST_SUITE_P(Formats,
                         ExternalTableTest,
                         ::testing::Values(LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE));

// ---------------------------------------------------------------------------
// Parameterized: write table with deletes, read back, verify deletions
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, WriteAndReadWithDeletes) {
  const auto& format = GetParam();
  const uint64_t num_rows = 30;
  std::vector<int64_t> deleted_ids = {3, 10, 25};

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  const auto expected_rows = static_cast<int64_t>(num_rows - deleted_ids.size());

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, static_cast<uint64_t>(expected_rows));

  int64_t total_rows = 0;
  std::vector<int64_t> all_ids;

  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    ASSERT_EQ(batch->num_rows(), static_cast<int64_t>(rg_infos[i].end_offset - rg_infos[i].start_offset));
    total_rows += batch->num_rows();

    auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
    ASSERT_NE(id_array, nullptr);
    for (int64_t j = 0; j < id_array->length(); ++j) {
      all_ids.push_back(id_array->Value(j));
    }
  }

  ASSERT_EQ(total_rows, expected_rows);

  std::unordered_set<int64_t> deleted_set(deleted_ids.begin(), deleted_ids.end());
  for (auto id : all_ids) {
    ASSERT_EQ(deleted_set.count(id), 0) << "Deleted row id=" << id << " should not appear in results";
  }

  std::unordered_set<int64_t> seen_ids(all_ids.begin(), all_ids.end());
  for (int64_t i = 0; i < static_cast<int64_t>(num_rows); ++i) {
    if (deleted_set.count(i) == 0) {
      ASSERT_EQ(seen_ids.count(i), 1u) << "Non-deleted row id=" << i << " should appear in results";
    }
  }

  // --- get_chunks: read all row groups in one call ---
  {
    std::vector<int> all_rg_indices;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      all_rg_indices.push_back(static_cast<int>(i));
    }
    ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(all_rg_indices));
    int64_t chunks_total = 0;
    for (const auto& b : batches) {
      chunks_total += b->num_rows();
    }
    ASSERT_EQ(chunks_total, static_cast<int64_t>(num_rows - deleted_ids.size()));
  }

  // --- take: logical indices ---
  // Logical indices {0, 1, 2, 3}: the first 4 non-deleted rows.
  // Deleted ids are {3, 10, 25}, so logical row 3 = id 4 (skipping deleted id=3).
  {
    std::vector<int64_t> take_indices = {0, 1, 2, 3};
    ASSERT_AND_ASSIGN(auto take_result, reader->take(take_indices));
    ASSERT_EQ(take_result->num_rows(), 4);
    auto take_ids = std::static_pointer_cast<arrow::Int64Array>(take_result->column(0)->chunk(0));
    ASSERT_EQ(take_ids->Value(0), 0);
    ASSERT_EQ(take_ids->Value(1), 1);
    ASSERT_EQ(take_ids->Value(2), 2);
    ASSERT_EQ(take_ids->Value(3), 4);  // logical row 3 = physical row 4 (id=3 was deleted)
  }

  // --- read_with_range: logical range [0, 5) ---
  // Returns the first 5 non-deleted rows: ids 0,1,2,4,5
  {
    ASSERT_AND_ASSIGN(auto range_reader, reader->read_with_range(0, 5));
    ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range_reader.get()));
    ASSERT_AND_ASSIGN(auto range_batch, range_table->CombineChunksToBatch());
    ASSERT_EQ(range_batch->num_rows(), 5);
    auto range_ids = std::static_pointer_cast<arrow::Int64Array>(range_batch->column(0));
    ASSERT_EQ(range_ids->Value(0), 0);
    ASSERT_EQ(range_ids->Value(1), 1);
    ASSERT_EQ(range_ids->Value(2), 2);
    ASSERT_EQ(range_ids->Value(3), 4);
    ASSERT_EQ(range_ids->Value(4), 5);
  }
}

// ---------------------------------------------------------------------------
// Parameterized: large table with many deletes spanning multiple row groups
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, LargeTableWithDeletesAcrossRowGroups) {
  const auto& format = GetParam();
  const uint64_t num_rows = 1000000;

  // Delete rows spanning multiple row groups (default logical_chunk_rows = 8192).
  // Covers: first/last row, row group boundaries, dense clusters, scattered.
  std::vector<int64_t> deleted_ids = {
      0,                                       // first row
      1,      2,      3,      4,      5,       // dense cluster at start
      8191,   8192,                            // row group 0/1 boundary
      16383,  16384,  16385,                   // row group 1/2 boundary cluster
      50000,  50001,  50002,  50003,  50004,   // dense cluster mid-file
      100000, 200000, 300000, 400000,          // scattered across file
      500000, 600000, 700000, 800000, 900000,  // scattered continued
      999998, 999999,                          // last rows
  };

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  const auto expected_rows = static_cast<int64_t>(num_rows - deleted_ids.size());

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, static_cast<uint64_t>(expected_rows));

  // Read all chunks, collect all ids
  int64_t total_rows = 0;
  std::unordered_set<int64_t> seen_ids;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    ASSERT_EQ(batch->num_rows(), static_cast<int64_t>(rg_infos[i].end_offset - rg_infos[i].start_offset));
    total_rows += batch->num_rows();

    auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
    ASSERT_NE(id_array, nullptr);
    for (int64_t j = 0; j < id_array->length(); ++j) {
      seen_ids.insert(id_array->Value(j));
    }
  }

  ASSERT_EQ(total_rows, expected_rows);

  // Verify deleted rows absent, non-deleted rows present
  std::unordered_set<int64_t> deleted_set(deleted_ids.begin(), deleted_ids.end());
  for (auto id : deleted_ids) {
    ASSERT_EQ(seen_ids.count(id), 0u) << "Deleted id=" << id << " should not appear";
  }
  ASSERT_EQ(static_cast<int64_t>(seen_ids.size()), expected_rows);

  // get_chunks: read all row groups in one call
  {
    std::vector<int> all_rg_indices;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      all_rg_indices.push_back(static_cast<int>(i));
    }
    ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(all_rg_indices));
    int64_t chunks_total = 0;
    for (const auto& b : batches) {
      chunks_total += b->num_rows();
    }
    ASSERT_EQ(chunks_total, expected_rows);
  }

  // take: logical indices around deletion boundaries
  {
    // Logical rows 0..4 should skip deleted ids {0,1,2,3,4,5} → ids 6,7,8,9,10
    std::vector<int64_t> take_indices = {0, 1, 2, 3, 4};
    ASSERT_AND_ASSIGN(auto take_result, reader->take(take_indices));
    ASSERT_EQ(take_result->num_rows(), 5);
    auto take_ids = std::static_pointer_cast<arrow::Int64Array>(take_result->column(0)->chunk(0));
    ASSERT_EQ(take_ids->Value(0), 6);
    ASSERT_EQ(take_ids->Value(1), 7);
    ASSERT_EQ(take_ids->Value(2), 8);
    ASSERT_EQ(take_ids->Value(3), 9);
    ASSERT_EQ(take_ids->Value(4), 10);
  }

  // read_with_range: logical range [0, 5)
  {
    ASSERT_AND_ASSIGN(auto range_reader, reader->read_with_range(0, 5));
    ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range_reader.get()));
    ASSERT_AND_ASSIGN(auto range_batch, range_table->CombineChunksToBatch());
    ASSERT_EQ(range_batch->num_rows(), 5);
    auto range_ids = std::static_pointer_cast<arrow::Int64Array>(range_batch->column(0));
    ASSERT_EQ(range_ids->Value(0), 6);
    ASSERT_EQ(range_ids->Value(1), 7);
    ASSERT_EQ(range_ids->Value(2), 8);
    ASSERT_EQ(range_ids->Value(3), 9);
    ASSERT_EQ(range_ids->Value(4), 10);
  }
}

// ---------------------------------------------------------------------------
// Parameterized: delete entire first row group worth of rows (contiguous block)
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, DeleteEntireFirstRowGroup) {
  const auto& format = GetParam();
  const uint64_t num_rows = 20000;  // ~2.4 row groups at chunk_rows=8192

  // Delete rows 0-8191: the entire first logical row group
  std::vector<int64_t> deleted_ids;
  for (int64_t i = 0; i < 8192; ++i) {
    deleted_ids.push_back(i);
  }

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  const auto expected_rows = static_cast<int64_t>(num_rows - deleted_ids.size());

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, static_cast<uint64_t>(expected_rows));

  int64_t total_rows = 0;
  std::vector<int64_t> all_ids;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
    ASSERT_EQ(batch->num_rows(), static_cast<int64_t>(rg_infos[i].end_offset - rg_infos[i].start_offset));
    total_rows += batch->num_rows();

    auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
    ASSERT_NE(id_array, nullptr);
    for (int64_t j = 0; j < id_array->length(); ++j) {
      all_ids.push_back(id_array->Value(j));
    }
  }
  ASSERT_EQ(total_rows, expected_rows);

  // First surviving row should be id=8192
  ASSERT_FALSE(all_ids.empty());
  ASSERT_EQ(all_ids.front(), 8192);
  ASSERT_EQ(all_ids.back(), static_cast<int64_t>(num_rows - 1));

  // No deleted ids should appear
  for (auto id : all_ids) {
    ASSERT_GE(id, 8192) << "Deleted id=" << id << " should not appear";
  }
}

// ---------------------------------------------------------------------------
// Parameterized: delete ALL rows in the file — reader should return empty data
// Lance removes the fragment from manifest on full delete, so skip Lance.
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, DeleteAllRows) {
  const auto& format = GetParam();
  if (format == LOON_FORMAT_LANCE_TABLE) {
    GTEST_SKIP() << "Lance removes fully-deleted fragments from manifest; not reachable in production";
  }
  const uint64_t num_rows = 100;

  // Delete every row
  std::vector<int64_t> deleted_ids;
  for (int64_t i = 0; i < static_cast<int64_t>(num_rows); ++i) {
    deleted_ids.push_back(i);
  }

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  // Row group infos should sum to 0 logical rows
  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  uint64_t total_logical = 0;
  for (const auto& rg : rg_infos) {
    total_logical += rg.end_offset - rg.start_offset;
  }
  ASSERT_EQ(total_logical, 0u);

  // read_with_range [0, 0) should return empty
  ASSERT_AND_ASSIGN(auto range_reader, reader->read_with_range(0, 0));
  ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range_reader.get()));
  ASSERT_EQ(range_table->num_rows(), 0);

  // take with empty indices should return empty
  ASSERT_AND_ASSIGN(auto take_result, reader->take({}));
  ASSERT_EQ(take_result->num_rows(), 0);
}

// ---------------------------------------------------------------------------
// Parameterized: read_with_range spanning multiple row groups
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, ReadWithRangeAcrossRowGroups) {
  const auto& format = GetParam();
  const uint64_t num_rows = 1000000;

  // Deletions scattered across multiple row groups
  std::vector<int64_t> deleted_ids = {
      100,   4000,  8000,   // in row group 0
      8192,  10000, 12000,  // in row group 1
      16384, 18000,         // in row group 2
  };

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  const auto expected_total = static_cast<int64_t>(num_rows - deleted_ids.size());

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, static_cast<uint64_t>(expected_total));

  // read_with_range [0, 20000): spans ~3 row groups (8192 each), crosses deletion boundaries
  {
    ASSERT_AND_ASSIGN(auto range_reader, reader->read_with_range(0, 20000));
    ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range_reader.get()));
    ASSERT_AND_ASSIGN(auto range_batch, range_table->CombineChunksToBatch());
    ASSERT_EQ(range_batch->num_rows(), 20000);

    auto range_ids = std::static_pointer_cast<arrow::Int64Array>(range_batch->column(0));

    // Verify no deleted ids in the range
    std::unordered_set<int64_t> deleted_set(deleted_ids.begin(), deleted_ids.end());
    for (int64_t j = 0; j < range_batch->num_rows(); ++j) {
      ASSERT_EQ(deleted_set.count(range_ids->Value(j)), 0u)
          << "Deleted id=" << range_ids->Value(j) << " at logical position " << j;
    }

    // Verify ordering: ids should be monotonically increasing
    for (int64_t j = 1; j < range_batch->num_rows(); ++j) {
      ASSERT_GT(range_ids->Value(j), range_ids->Value(j - 1)) << "IDs not monotonically increasing at position " << j;
    }
  }
}

// ---------------------------------------------------------------------------
// Parameterized: take with logical indices crossing row group boundaries
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, TakeAcrossRowGroupBoundary) {
  const auto& format = GetParam();
  const uint64_t num_rows = 1000000;

  // Delete some rows near the row group boundary (logical chunk_rows=8192)
  // Physical ids 8190, 8191, 8193 will be deleted.
  // After deletion, physical id 8192 shifts to a different logical index.
  std::vector<int64_t> deleted_ids = {8190, 8191, 8193};

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  const auto expected_total = static_cast<int64_t>(num_rows - deleted_ids.size());

  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, static_cast<uint64_t>(expected_total));

  // Physical ids near boundary: ..., 8188, 8189, [8190 del], [8191 del], 8192, [8193 del], 8194, ...
  // Logical mapping:
  //   logical 8188 → physical 8188 (id=8188)
  //   logical 8189 → physical 8189 (id=8189)
  //   logical 8190 → physical 8192 (id=8192, skipping 8190 & 8191)
  //   logical 8191 → physical 8194 (id=8194, skipping 8193)
  //   logical 8192 → physical 8195 (id=8195)
  {
    std::vector<int64_t> take_indices = {8188, 8189, 8190, 8191, 8192};
    ASSERT_AND_ASSIGN(auto take_result, reader->take(take_indices));
    ASSERT_EQ(take_result->num_rows(), 5);
    auto take_ids = std::static_pointer_cast<arrow::Int64Array>(take_result->column(0)->chunk(0));
    ASSERT_EQ(take_ids->Value(0), 8188);
    ASSERT_EQ(take_ids->Value(1), 8189);
    ASSERT_EQ(take_ids->Value(2), 8192);  // skipped 8190, 8191
    ASSERT_EQ(take_ids->Value(3), 8194);  // skipped 8193
    ASSERT_EQ(take_ids->Value(4), 8195);
  }
}

// ---------------------------------------------------------------------------
// Iceberg-only: comprehensive Azure URI flow test.
//
// Tests that the full URI transformation chain works end-to-end for Azure:
//   C++ abfss://container/path
//     → Rust normalize_uri → abfss://container@account.dfs.endpoint/path (for opendal)
//     → Rust denormalize_uri → abfss://container/path (back to C++)
//     → C++ ToMilvusUri → abfss://address/container/path (for StorageUri/fs cache)
//     → C++ MilvusURIToIcebergURI → abfss://container/path (for delete matching)
//
// Exercises every reader operation to ensure URIs resolve correctly at each layer.
// The delete matching path is critical: the positional delete parquet file stores
// file_path in opendal format (container@endpoint), while data_file_uri_ is in
// Milvus format (address/container). MilvusURIToIcebergURI normalizes both for comparison.
// ---------------------------------------------------------------------------
TEST_P(ExternalTableTest, IcebergAzureUriFlow) {
  const auto& format = GetParam();
  if (format != LOON_FORMAT_ICEBERG_TABLE) {
    GTEST_SKIP() << "Azure URI flow test is Iceberg-specific";
  }
  if (fs_config_.cloud_provider != kCloudProviderAzure) {
    GTEST_SKIP() << "Azure URI flow test requires Azure cloud provider";
  }

  const uint64_t num_rows = 50;
  // Delete rows at start, middle, and end to exercise all delete matching paths.
  std::vector<int64_t> deleted_ids = {0, 1, 24, 25, 48, 49};
  const auto expected_rows = static_cast<int64_t>(num_rows - deleted_ids.size());

  ASSERT_AND_ASSIGN(auto result, CreateTestTableWithDeletes(format, num_rows, deleted_ids));

  // --- Verify URI formats ---
  // cg_file.path should be in Milvus format: abfss://address/container/...
  // (ToMilvusUri applied in CreateIcebergTableWithDeletes)
  auto& cg_path = result.cgfile.path;
  std::cout << "  cg_file.path: " << cg_path << std::endl;
  ASSERT_TRUE(cg_path.find("abfss://") == 0) << "Expected abfss:// scheme, got: " << cg_path;
  // Should NOT contain '@' — Rust denormalized before ToMilvusUri converted to Milvus format
  ASSERT_EQ(cg_path.find('@'), std::string::npos) << "Milvus-format URI should not contain @: " << cg_path;

  // Verify the URI is parseable by StorageUri and resolves correctly
  ASSERT_AND_ASSIGN(auto parsed_uri, StorageUri::Parse(cg_path));
  ASSERT_FALSE(parsed_uri.scheme.empty());
  ASSERT_FALSE(parsed_uri.address.empty()) << "Milvus format should have address";
  ASSERT_FALSE(parsed_uri.bucket_name.empty()) << "Should have container as bucket_name";

  // MilvusURIToIcebergURI should produce abfss://container/...
  auto simple_uri = iceberg::MilvusURIToIcebergURI(cg_path);
  std::cout << "  MilvusURIToIcebergURI: " << simple_uri << std::endl;
  ASSERT_TRUE(simple_uri.find("abfss://") == 0);
  ASSERT_EQ(simple_uri.find('@'), std::string::npos);
  // The simplified URI should start with abfss://container_name/
  ASSERT_EQ(simple_uri.find("abfss://" + parsed_uri.bucket_name + "/"), 0u);

  // --- Verify delete metadata paths are also in simplified format ---
  auto metadata_it = result.cgfile.properties.find(api::kPropertyMetadata);
  if (metadata_it != result.cgfile.properties.end()) {
    auto& metadata_str = metadata_it->second;
    std::cout << "  delete_metadata: " << metadata_str << std::endl;
    // Delete file paths should be in Milvus format (address/container) after
    // ConvertDeleteMetadataPaths. They should NOT contain '@'.
    ASSERT_EQ(metadata_str.find('@'), std::string::npos)
        << "Delete metadata paths should not contain @: " << metadata_str;
  }

  // --- Create reader and exercise all operations ---
  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(result.schema, format, result.cgfile, properties_, columns, nullptr));

  // 1. get_row_group_infos: verify logical row counts
  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());
  ValidateRowGroupInfos(rg_infos, static_cast<uint64_t>(expected_rows));

  // 2. get_chunk: read each row group, verify no deleted rows appear
  {
    std::unordered_set<int64_t> deleted_set(deleted_ids.begin(), deleted_ids.end());
    int64_t total_rows = 0;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(i));
      ASSERT_EQ(batch->num_rows(), static_cast<int64_t>(rg_infos[i].end_offset - rg_infos[i].start_offset));
      total_rows += batch->num_rows();
      auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
      ASSERT_NE(id_array, nullptr);
      for (int64_t j = 0; j < id_array->length(); ++j) {
        ASSERT_EQ(deleted_set.count(id_array->Value(j)), 0u)
            << "Deleted id=" << id_array->Value(j) << " should not appear";
      }
    }
    ASSERT_EQ(total_rows, expected_rows);
  }

  // 3. get_chunks: batch read all row groups
  {
    std::vector<int> all_rg_indices;
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      all_rg_indices.push_back(static_cast<int>(i));
    }
    ASSERT_AND_ASSIGN(auto batches, reader->get_chunks(all_rg_indices));
    int64_t total = 0;
    for (const auto& b : batches) total += b->num_rows();
    ASSERT_EQ(total, expected_rows);
  }

  // 4. take: logical indices skip deleted rows correctly
  //    Physical: [0 del, 1 del, 2, 3, ..., 24 del, 25 del, 26, ...]
  //    Logical 0 → physical 2 (id=2), logical 1 → physical 3 (id=3)
  {
    std::vector<int64_t> take_indices = {0, 1, 2};
    ASSERT_AND_ASSIGN(auto take_result, reader->take(take_indices));
    ASSERT_EQ(take_result->num_rows(), 3);
    auto take_ids = std::static_pointer_cast<arrow::Int64Array>(take_result->column(0)->chunk(0));
    ASSERT_EQ(take_ids->Value(0), 2);  // first surviving row
    ASSERT_EQ(take_ids->Value(1), 3);
    ASSERT_EQ(take_ids->Value(2), 4);
  }

  // 5. read_with_range: logical range [0, 5)
  {
    ASSERT_AND_ASSIGN(auto range_reader, reader->read_with_range(0, 5));
    ASSERT_AND_ASSIGN(auto range_table, arrow::Table::FromRecordBatchReader(range_reader.get()));
    ASSERT_AND_ASSIGN(auto range_batch, range_table->CombineChunksToBatch());
    ASSERT_EQ(range_batch->num_rows(), 5);
    auto range_ids = std::static_pointer_cast<arrow::Int64Array>(range_batch->column(0));
    ASSERT_EQ(range_ids->Value(0), 2);
    ASSERT_EQ(range_ids->Value(1), 3);
    ASSERT_EQ(range_ids->Value(2), 4);
    ASSERT_EQ(range_ids->Value(3), 5);
    ASSERT_EQ(range_ids->Value(4), 6);
  }

  // 6. clone_reader: cloned reader inherits delete positions and URI info
  {
    ASSERT_AND_ASSIGN(auto cloned, reader->clone_reader());
    ASSERT_AND_ASSIGN(auto cloned_rg_infos, cloned->get_row_group_infos());
    ASSERT_EQ(cloned_rg_infos.size(), rg_infos.size());
    for (size_t i = 0; i < rg_infos.size(); ++i) {
      ASSERT_EQ(cloned_rg_infos[i].start_offset, rg_infos[i].start_offset);
      ASSERT_EQ(cloned_rg_infos[i].end_offset, rg_infos[i].end_offset);
    }
    // Verify cloned reader produces same data
    std::vector<int64_t> take_indices = {0, 1};
    ASSERT_AND_ASSIGN(auto take_result, cloned->take(take_indices));
    ASSERT_EQ(take_result->num_rows(), 2);
    auto take_ids = std::static_pointer_cast<arrow::Int64Array>(take_result->column(0)->chunk(0));
    ASSERT_EQ(take_ids->Value(0), 2);
    ASSERT_EQ(take_ids->Value(1), 3);
  }
}

// ===========================================================================
// Validate that Lance/Iceberg can write to GCS via S3-compatible protocol.
//
// Lance/Iceberg Rust backends (object_store/opendal) don't support GCS HMAC
// keys natively. This test verifies that using cloud_provider=aws with HMAC
// credentials pointing to the GCS S3-compat endpoint works for both formats.
//
// Uses the same env vars as the GCP impersonation test in external_table_arn_test.cpp.
// ===========================================================================

class GcpS3CompatWriteTest : public ::testing::Test {
  protected:
  void SetUp() override {
    address_ = GetEnvVar("GCP_IMP_TEST_ENV_ADDRESS").ValueOr("");
    bucket_ = GetEnvVar("GCP_IMP_TEST_ENV_BUCKET").ValueOr("");
    ak_ = GetEnvVar("GCP_IMP_TEST_ENV_ACCESS_KEY").ValueOr("");
    sk_ = GetEnvVar("GCP_IMP_TEST_ENV_SECRET_KEY").ValueOr("");

    if (address_.empty() || bucket_.empty() || ak_.empty() || sk_.empty()) {
      GTEST_SKIP() << "Requires GCP_IMP_TEST_ENV_{ADDRESS,BUCKET,ACCESS_KEY,SECRET_KEY}";
    }

    // S3-compat: cloud_provider=aws pointing to GCS endpoint with HMAC keys
    api::SetValue(props_, PROPERTY_FS_STORAGE_TYPE, "remote");
    api::SetValue(props_, PROPERTY_FS_CLOUD_PROVIDER, "aws");
    api::SetValue(props_, PROPERTY_FS_ADDRESS, address_.c_str());
    api::SetValue(props_, PROPERTY_FS_BUCKET_NAME, bucket_.c_str());
    api::SetValue(props_, PROPERTY_FS_REGION, "auto");
    api::SetValue(props_, PROPERTY_FS_ACCESS_KEY_ID, ak_.c_str());
    api::SetValue(props_, PROPERTY_FS_ACCESS_KEY_VALUE, sk_.c_str());
    api::SetValue(props_, PROPERTY_FS_USE_SSL, "true");

    // extfs for FormatReader URI resolution (same S3-compat config)
    api::SetValue(props_, "extfs.s3gcp.storage_type", "remote");
    api::SetValue(props_, "extfs.s3gcp.cloud_provider", "aws");
    api::SetValue(props_, "extfs.s3gcp.address", address_.c_str());
    api::SetValue(props_, "extfs.s3gcp.bucket_name", bucket_.c_str());
    api::SetValue(props_, "extfs.s3gcp.region", "auto");
    api::SetValue(props_, "extfs.s3gcp.access_key_id", ak_.c_str());
    api::SetValue(props_, "extfs.s3gcp.access_key_value", sk_.c_str());
    api::SetValue(props_, "extfs.s3gcp.use_ssl", "true");

    ASSERT_AND_ASSIGN(fs_, GetFileSystem(props_));
    FilesystemCache::getInstance().clean();

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    test_base_ = "zc/gcp-s3compat-test-" + std::to_string(ts);
  }

  void TearDown() override {
    if (fs_) {
      (void)DeleteTestDir(fs_, test_base_);
    }
    FilesystemCache::getInstance().clean();
  }

  std::string address_;
  std::string bucket_;
  std::string ak_;
  std::string sk_;
  api::Properties props_;
  ArrowFileSystemPtr fs_;
  std::string test_base_;
};

// Lance: full round-trip (write via Rust object_store + read via FormatReader).
TEST_F(GcpS3CompatWriteTest, LanceWriteAndRead) {
  const uint64_t num_rows = 50;

  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema({true, true, true, false}));
  ASSERT_AND_ASSIGN(auto batch, CreateTestData(schema, 0, false, num_rows, 4, 50, {true, true, true, false}));
  auto path = test_base_ + "/lance";
  lance::LanceTableWriter writer(path, schema, props_);
  ASSERT_STATUS_OK(writer.Write(batch));
  ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
  std::cout << "[GCP S3-compat] lance-table write OK: " << cgfile.ToString() << std::endl;

  std::vector<std::string> columns = {"id", "name", "value"};
  ASSERT_AND_ASSIGN(auto reader,
                    FormatReader::create(schema, LOON_FORMAT_LANCE_TABLE, cgfile, props_, columns, nullptr));
  ASSERT_AND_ASSIGN(auto rg_infos, reader->get_row_group_infos());

  int64_t total = 0;
  for (size_t i = 0; i < rg_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto rb, reader->get_chunk(i));
    total += rb->num_rows();
  }
  ASSERT_EQ(total, static_cast<int64_t>(num_rows));
  std::cout << "[GCP S3-compat] lance-table read OK: " << total << " rows" << std::endl;
}

// Iceberg: only the opendal-backed write + PlanFiles path is exercised here.
// The FormatReader read path goes through the C++ AWS SDK S3 filesystem, which
// under cloud_provider=aws doesn't apply the GCS response-checksum workaround
// (that's keyed on cloud_provider=gcp), so we skip it intentionally.
TEST_F(GcpS3CompatWriteTest, IcebergWriteAndPlanFiles) {
  const uint64_t num_rows = 50;
  auto path = test_base_ + "/iceberg";
  auto table_uri = "s3://" + bucket_ + "/" + path;

  ArrowFileSystemConfig config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(props_, config));
  auto storage_options = iceberg::ToStorageOptions(config);

  auto table_info = iceberg::CreateTestTable(table_uri, num_rows, false, {}, storage_options);

  auto file_infos = iceberg::PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
  ASSERT_FALSE(file_infos.empty()) << "PlanFiles returned no files";
  ASSERT_EQ(file_infos[0].record_count, num_rows);
  std::cout << "[GCP S3-compat] iceberg-table write+PlanFiles OK: " << file_infos[0].data_file_path << " ("
            << file_infos[0].record_count << " rows)" << std::endl;
}

}  // namespace milvus_storage
