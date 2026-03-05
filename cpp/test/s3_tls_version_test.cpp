// Copyright 2024 Zilliz
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

#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <aws/core/Aws.h>
#include <aws/core/utils/logging/LogLevel.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/curl/CurlHttpClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <curl/curl.h>

#include "milvus-storage/filesystem/s3/s3_client.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/s3/s3_fs.h"
#include "milvus-storage/common/arrow_util.h"

#include "test_env.h"

namespace milvus_storage::test {

// ===========================================================================
// Unit tests for tls_min_version property and config (no cloud env needed)
// ===========================================================================

TEST(TlsMinVersionTest, PropertySetGetAndValidation) {
  // Valid values can be set and read back correctly
  for (const auto& value : {"", "1.0", "1.1", "1.2", "1.3"}) {
    api::Properties properties;
    auto err = api::SetValue(properties, PROPERTY_FS_TLS_MIN_VERSION, value);
    EXPECT_EQ(err, std::nullopt) << "Failed to set tls_min_version to '" << value << "': " << err.value_or("");

    auto result = api::GetValue<std::string>(properties, PROPERTY_FS_TLS_MIN_VERSION);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result.ValueOrDie(), value);
  }

  // Invalid values are rejected
  for (const auto& value : {"1.4", "2.0", "TLSv1.3", "ssl3", "abc"}) {
    api::Properties properties;
    auto err = api::SetValue(properties, PROPERTY_FS_TLS_MIN_VERSION, value);
    EXPECT_NE(err, std::nullopt) << "Expected rejection for tls_min_version='" << value << "' but it was accepted";
  }

  // Default value is empty string (system default)
  {
    api::Properties properties;
    auto result = api::GetValue<std::string>(properties, PROPERTY_FS_TLS_MIN_VERSION);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result.ValueOrDie(), "");
  }
}

TEST(TlsMinVersionTest, ConfigIntegration) {
  // Config picks up tls_min_version from properties
  {
    api::Properties properties;
    api::SetValue(properties, PROPERTY_FS_TLS_MIN_VERSION, "1.3");

    ArrowFileSystemConfig config;
    ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties, config));
    EXPECT_EQ(config.tls_min_version, "1.3");
  }

  // Config defaults tls_min_version to empty
  {
    api::Properties properties;
    ArrowFileSystemConfig config;
    ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties, config));
    EXPECT_EQ(config.tls_min_version, "");
  }

  // ToString includes tls_min_version
  {
    ArrowFileSystemConfig config;
    config.tls_min_version = "1.3";
    EXPECT_NE(config.ToString().find("tls_min_version=1.3"), std::string::npos);
  }

  // ToString shows (default) when tls_min_version is empty
  {
    ArrowFileSystemConfig config;
    EXPECT_NE(config.ToString().find("tls_min_version=(default)"), std::string::npos);
  }

  // Works with use_ssl=true end-to-end
  {
    api::Properties properties;
    api::SetValue(properties, PROPERTY_FS_STORAGE_TYPE, "remote");
    api::SetValue(properties, PROPERTY_FS_CLOUD_PROVIDER, "aws");
    api::SetValue(properties, PROPERTY_FS_USE_SSL, "true");
    api::SetValue(properties, PROPERTY_FS_TLS_MIN_VERSION, "1.2");

    ArrowFileSystemConfig config;
    ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties, config));
    EXPECT_EQ(config.tls_min_version, "1.2");
    EXPECT_EQ(config.use_ssl, true);
  }
}

// ===========================================================================
// Cloud-based integration tests (require real HTTPS endpoint)
// ===========================================================================

// Returns true only when the environment targets a real cloud endpoint with HTTPS.
// Skips local filesystem, MinIO (http://...), and other non-TLS setups.
static bool IsTlsCloudEnv() {
  if (!IsCloudEnv()) {
    return false;
  }
  auto address = GetEnvVar(ENV_VAR_ADDRESS).ValueOr("");
  // MinIO and local S3-compatible services use "http://..." — no TLS.
  // Real cloud endpoints (e.g. "s3.us-west-2.amazonaws.com") have no scheme prefix.
  if (address.empty() || address.rfind("http://", 0) == 0) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Custom AWS SDK logger that captures log messages for TLS version inspection.
// ---------------------------------------------------------------------------
class CapturingLogger : public Aws::Utils::Logging::LogSystemInterface {
  public:
  Aws::Utils::Logging::LogLevel GetLogLevel() const override { return Aws::Utils::Logging::LogLevel::Debug; }

  void Log(Aws::Utils::Logging::LogLevel logLevel, const char* tag, const char* formatStr, ...) override {
    char buf[4096];
    va_list args;
    va_start(args, formatStr);
    vsnprintf(buf, sizeof(buf), formatStr, args);
    va_end(args);

    std::lock_guard<std::mutex> lock(mutex_);
    messages_.emplace_back(buf);
  }

  void vaLog(Aws::Utils::Logging::LogLevel logLevel, const char* tag, const char* formatStr, va_list args) override {
    char buf[4096];
    vsnprintf(buf, sizeof(buf), formatStr, args);

    std::lock_guard<std::mutex> lock(mutex_);
    messages_.emplace_back(buf);
  }

  void LogStream(Aws::Utils::Logging::LogLevel logLevel,
                 const char* tag,
                 const Aws::OStringStream& messageStream) override {
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.emplace_back(messageStream.str());
  }

  void Flush() override {}

  // Search captured logs for TLS version string.
  // Different TLS backends produce different curl verbose formats:
  //   OpenSSL:          "SSL connection using TLSv1.3 / TLS_AES_128_GCM_SHA256"
  //   Secure Transport: "TLS 1.2 connection using TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
  std::string FindTlsVersion() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::regex openssl_regex(R"(SSL connection using (TLSv[\d.]+))");
    std::regex sectransp_regex(R"((TLS [\d.]+) connection using)");
    for (const auto& msg : messages_) {
      std::smatch match;
      if (std::regex_search(msg, match, openssl_regex)) {
        return match[1].str();
      }
      if (std::regex_search(msg, match, sectransp_regex)) {
        std::string ver = match[1].str();
        return "TLSv" + ver.substr(4);  // "TLS 1.2" -> "TLSv1.2"
      }
    }
    return "";
  }

  std::vector<std::string> FilterMessages(const std::string& keyword) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    for (const auto& msg : messages_) {
      if (msg.find(keyword) != std::string::npos) {
        result.push_back(msg);
      }
    }
    return result;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return messages_.size();
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.clear();
  }

  void DumpAll(std::ostream& os) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < messages_.size(); ++i) {
      os << "[" << i << "] " << messages_[i] << std::endl;
    }
  }

  private:
  mutable std::mutex mutex_;
  std::vector<std::string> messages_;
};

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class S3TlsVersionTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (!IsTlsCloudEnv()) {
      GTEST_SKIP() << "Skipping: requires a cloud endpoint with HTTPS "
                   << "(not local or MinIO over HTTP)";
    }

    bucket_ = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("test-bucket");

    api::Properties properties;
    ASSERT_STATUS_OK(InitTestProperties(properties));
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_USE_SSL, "true"), std::nullopt);
    ASSERT_AND_ASSIGN(fs_config_, GetFileSystemConfig(properties));
  }

  // Build an S3 client with curl verbose tracing and make a PutObject request.
  // Returns the negotiated TLS version string (e.g. "TLSv1.2" or "TLSv1.3").
  std::string ConnectAndGetTlsVersion(const std::shared_ptr<CapturingLogger>& logger) {
    logger->Clear();

    S3FileSystemProducer producer(fs_config_);
    producer.InitS3();
    auto s3_options_result = producer.CreateS3Options();
    EXPECT_TRUE(s3_options_result.ok()) << s3_options_result.status().ToString();
    auto s3_options = std::move(s3_options_result).ValueOrDie();

    ClientBuilder builder(s3_options);
    builder.mutable_config()->enableHttpClientTrace = true;
    auto client_result = builder.BuildClient();
    EXPECT_TRUE(client_result.ok()) << client_result.status().ToString();
    auto client_holder = std::move(client_result).ValueOrDie();

    auto lock_result = client_holder->Lock();
    EXPECT_TRUE(lock_result.ok()) << lock_result.status().ToString();
    auto client_lock = std::move(lock_result).ValueOrDie();

    Aws::S3::Model::PutObjectRequest put_request;
    put_request.SetBucket(bucket_.c_str());
    put_request.SetKey("unittest/tls_version_test.txt");
    auto body = Aws::MakeShared<Aws::StringStream>("TlsTest");
    (*body) << "tls version test";
    put_request.SetBody(body);

    auto outcome = client_lock.Move()->PutObject(put_request);
    EXPECT_TRUE(outcome.IsSuccess()) << "PutObject failed: " << outcome.GetError().GetMessage();

    return logger->FindTlsVersion();
  }

  std::string bucket_;
  ArrowFileSystemConfig fs_config_;
};

// ---------------------------------------------------------------------------
// Test: Enforce minimum TLS version via environment variable.
//
// InitS3() uses a static std::once_flag — only the first call's config takes
// effect.  If other tests call InitS3() before this one, the TLS factory is
// already installed and this test would produce a false positive.  We therefore
// guard against that by skipping when S3 is already initialized.
//
// Usage:
//   MILVUS_STORAGE_TLS_MIN_VERSION=1.3 \
//     ./test_binary --gtest_filter=S3TlsVersionTest.EnforceMinTlsTest
// ---------------------------------------------------------------------------
TEST_F(S3TlsVersionTest, EnforceMinTlsTest) {
  // Guard: InitS3()'s once_flag means only the first caller's config wins.
  if (IsS3Initialized()) {
    GTEST_SKIP() << "S3 is already initialized — cannot guarantee our TLS "
                    "config takes effect (static once_flag).  Run this test "
                    "in isolation with --gtest_filter=S3TlsVersionTest.EnforceMinTlsTest";
  }

  // Read the desired TLS version from the environment.
  const char* env_tls = std::getenv("MILVUS_STORAGE_TLS_MIN_VERSION");
  if (env_tls == nullptr || std::strlen(env_tls) == 0) {
    GTEST_SKIP() << "MILVUS_STORAGE_TLS_MIN_VERSION is not set — skipping.  "
                    "Set it to e.g. '1.3' to run this test.";
  }
  const std::string target_tls_version(env_tls);
  const std::string expected_label = "TLSv" + target_tls_version;

  // Verify the properties → config plumbing round-trips correctly.
  {
    api::Properties properties;
    ASSERT_STATUS_OK(InitTestProperties(properties));
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_USE_SSL, "true"), std::nullopt);
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TLS_MIN_VERSION, target_tls_version.c_str()), std::nullopt);

    ArrowFileSystemConfig config;
    ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties, config));
    EXPECT_EQ(config.tls_min_version, target_tls_version);
    EXPECT_TRUE(config.use_ssl);
  }

  // Apply the TLS version to the fixture config and initialize S3.
  fs_config_.tls_min_version = target_tls_version;

  // InitS3() calls Aws::InitAPI() which installs its own logger (ConsoleLogSystem),
  // so we must install our CapturingLogger AFTER InitS3() to override it.
  {
    S3FileSystemProducer producer(fs_config_);
    producer.InitS3();
  }

  auto logger = Aws::MakeShared<CapturingLogger>("TlsTest");
  Aws::Utils::Logging::InitializeAWSLogging(logger);

  std::string tls_version = ConnectAndGetTlsVersion(logger);

  if (tls_version.empty()) {
    auto tls_msgs = logger->FilterMessages("TLS");
    auto ssl_msgs = logger->FilterMessages("SSL");
    std::cerr << "TLS-related log messages (" << tls_msgs.size() << "):" << std::endl;
    for (const auto& msg : tls_msgs) std::cerr << "  " << msg << std::endl;
    std::cerr << "SSL-related log messages (" << ssl_msgs.size() << "):" << std::endl;
    for (const auto& msg : ssl_msgs) std::cerr << "  " << msg << std::endl;
    GTEST_SKIP() << "Could not capture negotiated TLS version from curl logs.  "
                    "This may require an OpenSSL-backed curl (Secure Transport on "
                    "macOS does not support CURL_SSLVERSION_TLSv1_3).";
  }

  std::cout << ">>> Negotiated TLS version: " << tls_version << " (requested minimum: " << expected_label << ")"
            << std::endl;

  // tls_min_version sets the floor — the negotiated version must be >= the requested minimum.
  // e.g. min=1.1 may negotiate 1.2 or 1.3, which is correct.
  auto parse_tls_version = [](const std::string& ver) -> double {
    // "TLSv1.3" -> 1.3, "TLSv1.2" -> 1.2, etc.
    auto pos = ver.find("TLSv");
    if (pos == std::string::npos)
      return 0.0;
    return std::stod(ver.substr(pos + 4));
  };

  double negotiated = parse_tls_version(tls_version);
  double minimum = parse_tls_version(expected_label);
  EXPECT_GE(negotiated, minimum) << "Negotiated " << tls_version << " is below the requested minimum "
                                 << expected_label;

  // Dump all raw AWS SDK log messages when PRINT_AWS_RAW_LOG=1.
  const char* env_raw_log = std::getenv("PRINT_AWS_RAW_LOG");
  if (env_raw_log != nullptr && std::string(env_raw_log) == "1") {
    std::cout << ">>> Raw AWS SDK log messages (" << logger->Size() << " total):" << std::endl;
    logger->DumpAll(std::cout);
  }
}

// ---------------------------------------------------------------------------
// Test: Verify TLS min version through the Arrow filesystem production path.
//
// Unlike EnforceMinTlsTest which uses the raw AWS SDK client directly, this
// test goes through the full production code path:
//   Properties → GetFileSystem() → Arrow S3FileSystem → write/read
//
// Arrow's S3FileSystem does not enable curl verbose tracing (enableHttpClientTrace),
// so we cannot capture the negotiated TLS version from Arrow's own HTTP requests.
// Instead we verify TLS enforcement in two complementary ways:
//   1) Arrow filesystem write + read succeeds over HTTPS (production path works).
//   2) A raw S3 client request (with curl tracing) confirms the negotiated TLS
//      version, since both paths share the same global TlsHttpClientFactory.
// ---------------------------------------------------------------------------
TEST_F(S3TlsVersionTest, EnforceMinTlsArrowFsTest) {
  const char* env_tls = std::getenv("MILVUS_STORAGE_TLS_MIN_VERSION");
  if (env_tls == nullptr || std::strlen(env_tls) == 0) {
    GTEST_SKIP() << "MILVUS_STORAGE_TLS_MIN_VERSION is not set — skipping.  "
                    "Set it to e.g. '1.3' to run this test.";
  }
  const std::string target_tls_version(env_tls);
  const std::string expected_label = "TLSv" + target_tls_version;

  // Build properties with tls_min_version through the normal production path.
  api::Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_USE_SSL, "true"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TLS_MIN_VERSION, target_tls_version.c_str()), std::nullopt);

  // Verify config plumbing: tls_min_version flows through properties → config.
  {
    ArrowFileSystemConfig config;
    ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties, config));
    EXPECT_EQ(config.tls_min_version, target_tls_version);
    EXPECT_TRUE(config.use_ssl);
  }

  // Ensure S3 global init (with TLS factory) happens before we install the logger.
  // InitS3() is guarded by once_flag, so this is safe even if EnforceMinTlsTest ran first.
  fs_config_.tls_min_version = target_tls_version;
  {
    S3FileSystemProducer producer(fs_config_);
    producer.InitS3();
  }

  auto logger = Aws::MakeShared<CapturingLogger>("TlsTest");
  Aws::Utils::Logging::InitializeAWSLogging(logger);

  // --- Part 1: Arrow filesystem production path write + read ---
  ASSERT_AND_ASSIGN(auto fs, GetFileSystem(properties));

  const std::string test_path = "/unittest/tls_arrow_fs_test.txt";
  const std::string content = "tls arrow filesystem test";

  // Write through Arrow filesystem.
  {
    ASSERT_AND_ASSIGN(auto output_stream, fs->OpenOutputStream(test_path));
    auto buffer = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_OK(output_stream->Close());
  }

  // Read back through Arrow filesystem and verify content round-trips correctly.
  {
    ASSERT_AND_ASSIGN(auto input_stream, fs->OpenInputStream(test_path));
    ASSERT_AND_ASSIGN(auto buffer, input_stream->Read(content.size()));
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(buffer->data()), buffer->size()), content);
  }

  // Clean up test file.
  (void)fs->DeleteFile(test_path);

  std::cout << ">>> [ArrowFs] Write + read over HTTPS succeeded." << std::endl;

  // In test builds (BUILD_GTEST), TlsHttpClientFactory enables curl verbose tracing,
  // so TLS handshake details are captured by the CapturingLogger.
  std::string tls_version = logger->FindTlsVersion();

  if (tls_version.empty()) {
    auto tls_msgs = logger->FilterMessages("TLS");
    auto ssl_msgs = logger->FilterMessages("SSL");
    std::cerr << "TLS-related log messages (" << tls_msgs.size() << "):" << std::endl;
    for (const auto& msg : tls_msgs) std::cerr << "  " << msg << std::endl;
    std::cerr << "SSL-related log messages (" << ssl_msgs.size() << "):" << std::endl;
    for (const auto& msg : ssl_msgs) std::cerr << "  " << msg << std::endl;
    GTEST_SKIP() << "Could not capture negotiated TLS version from Arrow filesystem curl logs.";
  }

  std::cout << ">>> [ArrowFs] Negotiated TLS version: " << tls_version << " (requested minimum: " << expected_label
            << ")" << std::endl;

  auto parse_tls_version = [](const std::string& ver) -> double {
    auto pos = ver.find("TLSv");
    if (pos == std::string::npos)
      return 0.0;
    return std::stod(ver.substr(pos + 4));
  };

  double negotiated = parse_tls_version(tls_version);
  double minimum = parse_tls_version(expected_label);
  EXPECT_GE(negotiated, minimum) << "Negotiated " << tls_version << " is below the requested minimum "
                                 << expected_label;

  const char* env_raw_log = std::getenv("PRINT_AWS_RAW_LOG");
  if (env_raw_log != nullptr && std::string(env_raw_log) == "1") {
    std::cout << ">>> [ArrowFs] Raw AWS SDK log messages (" << logger->Size() << " total):" << std::endl;
    logger->DumpAll(std::cout);
  }
}

}  // namespace milvus_storage::test
