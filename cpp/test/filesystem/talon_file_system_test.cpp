// Copyright 2026 Zilliz
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

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <exception>
#include <functional>
#include <fstream>
#include <iterator>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/memory.h>
#include <arrow/util/async_generator.h>
#include <arrow/util/io_util.h>
#include <nlohmann/json.hpp>
#include <folly/ScopeGuard.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/talon/talon_file_system_producer.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "milvus-storage/ffi_filesystem_metrics_c.h"
#include "milvus-storage/ffi_talon_c.h"
#include "talon/talon_bridge.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/writer.h"
#include "test_env.h"

namespace milvus_storage::talon::internal {
arrow::Result<int64_t> ToArrowIoResult(const char* operation,
                                       int32_t error_code,
                                       uint64_t value,
                                       const char* error_msg);
}

namespace milvus_storage::test {

class TalonCallbackDeathTest : public ::testing::TestWithParam<bool> {};

TEST_P(TalonCallbackDeathTest, ThrowingContinuationDoesNotRetryCompletion) {
  const auto original_death_test_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        ::alarm(10);
        const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener < 0)
          std::_Exit(1);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t address_size = sizeof(address);
        if (::bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
            ::listen(listener, 1) != 0 ||
            ::getsockname(listener, reinterpret_cast<sockaddr*>(&address), &address_size) != 0) {
          std::_Exit(1);
        }
        ArrowFileSystemConfig config;
        config.talon_coordinator = "127.0.0.1:" + std::to_string(ntohs(address.sin_port));
        config.talon_block_size = 8388608;
        config.talon_max_idle_per_addr = 8;
        const auto client = talon::TalonClient::Make(config);
        if (!client.ok())
          std::_Exit(1);
        const bool raw_stat = GetParam();
        uint8_t output = 0;
        // Only an exception escaping completion should reach these handlers.
        if (raw_stat) {
          (*client)->StatAsync("s3://bucket/key").AddCallback([](const arrow::Result<talon::TalonObjectStat>&) {
            std::set_terminate([] { std::_Exit(86); });
            throw std::runtime_error("Talon consumer callback failed");
          });
        } else {
          (*client)->ReadAsync("s3://bucket/key", 0, 1, &output).AddCallback([](const arrow::Result<int64_t>&) {
            std::set_terminate([] { std::_Exit(86); });
            throw 42;
          });
        }
        // Hold the RPC open until the continuation is registered, then fail it.
        const int connection = ::accept(listener, nullptr, nullptr);
        if (connection < 0)
          std::_Exit(1);
        ::close(listener);
        ::close(connection);
        // The old catch-and-retry path silently survives until the alarm fires.
        for (;;) ::pause();
      },
      ::testing::ExitedWithCode(86), "");
  GTEST_FLAG_SET(death_test_style, original_death_test_style);
}

INSTANTIATE_TEST_SUITE_P(Bridge, TalonCallbackDeathTest, ::testing::Bool());

TEST(TalonRawBridgeTest, PropertiesNeedOnlyClientConfiguration) {
  api::Properties properties;
  properties[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:1");
  properties[PROPERTY_FS_TALON_BLOCK_SIZE] = uint32_t{8388608};
  properties[PROPERTY_FS_TALON_MAX_IDLE_PER_ADDR] = uint32_t{8};
  ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
  ASSERT_TRUE(talon::TalonClient::Make(config).ok());
}

TEST(TalonRawBridgeTest, RejectsInvalidCredentialEndpointBeforeDispatch) {
  api::Properties properties;
  properties[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:1");
  properties["fs.credential_endpoint"] = std::string("file:///broker-secret");
  properties[PROPERTY_FS_CLOUD_PROVIDER] = std::string("aws");
  properties[PROPERTY_FS_ADDRESS] = std::string("s3.us-west-2.amazonaws.com");
  properties[PROPERTY_FS_REGION] = std::string("us-west-2");
  properties[PROPERTY_FS_BUCKET_NAME] = std::string("bucket");
  properties[PROPERTY_FS_ROLE_ARN] = std::string("role-secret");
  ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
  ASSERT_AND_ASSIGN(const auto client, talon::TalonClient::Make(config));
  const auto result = client->Load({"s3://bucket/empty", 0, ""});
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsInvalid());
  EXPECT_EQ(result.status().message().find("broker-secret"), std::string::npos);
  EXPECT_EQ(result.status().message().find("role-secret"), std::string::npos);
}

TEST(TalonRawBridgeTest, InvalidBrokerIdentityEncodingReturnsStatus) {
  api::Properties properties;
  properties[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:1");
  properties[PROPERTY_FS_CREDENTIAL_ENDPOINT] = std::string("https://broker/assume-role");
  properties[PROPERTY_FS_REGION] = std::string("us-west-2");
  properties[PROPERTY_FS_ROLE_ARN] = std::string("arn:aws:iam::123456789012:role/reader");
  properties[PROPERTY_FS_EXTERNAL_ID] = std::string("secret-") + char(0xff);
  EXPECT_NO_THROW({
    ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
    const auto result = talon::TalonClient::Make(config);
    EXPECT_TRUE(result.status().IsInvalid());
    EXPECT_EQ(result.status().message().find("secret-"), std::string::npos);
  });
}

TEST(TalonRawBridgeTest, RejectsUnsupportedBrokerCloudsWithoutAwsSubstitution) {
  for (const auto* provider : {"aliyun", "tencent", "huawei", "volcengine"}) {
    api::Properties properties;
    properties[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:1");
    properties[PROPERTY_FS_CREDENTIAL_ENDPOINT] = std::string("https://broker/assume-role");
    properties[PROPERTY_FS_CLOUD_PROVIDER] = std::string(provider);
    properties[PROPERTY_FS_REGION] = std::string("us-west-2");
    properties[PROPERTY_FS_ROLE_ARN] = std::string("role");
    ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
    const auto result = talon::TalonClient::Make(config);
    ASSERT_FALSE(result.ok()) << provider;
    EXPECT_TRUE(result.status().IsInvalid());
  }
}

class TalonBrokerRequestTest : public ::testing::Test {
  protected:
  void TearDown() override {
    if (connection_ >= 0)
      ::close(connection_);
    if (listener_ >= 0)
      ::close(listener_);
  }

  enum class Operation { Stat, ImplicitStat, Read, Load, BatchLoad };

  void CaptureRequest(api::Properties properties,
                      const std::string& provider,
                      const std::string& uri,
                      Operation operation,
                      nlohmann::json& descriptor,
                      const LoonTalonRequestOptions* overrides = nullptr) {
    const auto namespace_it = properties.find("fs.talon.namespace");
    std::string expected_namespace =
        namespace_it == properties.end() ? "" : std::get<std::string>(namespace_it->second);
    const auto broker_it = properties.find(PROPERTY_FS_CREDENTIAL_ENDPOINT);
    bool expect_broker = broker_it != properties.end() && !std::get<std::string>(broker_it->second).empty();
    if (overrides && overrides->namespace_name)
      expected_namespace = overrides->namespace_name;
    if (overrides && overrides->credential_broker)
      expect_broker = *overrides->credential_broker != '\0';
    listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(listener_, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ASSERT_EQ(::bind(listener_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)), 0);
    ASSERT_EQ(::listen(listener_, 1), 0);
    socklen_t size = sizeof(address);
    ASSERT_EQ(::getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &size), 0);
    properties[PROPERTY_FS_TALON_COORDINATOR] = "127.0.0.1:" + std::to_string(ntohs(address.sin_port));
    ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
    ASSERT_AND_ASSIGN(auto client, talon::TalonClient::Make(config));
    if (overrides && overrides->namespace_name) {
      ASSERT_AND_ASSIGN(auto request_client, client->WithNamespace(overrides->namespace_name));
      client = std::move(request_client);
    }
    if (overrides && overrides->credential_broker) {
      ASSERT_AND_ASSIGN(auto request_client, client->WithCredentialBroker(overrides->credential_broker));
      client = std::move(request_client);
    }
    arrow::Future<int64_t> future;
    uint8_t output = 0;
    if (operation == Operation::ImplicitStat) {
      ASSERT_AND_ASSIGN(auto reader, client->OpenObject(provider, "bucket", "key", std::nullopt));
      future = reader.ReadAtAsync(0, 1, &output);
    } else if (operation == Operation::Read) {
      future = client->ReadAsync(uri, 0, 1, &output, talon::TalonObjectStat{1, "v1"});
    } else if (operation == Operation::Load) {
      future = client->LoadAsync({uri, 1, "v1"}).Then([](const talon::TalonLoadResult& result) {
        return int64_t(result.size);
      });
    } else if (operation == Operation::BatchLoad) {
      future = client->BatchLoadAsync({{uri, 1, "v1"}})
                   .Then([](const talon::TalonBatchLoadResult& result) -> arrow::Result<int64_t> {
                     ARROW_RETURN_NOT_OK(result.status);
                     if (result.results.size() != 1)
                       return arrow::Status::Invalid("Expected one batch result");
                     return int64_t(result.results[0].size);
                   });
    } else {
      future = client->StatAsync(uri).Then([](const talon::TalonObjectStat& stat) { return int64_t(stat.size); });
    }
    client.reset();  // Submitted requests must retain both client and credentials.
    for (int exchange = 0; exchange < 4; ++exchange) {
      pollfd ready{listener_, POLLIN, 0};
      ASSERT_EQ(::poll(&ready, 1, 5000), 1);
      connection_ = ::accept(listener_, nullptr, nullptr);
      ASSERT_GE(connection_, 0);
      const timeval timeout{5, 0};
      ASSERT_EQ(::setsockopt(connection_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), 0);
      std::array<uint8_t, 16> header{};
      ASSERT_EQ(::recv(connection_, header.data(), header.size(), MSG_WAITALL), header.size());
      uint32_t payload_size;
      std::memcpy(&payload_size, header.data() + 12, sizeof(payload_size));
      payload_size = ntohl(payload_size);
      ASSERT_LT(payload_size, 32768);
      std::vector<uint8_t> payload(payload_size);
      ASSERT_EQ(::recv(connection_, payload.data(), payload.size(), MSG_WAITALL), payload.size());
      size_t metadata_end = 0;
      std::string wire_namespace;
      if (header[2] >= 2) {
        ASSERT_GE(payload.size(), 2);
        metadata_end = 2 + (uint16_t(payload[0]) << 8) + payload[1];
        ASSERT_LE(metadata_end, payload.size());
        for (size_t pos = 2; pos < metadata_end;) {
          ASSERT_LE(pos + 3, metadata_end);
          const uint8_t key = payload[pos];
          const size_t length = (uint16_t(payload[pos + 1]) << 8) + payload[pos + 2];
          pos += 3;
          ASSERT_LE(pos + length, metadata_end);
          if (key == 5)
            wire_namespace.assign(payload.begin() + pos, payload.begin() + pos + length);
          if (key == 6)
            descriptor = nlohmann::json::parse(payload.begin() + pos, payload.begin() + pos + length);
          pos += length;
        }
      }
      if (header[3] == 11) {
        // Named versioned reads carry NamespaceId in the business payload,
        // rather than duplicating it in the metadata envelope.
        ASSERT_GE(payload.size(), metadata_end + 12);
        EXPECT_EQ(payload[metadata_end], 0);  // NamespaceId::Named
        uint64_t length = 0;
        for (int i = 0; i < 8; ++i) length |= uint64_t(payload[metadata_end + 4 + i]) << (8 * i);
        ASSERT_LE(length, payload.size() - metadata_end - 12);
        wire_namespace.assign(payload.begin() + metadata_end + 12, payload.begin() + metadata_end + 12 + length);
      }
      ASSERT_GE(payload.size(), metadata_end + 6);
      const bool membership = header[3] == 0 && payload[metadata_end + 2] == 4;
      std::vector<uint8_t> body;
      if (membership) {
        EXPECT_TRUE(descriptor.empty());  // Shared discovery must not carry request options.
        EXPECT_TRUE(wire_namespace.empty());
        // Control schema 6, MembershipList: one serving worker at this listener.
        body = {6, 0, 5, 0, 0, 0};
        auto append_u64 = [&body](uint64_t value) {
          for (int i = 0; i < 8; ++i) body.push_back(static_cast<uint8_t>(value >> (8 * i)));
        };
        auto append_string = [&body, &append_u64](const std::string& value) {
          append_u64(value.size());
          body.insert(body.end(), value.begin(), value.end());
        };
        append_u64(1);      // topology token
        append_u64(1);      // state token
        append_u64(30000);  // discovery freshness
        append_u64(1);      // worker count
        append_string("worker");
        body.insert(body.end(), {0, 0, 2, 0, 0, 0});  // no zone, not retired, Serving
        append_string("instance");
        append_string("127.0.0.1:" + std::to_string(ntohs(address.sin_port)));
      } else {
        EXPECT_EQ(wire_namespace, expected_namespace == "default" ? "" : expected_namespace);
        EXPECT_EQ(!descriptor.empty(), expect_broker);
        if (expect_broker) {
          EXPECT_EQ(header[2], 4);  // Broker metadata requires the v4 envelope.
        } else if (!wire_namespace.empty() && header[3] != 11) {
          EXPECT_EQ(header[2], 3);
        } else {
          EXPECT_LE(header[2], 2);
        }
        if (operation == Operation::Read) {
          EXPECT_TRUE(header[3] == 10 || header[3] == 11);
          header[3] = 2;  // Raw GetRange response.
          body = {0x5a};
        } else if (operation == Operation::Load) {
          EXPECT_EQ(payload[metadata_end + 2], 19);  // LoadBlock
          body = {6, 0, 6, 0, 0, 0, 1, 0};           // Ack(ok=true, detail=None)
        } else if (operation == Operation::BatchLoad) {
          EXPECT_EQ(payload[metadata_end + 2], 20);  // BatchLoad
          body = {6, 0, 21, 0, 0, 0};                // BatchLoadResult with no failures
          body.resize(14, 0);
        } else {
          EXPECT_EQ(payload[metadata_end + 2], 8);  // StatObject
          // Object backend follows the schema and message discriminant.
          ASSERT_GE(payload.size(), metadata_end + 10);
          EXPECT_EQ(payload[metadata_end + 6], provider == "gcp" ? 1 : provider == "azure" ? 2 : 0);
          body = {6, 0, 9, 0, 0, 0};  // ObjectStat(size=0, version="")
          body.resize(22, 0);
        }
      }
      std::vector<uint8_t> response(header.begin(), header.end());
      const uint32_t response_size = htonl(body.size());
      std::memcpy(response.data() + 12, &response_size, sizeof(response_size));
      response.insert(response.end(), body.begin(), body.end());
      ASSERT_EQ(::send(connection_, response.data(), response.size(), 0), response.size());
      ::close(connection_);
      connection_ = -1;
      if (!membership) {
        ASSERT_TRUE(future.Wait(5.0));
        ASSERT_AND_ASSIGN(const auto result, future.result());
        EXPECT_EQ(result, operation == Operation::Stat || operation == Operation::ImplicitStat ? 0 : 1);
        if (operation == Operation::Read)
          EXPECT_EQ(output, 0x5a);
        return;
      }
    }
    FAIL() << "No object request after discovery";
  }

  int listener_ = -1;
  int connection_ = -1;
};

TEST_F(TalonBrokerRequestTest, S3DescriptorIncludesOriginAndEscapedIdentity) {
  api::Properties properties;
  properties[PROPERTY_FS_CREDENTIAL_ENDPOINT] = std::string("https://broker/assume-role");
  properties[PROPERTY_FS_ADDRESS] = std::string("https://s3.us-west-2.amazonaws.com");
  properties[PROPERTY_FS_REGION] = std::string("us-west-2");
  properties[PROPERTY_FS_BUCKET_NAME] = std::string("bucket");
  properties[PROPERTY_FS_ROLE_ARN] = std::string("arn:aws:iam::123456789012:role/reader");
  properties[PROPERTY_FS_EXTERNAL_ID] = std::string("quoted\"identity\\value");
  properties[PROPERTY_FS_USE_VIRTUAL_HOST] = true;
  nlohmann::json descriptor;
  CaptureRequest(properties, "aws", "s3://bucket/key", Operation::Stat, descriptor);
  ASSERT_FALSE(descriptor.empty());
  EXPECT_EQ(descriptor["url"], "https://broker/assume-role");
  EXPECT_EQ(descriptor["request"], (nlohmann::json{{"csp", "aws"},
                                                   {"region", "us-west-2"},
                                                   {"bucket", "bucket"},
                                                   {"durationSeconds", 900},
                                                   {"roleArn", "arn:aws:iam::123456789012:role/reader"},
                                                   {"externalId", "quoted\"identity\\value"}}));
  EXPECT_EQ(descriptor["origin"], (nlohmann::json{{"endpoint", "https://s3.us-west-2.amazonaws.com"},
                                                  {"region", "us-west-2"},
                                                  {"use_virtual_host", true},
                                                  {"request_timeout_ms", 3000}}));
}

TEST_F(TalonBrokerRequestTest, GcpImplicitStatUsesBearerBackendAndRetainsDescriptor) {
  api::Properties properties;
  properties[PROPERTY_FS_CREDENTIAL_ENDPOINT] = std::string("https://broker/assume-role");
  properties[PROPERTY_FS_CLOUD_PROVIDER] = std::string("gcp");
  properties[PROPERTY_FS_ADDRESS] = std::string("storage.googleapis.com");
  properties[PROPERTY_FS_USE_SSL] = true;
  properties[PROPERTY_FS_BUCKET_NAME] = std::string("bucket");
  properties[PROPERTY_FS_GCP_TARGET_SERVICE_ACCOUNT] = std::string("reader@example.iam.gserviceaccount.com");
  nlohmann::json descriptor;
  CaptureRequest(properties, "gcp", "gcs://bucket/key", Operation::ImplicitStat, descriptor);
  ASSERT_FALSE(descriptor.empty());
  EXPECT_EQ(descriptor["request"]["csp"], "gcp");
  EXPECT_EQ(descriptor["request"]["gcpServiceAccountEmail"], "reader@example.iam.gserviceaccount.com");
  EXPECT_EQ(descriptor["origin"]["endpoint"], "https://storage.googleapis.com");
}

class TalonBrokerOperationTest : public TalonBrokerRequestTest,
                                 public ::testing::WithParamInterface<std::tuple<int, bool, bool>> {};

TEST_P(TalonBrokerOperationTest, OptionalNamespaceAndBrokerReachEveryOperation) {
  const auto [operation, with_namespace, with_broker] = GetParam();
  api::Properties properties;
  if (with_namespace)
    properties["fs.talon.namespace"] = std::string("tenant-a");
  if (with_broker) {
    properties[PROPERTY_FS_CREDENTIAL_ENDPOINT] = std::string("https://broker/assume-role");
    properties[PROPERTY_FS_ADDRESS] = std::string("https://s3.us-west-2.amazonaws.com");
    properties[PROPERTY_FS_REGION] = std::string("us-west-2");
    properties[PROPERTY_FS_BUCKET_NAME] = std::string("bucket");
    properties[PROPERTY_FS_ROLE_ARN] = std::string("arn:aws:iam::123456789012:role/reader");
  }
  const Operation operations[] = {Operation::Stat, Operation::ImplicitStat, Operation::Read, Operation::Load,
                                  Operation::BatchLoad};
  nlohmann::json descriptor;
  CaptureRequest(properties, "aws", "s3://bucket/key", operations[operation], descriptor);
  if (with_broker) {
    ASSERT_FALSE(descriptor.empty());
    EXPECT_EQ(descriptor["url"], "https://broker/assume-role");
    EXPECT_EQ(descriptor["request"]["roleArn"], "arn:aws:iam::123456789012:role/reader");
  } else {
    EXPECT_TRUE(descriptor.empty());
  }
}

INSTANTIATE_TEST_SUITE_P(OptionalRequestOptions,
                         TalonBrokerOperationTest,
                         ::testing::Combine(::testing::Range(0, 5), ::testing::Bool(), ::testing::Bool()));

TEST(TalonRawBridgeTest, RejectsInvalidNamespaceWithoutRequiringBroker) {
  api::Properties properties;
  properties[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:1");
  for (const auto& name : {std::string("../escape"), std::string("worker_identity"), std::string(256, 'n')}) {
    properties["fs.talon.namespace"] = name;
    ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
    ASSERT_AND_ASSIGN(const auto client, talon::TalonClient::Make(config));
    EXPECT_TRUE(client->Load({"s3://bucket/empty", 0, ""}).status().IsInvalid());
  }
}

TEST(TalonRawBridgeTest, EmptyNamespaceAndBrokerDoNotRequireCloudProperties) {
  api::Properties properties;
  properties[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:1");
  properties[PROPERTY_FS_CREDENTIAL_ENDPOINT] = std::string();
  for (const auto* name : {"", "default", "tenant-a"}) {
    properties[PROPERTY_FS_TALON_NAMESPACE] = std::string(name);
    ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
    EXPECT_TRUE(talon::TalonClient::Make(config).ok());
  }
}

TEST(TalonRawBridgeTest, InvalidNamespaceEncodingReturnsStatus) {
  api::Properties properties;
  properties[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:1");
  properties[PROPERTY_FS_TALON_NAMESPACE] = std::string(1, char(0xff));
  EXPECT_NO_THROW({
    ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
    const auto result = talon::TalonClient::Make(config);
    EXPECT_TRUE(result.status().IsInvalid());
  });
}

class TalonRequestOverrideTest : public TalonBrokerRequestTest, public ::testing::WithParamInterface<int> {};

TEST_P(TalonRequestOverrideTest, OverridesInheritAndClearIndependently) {
  api::Properties properties;
  properties[PROPERTY_FS_TALON_NAMESPACE] = std::string("base-ns");
  properties[PROPERTY_FS_CREDENTIAL_ENDPOINT] = std::string("https://broker/default");
  properties[PROPERTY_FS_REGION] = std::string("us-west-2");
  properties[PROPERTY_FS_ROLE_ARN] = std::string("role");
  const char* broker =
      R"({"url":"https://broker/request","request":{"csp":"aws"},"origin":{"endpoint":"storage.example","region":"us-west-2"}})";
  const LoonTalonRequestOptions options[] = {
      {nullptr, nullptr}, {"request-ns", nullptr}, {nullptr, broker}, {"", ""}, {"default", nullptr}};
  nlohmann::json descriptor;
  CaptureRequest(properties, "aws", "s3://bucket/key", Operation::Stat, descriptor, &options[GetParam()]);
  if (GetParam() == 3) {
    EXPECT_TRUE(descriptor.empty());
  } else {
    ASSERT_FALSE(descriptor.empty());
    EXPECT_EQ(descriptor["url"], GetParam() == 2 ? "https://broker/request" : "https://broker/default");
  }
}

INSTANTIATE_TEST_SUITE_P(Independent, TalonRequestOverrideTest, ::testing::Range(0, 5));

TEST(TalonRawBridgeTest, EmptyOperationsDoNotContactCoordinator) {
  ArrowFileSystemConfig config;
  config.talon_coordinator = "127.0.0.1:1";
  config.talon_block_size = 8388608;
  config.talon_max_idle_per_addr = 8;
  ASSERT_AND_ASSIGN(const auto client, talon::TalonClient::Make(config));
  uint8_t output = 0x7f;
  ASSERT_AND_ASSIGN(const auto read, client->Read("s3://bucket/empty", 0, 1, &output, talon::TalonObjectStat{0, ""}));
  EXPECT_EQ(read, 0);
  EXPECT_EQ(output, 0x7f);
  ASSERT_AND_ASSIGN(const auto load, client->Load({"s3://bucket/empty", 0, ""}));
  EXPECT_EQ(load.size, 0);
  EXPECT_EQ(load.blocks, 0);
  ASSERT_AND_ASSIGN(const auto batch, client->BatchLoad({}));
  EXPECT_TRUE(batch.status.ok());
  EXPECT_TRUE(batch.results.empty());
  EXPECT_TRUE(batch.failed_files.empty());
}

TEST(TalonRawBridgeTest, ValidatesRawInputsBeforeNetworkIO) {
  ArrowFileSystemConfig config;
  config.talon_coordinator = "127.0.0.1:1";
  config.talon_block_size = 8388608;
  config.talon_max_idle_per_addr = 8;
  ASSERT_AND_ASSIGN(const auto client, talon::TalonClient::Make(config));
  uint8_t output = 0x7f;
  EXPECT_TRUE(client->Stat("invalid").status().IsInvalid());
  EXPECT_TRUE(client->Read("invalid", 0, 0, nullptr).status().IsInvalid());
  EXPECT_TRUE(client->Read("s3://bucket/key", 0, 1, nullptr).status().IsInvalid());
  EXPECT_TRUE(client->Read("s3://bucket/key", 0, UINT64_MAX, &output).status().IsInvalid());
  EXPECT_TRUE(client->Read("s3://bucket/key", UINT64_MAX, 1, &output).status().IsInvalid());
  EXPECT_TRUE(client->Load({"invalid", 0, ""}).status().IsInvalid());
  EXPECT_TRUE(client->Load({"s3://bucket/key", UINT64_MAX, ""}).status().IsInvalid());
  const auto batch = client->BatchLoad({{"s3://bucket/key", 0, ""}, {"invalid", 0, ""}});
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_TRUE(batch->status.IsInvalid()) << batch->status;
  EXPECT_TRUE(batch->results.empty());
  EXPECT_TRUE(batch->failed_files.empty());
  EXPECT_EQ(output, 0x7f);
}

TEST(TalonRawBridgeTest, InvalidConfigurationPreservesDiagnostic) {
  api::Properties properties;
  properties[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:1");
  properties[PROPERTY_FS_TALON_BLOCK_SIZE] = uint32_t{0};
  properties[PROPERTY_FS_TALON_MAX_IDLE_PER_ADDR] = uint32_t{8};
  ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
  const auto result = talon::TalonClient::Make(config);
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().message().find("block_size"), std::string::npos) << result.status();
}

TEST(TalonBrokerLiveTest, ReadsAndPrewarmsWithRequestCredentials) {
  const char* case_path = std::getenv("TEST_ENV_TALON_BROKER_CASE");
  if (case_path == nullptr) {
    GTEST_SKIP()
        << "Set TEST_ENV_TALON_BROKER_CASE to a JSON file with properties, uri, key, reference, and credential_broker";
  }
  std::ifstream case_file(case_path);
  ASSERT_TRUE(case_file.good());
  const auto live = nlohmann::json::parse(case_file);
  api::Properties properties;
  for (const auto& [name, value] : live.at("properties").items()) {
    ASSERT_EQ(api::SetValue(properties, name.c_str(), value.get<std::string>().c_str()), std::nullopt);
  }
  const auto uri = live.at("uri").get<std::string>();
  std::ifstream reference(live.at("reference").get<std::string>(), std::ios::binary);
  ASSERT_TRUE(reference.good());
  const std::vector<uint8_t> expected{std::istreambuf_iterator<char>(reference), std::istreambuf_iterator<char>()};
  ASSERT_FALSE(expected.empty());
  ASSERT_LE(expected.size(), 1024);
  ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
  ASSERT_AND_ASSIGN(const auto client, talon::TalonClient::Make(config));
  ASSERT_AND_ASSIGN(const auto stat, client->Stat(uri));
  ASSERT_GE(stat.size, expected.size());
  ASSERT_LE(stat.size, 1024U * 1024U);  // Bound real-cloud prewarm traffic.
  RecordProperty("object_size", std::to_string(stat.size));
  RecordProperty("compared_bytes", std::to_string(expected.size()));
  std::vector<uint8_t> output(expected.size());
  for (int i = 0; i < 2; ++i) {
    ASSERT_AND_ASSIGN(const auto count, client->Read(uri, 0, output.size(), output.data()));
    ASSERT_EQ(count, expected.size());
    EXPECT_TRUE(output == expected);
  }
  ASSERT_AND_ASSIGN(const auto provider, api::GetValue<std::string>(properties, PROPERTY_FS_CLOUD_PROVIDER));
  ASSERT_AND_ASSIGN(const auto bucket, api::GetValue<std::string>(properties, PROPERTY_FS_BUCKET_NAME));
  ASSERT_AND_ASSIGN(auto reader, client->OpenObject(provider, bucket, live.at("key").get<std::string>(), std::nullopt));
  ASSERT_AND_ASSIGN(const auto read, reader.ReadAtAsync(0, output.size(), output.data()).result());
  EXPECT_EQ(read, expected.size());
  EXPECT_TRUE(output == expected);
  const talon::TalonLoadRequest request{uri, stat.size, stat.version};
  ASSERT_AND_ASSIGN(const auto load, client->Load(request));
  EXPECT_EQ(load.size, stat.size);
  ASSERT_AND_ASSIGN(const auto batch, client->BatchLoad({request}));
  ASSERT_STATUS_OK(batch.status);
  ASSERT_EQ(batch.results.size(), 1);
  EXPECT_EQ(batch.results[0].size, stat.size);
  ASSERT_AND_ASSIGN(const auto after_prewarm, client->Read(uri, 0, output.size(), output.data(), stat));
  EXPECT_EQ(after_prewarm, expected.size());
  EXPECT_TRUE(output == expected);

  // A second namespace uses the public C request options on a client with no
  // cloud defaults. The live harness verifies a separate cold fill, then reuse.
  ASSERT_AND_ASSIGN(const auto namespace_name, api::GetValue<std::string>(properties, PROPERTY_FS_TALON_NAMESPACE));
  ASSERT_AND_ASSIGN(const auto coordinator, api::GetValue<std::string>(properties, PROPERTY_FS_TALON_COORDINATOR));
  ASSERT_AND_ASSIGN(const auto block_size, api::GetValue<uint32_t>(properties, PROPERTY_FS_TALON_BLOCK_SIZE));
  std::array<std::string, 3> ffi_keys = {PROPERTY_FS_TALON_COORDINATOR, PROPERTY_FS_TALON_BLOCK_SIZE,
                                         PROPERTY_FS_TALON_MAX_IDLE_PER_ADDR};
  std::array<std::string, 3> ffi_values = {coordinator, std::to_string(block_size), "8"};
  LoonProperty ffi_items[3];
  for (size_t i = 0; i < 3; ++i) ffi_items[i] = {ffi_keys[i].data(), ffi_values[i].data()};
  const LoonProperties ffi_properties{ffi_items, 3};
  LoonTalonClientHandle ffi_client = 0;
  const auto ffi_cleanup = folly::makeGuard([&] { loon_talon_client_destroy(ffi_client); });
  const auto ffi_ok = [](LoonFFIResult status) {
    const bool ok = status.err_code == LOON_SUCCESS;
    if (!ok)
      ADD_FAILURE() << loon_ffi_get_errmsg(&status);
    loon_ffi_free_result(&status);
    return ok;
  };
  ASSERT_TRUE(ffi_ok(loon_talon_client_create(&ffi_properties, &ffi_client)));
  const std::string ffi_namespace = namespace_name + "-ffi";
  const auto broker_json = live.at("credential_broker").get<std::string>();
  const LoonTalonRequestOptions options{ffi_namespace.c_str(), broker_json.c_str()};
  LoonTalonResult* raw = nullptr;
  ASSERT_TRUE(ffi_ok(loon_talon_stat(ffi_client, uri.c_str(), &options, &raw)));
  std::unique_ptr<LoonTalonResult, decltype(&loon_talon_result_free)> ffi_result(raw, loon_talon_result_free);
  ASSERT_NE(ffi_result, nullptr);
  EXPECT_EQ(ffi_result->data.stat.size, stat.size);
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(ffi_ok(loon_talon_read(ffi_client, uri.c_str(), 0, output.size(), stat.version.c_str(), &stat.size,
                                       output.data(), &options, &raw)));
    ffi_result.reset(raw);
    ASSERT_NE(ffi_result, nullptr);
    EXPECT_EQ(ffi_result->data.bytes_read, expected.size());
    EXPECT_TRUE(output == expected);
  }
  const LoonTalonLoadRequest ffi_request{uri.c_str(), stat.size, stat.version.c_str()};
  ASSERT_TRUE(ffi_ok(loon_talon_load(ffi_client, &ffi_request, &options, &raw)));
  ffi_result.reset(raw);
  ASSERT_NE(ffi_result, nullptr);
  EXPECT_EQ(ffi_result->data.load.entries.results[0].size, stat.size);
  ASSERT_TRUE(ffi_ok(loon_talon_batch_load(ffi_client, &ffi_request, 1, &options, &raw)));
  ffi_result.reset(raw);
  ASSERT_NE(ffi_result, nullptr);
  EXPECT_EQ(ffi_result->data.load.count, 1);
  auto denied = loon_talon_read(ffi_client, uri.c_str(), 0, output.size(), stat.version.c_str(), &stat.size,
                                output.data(), nullptr, &raw);
  ffi_result.reset(raw);
  EXPECT_NE(denied.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&denied);

  auto static_config = config;
  static_config.credential_endpoint.clear();
  ASSERT_AND_ASSIGN(const auto static_client, talon::TalonClient::Make(static_config));
  EXPECT_FALSE(static_client->Read(uri, 0, output.size(), output.data(), stat).ok());
  if (live.value("reject_external_id", false)) {
    auto wrong_config = config;
    wrong_config.external_id += "-invalid";
    ASSERT_AND_ASSIGN(const auto wrong_identity, talon::TalonClient::Make(wrong_config));
    EXPECT_FALSE(wrong_identity->Read(uri, 0, output.size(), output.data(), stat).ok());
  }
}

namespace {

class ScopedEnv final {
  public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name); old != nullptr) {
      old_value_ = old;
    }
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }

  ~ScopedEnv() {
    if (old_value_.has_value()) {
      setenv(name_.c_str(), old_value_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  private:
  const std::string name_;
  std::optional<std::string> old_value_;
};

}  // namespace

TEST_F(TalonBrokerRequestTest, AzureDescriptorExpandsAccountAndCloudSuffix) {
  ScopedEnv azurite("USE_AZURITE", nullptr);
  api::Properties properties;
  properties[PROPERTY_FS_CREDENTIAL_ENDPOINT] = std::string("https://broker/assume-role");
  properties[PROPERTY_FS_CLOUD_PROVIDER] = std::string("azure");
  properties[PROPERTY_FS_ADDRESS] = std::string("core.windows.net");
  properties[PROPERTY_FS_USE_SSL] = true;
  properties[PROPERTY_FS_REGION] = std::string("westus3");
  properties[PROPERTY_FS_BUCKET_NAME] = std::string("bucket");
  properties[PROPERTY_FS_ACCESS_KEY_ID] = std::string("account");
  properties[PROPERTY_FS_AZURE_CLIENT_ID] = std::string("client");
  properties[PROPERTY_FS_AZURE_TENANT_ID] = std::string("tenant");
  nlohmann::json descriptor;
  CaptureRequest(properties, "azure", "az://bucket/key", Operation::Stat, descriptor);
  ASSERT_FALSE(descriptor.empty());
  EXPECT_EQ(descriptor["request"], (nlohmann::json{{"csp", "azure"},
                                                   {"region", "westus3"},
                                                   {"bucket", "bucket"},
                                                   {"durationSeconds", 900},
                                                   {"azureAccountName", "account"},
                                                   {"azureClientId", "client"},
                                                   {"azureTenantId", "tenant"}}));
  EXPECT_EQ(descriptor["origin"]["endpoint"], "https://account.blob.core.windows.net");
  EXPECT_EQ(descriptor["origin"]["use_virtual_host"], true);
}

class TalonBrokerIamTest : public TalonBrokerRequestTest,
                           public ::testing::WithParamInterface<std::tuple<const char*, bool>> {};

TEST_P(TalonBrokerIamTest, UseIamSelectsBrokerIdentityAndOmitsTargetFields) {
  ScopedEnv azurite("USE_AZURITE", nullptr);
  const auto [provider, with_target_fields] = GetParam();
  api::Properties properties;
  properties[PROPERTY_FS_CLOUD_PROVIDER] = std::string(provider);
  properties[PROPERTY_FS_CREDENTIAL_ENDPOINT] = std::string("https://broker/credentials");
  properties[PROPERTY_FS_REGION] = std::string();
  properties[PROPERTY_FS_USE_IAM] = true;
  properties[PROPERTY_FS_USE_SSL] = true;
  properties[PROPERTY_FS_BUCKET_NAME] = std::string("bucket");
  std::string uri;
  if (std::string(provider) == "aws") {
    properties[PROPERTY_FS_ADDRESS] = std::string("s3.us-west-2.amazonaws.com");
    properties[PROPERTY_FS_REGION] = std::string("us-west-2");
    uri = "s3://bucket/key";
  } else if (std::string(provider) == "gcp") {
    properties[PROPERTY_FS_ADDRESS] = std::string("storage.googleapis.com");
    uri = "gcs://bucket/key";
  } else {
    properties[PROPERTY_FS_ADDRESS] = std::string("core.windows.net");
    properties[PROPERTY_FS_ACCESS_KEY_ID] = std::string("account");
    uri = "az://bucket/key";
  }
  if (with_target_fields) {
    properties[PROPERTY_FS_ROLE_ARN] = std::string("unused-role");
    properties[PROPERTY_FS_EXTERNAL_ID] = std::string("unused-external-id");
    properties[PROPERTY_FS_GCP_TARGET_SERVICE_ACCOUNT] = std::string("unused@example.com");
    properties[PROPERTY_FS_AZURE_CLIENT_ID] = std::string("unused-client");
    properties[PROPERTY_FS_AZURE_TENANT_ID] = std::string("unused-tenant");
  }
  nlohmann::json descriptor;
  CaptureRequest(properties, provider, uri, Operation::ImplicitStat, descriptor);
  ASSERT_FALSE(descriptor.empty());
  const auto& request = descriptor.at("request");
  EXPECT_EQ(request.at("credentialType"), "iam");
  EXPECT_EQ(request.at("csp"), provider);
  EXPECT_EQ(request.at("bucket"), "bucket");
  for (const auto* field : {"roleArn", "externalId", "gcpServiceAccountEmail", "azureClientId", "azureTenantId"}) {
    EXPECT_FALSE(request.contains(field));
  }
  if (std::string(provider) == "azure") {
    EXPECT_EQ(request.at("azureAccountName"), "account");
  }
}

INSTANTIATE_TEST_SUITE_P(Clouds,
                         TalonBrokerIamTest,
                         ::testing::Combine(::testing::Values("aws", "gcp", "azure"), ::testing::Bool()));

TEST(TalonRawBridgeTest, UseIamSelectsBrokerMode) {
  ArrowFileSystemConfig config;
  config.talon_coordinator = "127.0.0.1:1";
  config.address = "s3.us-west-2.amazonaws.com";
  config.bucket_name = "bucket";
  config.region = "us-west-2";
  config.credential_endpoint = "https://broker/credentials";
  EXPECT_TRUE(talon::TalonClient::Make(config).status().IsInvalid());
  config.use_iam = true;
  EXPECT_TRUE(talon::TalonClient::Make(config).ok());
  config.use_iam = false;
  config.role_arn = "arn:aws:iam::123456789012:role/reader";
  EXPECT_TRUE(talon::TalonClient::Make(config).ok());
}

TEST(TalonRawBridgeTest, AzureBrokerIamValidatesActiveRequestConfiguration) {
  ArrowFileSystemConfig config;
  config.talon_coordinator = "127.0.0.1:1";
  config.cloud_provider = kCloudProviderAzure;
  config.address = "core.windows.net";
  config.access_key_id = "account";
  config.bucket_name = "container";
  config.credential_endpoint = "https://broker/credentials";
  config.use_iam = true;
  EXPECT_TRUE(talon::TalonClient::Make(config).ok());
  auto missing_account = config;
  missing_account.access_key_id.clear();
  EXPECT_TRUE(talon::TalonClient::Make(missing_account).status().IsInvalid());
  auto missing_bucket = config;
  missing_bucket.bucket_name.clear();
  EXPECT_TRUE(talon::TalonClient::Make(missing_bucket).status().IsInvalid());
  auto invalid_timeout = config;
  invalid_timeout.request_timeout_ms = 0;
  EXPECT_TRUE(talon::TalonClient::Make(invalid_timeout).status().IsInvalid());
  config.credential_endpoint = "file:///credentials";
  ASSERT_AND_ASSIGN(const auto client, talon::TalonClient::Make(config));
  EXPECT_TRUE(client->Load({"az://container/empty", 0, ""}).status().IsInvalid());
}

TEST_F(TalonBrokerRequestTest, UseIamWithoutBrokerEndpointKeepsStaticRequests) {
  api::Properties properties;
  properties[PROPERTY_FS_USE_IAM] = true;
  nlohmann::json descriptor;
  CaptureRequest(properties, "aws", "s3://bucket/key", Operation::Stat, descriptor);
  EXPECT_TRUE(descriptor.empty());
}

namespace {

class MetadataInputFile final : public arrow::io::RandomAccessFile {
  public:
  explicit MetadataInputFile(int64_t size = 7)
      : metadata_result(std::shared_ptr<const arrow::KeyValueMetadata>(arrow::key_value_metadata(
            {"ETag", "Content-Length", "owner"}, {"\"origin-etag\"", std::to_string(size), "test-owner"}))),
        data_(arrow::Buffer::FromString(size == 7 ? "payload" : std::string(size, 'x'))) {}

  arrow::Status Close() override { return data_.Close(); }
  bool closed() const override { return data_.closed(); }
  arrow::Result<int64_t> Tell() const override { return data_.Tell(); }
  arrow::Status Seek(int64_t position) override { return data_.Seek(position); }
  arrow::Result<int64_t> GetSize() override { return data_.GetSize(); }
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override { return data_.Read(nbytes, out); }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override { return data_.Read(nbytes); }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    ++read_calls;
    ARROW_RETURN_NOT_OK(read_status);
    return data_.ReadAt(position, nbytes, out);
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    ++read_calls;
    ARROW_RETURN_NOT_OK(read_status);
    return data_.ReadAt(position, nbytes);
  }

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    ++metadata_read_calls;
    return metadata_result;
  }

  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    async_pool = io_context.pool();
    if (async_metadata_read_calls.fetch_add(1) == 0) {
      metadata_requested.MarkFinished();
    }
    return async_result.is_valid()
               ? async_result
               : arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>>::MakeFinished(metadata_result);
  }

  ~MetadataInputFile() override {
    if (on_destroy) {
      on_destroy();
    }
  }

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> metadata_result;
  int metadata_read_calls = 0;
  std::atomic<int> read_calls{0};
  arrow::Status read_status;
  std::atomic<int> async_metadata_read_calls{0};
  arrow::Future<> metadata_requested = arrow::Future<>::Make();
  std::function<void()> on_destroy;
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> async_result;
  arrow::MemoryPool* async_pool = nullptr;

  private:
  arrow::io::BufferReader data_;
};

class RejectingMemoryPool final : public arrow::ProxyMemoryPool {
  public:
  RejectingMemoryPool() : arrow::ProxyMemoryPool(arrow::default_memory_pool()) {}

  arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override {
    last_requested_size = size;
    if (size == 0) {
      return arrow::ProxyMemoryPool::Allocate(size, alignment, out);
    }
    // Stop before network I/O so tests can inspect the allocation request itself.
    return arrow::Status::OutOfMemory("Rejected test allocation");
  }

  int64_t last_requested_size = -1;
};

class AsyncOriginInputFile final : public arrow::io::RandomAccessFile, public NonBlockingRandomAccessFile {
  public:
  arrow::Status Close() override { return data_.Close(); }
  bool closed() const override { return data_.closed(); }
  arrow::Result<int64_t> Tell() const override { return data_.Tell(); }
  arrow::Status Seek(int64_t position) override { return data_.Seek(position); }

  arrow::Result<int64_t> Read(int64_t, void*) override { return arrow::Status::Invalid("Unexpected blocking Read"); }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t) override {
    return arrow::Status::Invalid("Unexpected blocking Read");
  }

  // Initialization reads the size cached with origin metadata. Data reads must
  // still use the native async API below rather than the blocking ReadAt.
  arrow::Result<int64_t> GetSize() override { return data_.GetSize(); }

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    return std::shared_ptr<const arrow::KeyValueMetadata>(
        arrow::key_value_metadata({"ETag", "Content-Length"}, {"\"origin-etag\"", "7"}));
  }

  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext&) override {
    return arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>>::MakeFinished(ReadMetadata());
  }

  arrow::Result<int64_t> ReadAt(int64_t, int64_t, void*) override {
    return arrow::Status::Invalid("Unexpected blocking ReadAt");
  }

  arrow::Future<int64_t> GetSizeAsync() override { return arrow::Future<int64_t>::MakeFinished(data_.GetSize()); }

  arrow::Future<int64_t> ReadAtAsyncInto(int64_t position, int64_t nbytes, uint8_t* out) override {
    const auto result = data_.ReadAt(position, nbytes, out);
    EXPECT_TRUE(result.ok()) << result.status();
    started.MarkFinished();
    return completion;
  }

  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& io_context,
                                                          int64_t position,
                                                          int64_t nbytes) override {
    async_pool = io_context.pool();
    auto allocated = arrow::AllocateResizableBuffer(nbytes, async_pool);
    if (!allocated.ok()) {
      return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(allocated.status());
    }
    auto buffer = std::move(allocated).ValueOrDie();
    auto future = ReadAtAsyncInto(position, nbytes, buffer->mutable_data());
    return future.Then(
        [buffer = std::move(buffer)](int64_t bytes_read) mutable -> arrow::Result<std::shared_ptr<arrow::Buffer>> {
          ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read));
          return std::shared_ptr<arrow::Buffer>(std::move(buffer));
        });
  }

  arrow::Future<> started = arrow::Future<>::Make();
  arrow::Future<int64_t> completion = arrow::Future<int64_t>::Make();
  arrow::MemoryPool* async_pool = nullptr;

  private:
  arrow::io::BufferReader data_{arrow::Buffer::FromString("payload")};
};

class RecordingFileSystem final : public arrow::fs::LocalFileSystem, public Observable {
  public:
  explicit RecordingFileSystem(std::string identity,
                               const arrow::io::IOContext& io_context = arrow::io::default_io_context())
      : arrow::fs::LocalFileSystem(io_context), identity_(std::move(identity)) {}

  std::string type_name() const override { return "recording"; }

  arrow::Result<std::unordered_map<std::string, std::shared_ptr<FilesystemMetrics>>> GetMetricsSources()
      const override {
    return std::unordered_map<std::string, std::shared_ptr<FilesystemMetrics>>{{kOriginMetricsSource, metrics}};
  }

  const std::shared_ptr<FilesystemMetrics> metrics = std::make_shared<FilesystemMetrics>();

  bool Equals(const arrow::fs::FileSystem& other) const override {
    ++equals_calls_;
    if (this == &other) {
      return true;
    }
    const auto* recording = dynamic_cast<const RecordingFileSystem*>(&other);
    return recording != nullptr && identity_ == recording->identity_;
  }

  arrow::Result<arrow::fs::FileInfoVector> GetFileInfo(const arrow::fs::FileSelector& selector) override {
    ++synchronous_listing_calls_;
    return arrow::fs::FileInfoVector{
        arrow::fs::FileInfo(selector.base_dir + "/synchronous", arrow::fs::FileType::File)};
  }

  arrow::fs::FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& selector) override {
    ++generator_calls_;
    last_generator_selector_ = selector;
    std::vector<arrow::fs::FileInfoVector> batches;
    batches.push_back({arrow::fs::FileInfo(selector.base_dir + "/first", arrow::fs::FileType::File)});
    batches.push_back({arrow::fs::FileInfo(selector.base_dir + "/second", arrow::fs::FileType::File)});
    return arrow::MakeVectorGenerator(std::move(batches));
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    const std::lock_guard<std::mutex> lock(input_mutex_);
    ++input_open_calls;
    last_input_path = path;
    ARROW_RETURN_NOT_OK(input_open_status);
    if (fallback_file != nullptr) {
      return fallback_file;
    }
    return input_file;
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    last_input_size = info.size();
    return OpenInputFile(info.path());
  }

  std::shared_ptr<MetadataInputFile> input_file = std::make_shared<MetadataInputFile>();
  std::shared_ptr<arrow::io::RandomAccessFile> fallback_file;
  arrow::Status input_open_status;
  int input_open_calls = 0;
  std::string last_input_path;
  std::atomic<int64_t> last_input_size{-1};

  int synchronous_listing_calls() const { return synchronous_listing_calls_; }
  int generator_calls() const { return generator_calls_; }
  int equals_calls() const { return equals_calls_; }
  const arrow::fs::FileSelector& last_generator_selector() const { return last_generator_selector_; }

  private:
  std::mutex input_mutex_;
  const std::string identity_;
  mutable int equals_calls_ = 0;
  int synchronous_listing_calls_ = 0;
  int generator_calls_ = 0;
  arrow::fs::FileSelector last_generator_selector_;
};

static arrow::Result<std::shared_ptr<FilesystemMetrics>> FindMetricsSource(const ArrowFileSystemPtr& fs,
                                                                           const std::string& source) {
  const auto observable = std::dynamic_pointer_cast<Observable>(fs);
  if (observable == nullptr) {
    return nullptr;
  }
  return observable->GetMetrics(source);
}

class TalonFileSystemServiceFreeTest : public ::testing::Test {
  protected:
  static ArrowFileSystemConfig Config() {
    ArrowFileSystemConfig config;
    config.storage_type = "remote";
    config.cloud_provider = kCloudProviderAWS;
    config.bucket_name = "test-bucket";
    config.talon_mode = TalonMode::Full;
    config.talon_coordinator = "127.0.0.1:7000";
    config.talon_block_size = 8388608;
    return config;
  }

  static arrow::Result<ArrowFileSystemPtr> Wrap(const ArrowFileSystemConfig& config,
                                                std::shared_ptr<RecordingFileSystem> origin) {
    ARROW_ASSIGN_OR_RAISE(auto talon_fs, TalonFileSystemProducer(config, std::move(origin)).Make());
    return std::make_shared<FileSystemProxy>(config.bucket_name, std::move(talon_fs));
  }
};

class TalonIntegrationTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (!IsTalonEnv()) {
      GTEST_SKIP() << "Talon integration environment is not configured";
    }

    ASSERT_STATUS_OK(InitTestProperties(properties_));
    FilesystemCache::getInstance().clean();
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    path_ = "talon-integration/" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + "-" +
            std::to_string(getpid());
    parquet_path_ = path_ + "-parquet";
    expected_.resize(1024 * 1024);
    for (size_t i = 0; i < expected_.size(); ++i) {
      expected_[i] = static_cast<uint8_t>(i % 251);
    }

    ASSERT_AND_ASSIGN(auto output, fs_->OpenOutputStream(path_));
    ASSERT_STATUS_OK(output->Write(expected_.data(), static_cast<int64_t>(expected_.size())));
    ASSERT_STATUS_OK(output->Close());
  }

  void TearDown() override {
    if (fs_ != nullptr) {
      // Let request traces distinguish read-path HEADs from cleanup's existence checks.
      RecordProperty("read_phase_end_unix_ns", std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                  std::chrono::system_clock::now().time_since_epoch())
                                                                  .count()));
      (void)fs_->DeleteFile(path_);
      (void)fs_->DeleteDir(parquet_path_);
      FilesystemCache::getInstance().clean();
    }
  }

  api::Properties properties_;
  ArrowFileSystemPtr fs_;
  std::string path_;
  std::string parquet_path_;
  std::vector<uint8_t> expected_;
};

class TalonCloudMetadataTest : public ::testing::TestWithParam<bool> {
  protected:
  void SetUp() override {
    if (!IsCloudEnv()) {
      GTEST_SKIP() << "Cloud metadata validation requires a cloud environment";
    }
    api::Properties properties;
    ASSERT_STATUS_OK(InitTestProperties(properties));
    ASSERT_AND_ASSIGN(config_, GetFileSystemConfig(properties));
    if (GetParam() && (config_.cloud_provider == kCloudProviderAzure || config_.cloud_provider == kCloudProviderGCP)) {
      GTEST_SKIP() << "This provider does not use the S3 CRT reader";
    }
    config_.talon_mode = TalonMode::Disabled;
    config_.s3_crt_async_read = GetParam();
    ASSERT_AND_ASSIGN(origin_fs_, CreateArrowFileSystem(config_));
    path_ = "talon-etag-validation-" + std::to_string(getpid()) + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    RecordProperty("provider", config_.cloud_provider);
    RecordProperty("reader", GetParam() ? "crt" : "default");
    RecordProperty("object_path", path_);
  }

  void TearDown() override {
    if (origin_fs_ != nullptr && !path_.empty()) {
      (void)origin_fs_->DeleteFile(path_);
    }
  }

  ArrowFileSystemConfig config_;
  ArrowFileSystemPtr origin_fs_;
  std::string path_;
};

}  // namespace

TEST(TalonConfigTest, DefaultsAreDisabled) {
  api::Properties properties;
  ArrowFileSystemConfig config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties, config));
  EXPECT_EQ(config.talon_mode, TalonMode::Disabled);
  EXPECT_EQ(config.talon_small_read_threshold, 524288U);
  EXPECT_TRUE(config.talon_coordinator.empty());
  EXPECT_EQ(config.talon_block_size, 256U * 1024U * 1024U);
  EXPECT_EQ(config.talon_max_idle_per_addr, 256U);
}

TEST(TalonConfigTest, RoutingPropertiesValidateInput) {
  api::Properties properties;
  for (const char* mode : {"0", "1", "2"}) {
    EXPECT_EQ(api::SetValue(properties, "fs.talon.mode", mode), std::nullopt);
  }
  for (const char* mode : {"", "-1", "3", "256", "4294967296", "disabled", "full", "small_reads"}) {
    EXPECT_NE(api::SetValue(properties, "fs.talon.mode", mode), std::nullopt);
  }
  for (const char* threshold : {"1", "1048576"}) {
    EXPECT_EQ(api::SetValue(properties, "fs.talon.small_read_threshold", threshold), std::nullopt);
  }
  for (const char* threshold : {"0", "-1", "1048577", "1MiB"}) {
    EXPECT_NE(api::SetValue(properties, "fs.talon.small_read_threshold", threshold), std::nullopt);
  }
}

TEST(TalonConfigTest, TypedRoutingPropertiesAreValidated) {
  api::Properties properties;
  properties[PROPERTY_FS_STORAGE_TYPE] = std::string("remote");
  properties[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:7000");
  properties[PROPERTY_FS_TALON_MODE] = std::string("invalid");
  EXPECT_TRUE(GetFileSystemConfig(properties).status().IsInvalid());
  for (const uint32_t mode : {3U, 255U, 256U, 258U, UINT32_MAX}) {
    properties[PROPERTY_FS_TALON_MODE] = mode;
    EXPECT_TRUE(GetFileSystemConfig(properties).status().IsInvalid());
  }
  properties[PROPERTY_FS_TALON_MODE] = static_cast<uint32_t>(TalonMode::SmallReads);
  for (const uint32_t threshold : {0U, 1048577U}) {
    properties[PROPERTY_FS_TALON_SMALL_READ_THRESHOLD] = threshold;
    EXPECT_TRUE(GetFileSystemConfig(properties).status().IsInvalid());
  }
  for (const uint32_t threshold : {1U, 1048576U}) {
    properties[PROPERTY_FS_TALON_SMALL_READ_THRESHOLD] = threshold;
    ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
    EXPECT_EQ(config.talon_mode, TalonMode::SmallReads);
    EXPECT_EQ(config.talon_small_read_threshold, threshold);
  }
}

TEST_F(TalonFileSystemServiceFreeTest, SmallReadsBypassTalonAboveThreshold) {
  api::Properties properties;
  properties[PROPERTY_FS_STORAGE_TYPE] = std::string("remote");
  properties[PROPERTY_FS_CLOUD_PROVIDER] = std::string(kCloudProviderAWS);
  properties[PROPERTY_FS_BUCKET_NAME] = std::string("test-bucket");
  properties[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:0");
  properties[PROPERTY_FS_TALON_MODE] = static_cast<uint32_t>(TalonMode::SmallReads);
  properties["fs.talon.small_read_threshold"] = uint32_t{3};
  ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties));
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  // Large reads must work without the ETag required to initialize a Talon reader.
  origin->input_file->metadata_result = arrow::Status::IOError("metadata unavailable");
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  uint8_t out[4] = {};
  ASSERT_AND_ASSIGN(const auto bytes_read, file->ReadAt(1, 4, out));
  EXPECT_EQ(bytes_read, 4);
  EXPECT_EQ(std::string(reinterpret_cast<char*>(out), 4), "aylo");
  ASSERT_AND_ASSIGN(auto buffer, file->ReadAt(1, 4));
  EXPECT_EQ(buffer->ToString(), "aylo");
  ASSERT_AND_ASSIGN(auto async_buffer, file->ReadAsync(1, 4).result());
  EXPECT_EQ(async_buffer->ToString(), "aylo");
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);
  ASSERT_AND_ASSIGN(const auto async_bytes, async_file->ReadAtAsyncInto(1, 4, out).result());
  EXPECT_EQ(async_bytes, 4);
  ASSERT_STATUS_OK(file->Seek(1));
  ASSERT_AND_ASSIGN(auto sequential_buffer, file->Read(4));
  EXPECT_EQ(sequential_buffer->ToString(), "aylo");
  ASSERT_STATUS_OK(file->Seek(1));
  ASSERT_AND_ASSIGN(const auto sequential_bytes, file->Read(4, out));
  EXPECT_EQ(sequential_bytes, 4);
  EXPECT_EQ(origin->input_file->async_metadata_read_calls.load(), 0);
}

TEST_F(TalonFileSystemServiceFreeTest, ReadRoutingIncludesThreshold) {
  for (const auto mode : {TalonMode::Full, TalonMode::SmallReads}) {
    for (const int64_t nbytes : {2, 3, 4}) {
      SCOPED_TRACE(static_cast<int>(mode));
      SCOPED_TRACE(nbytes);
      auto config = Config();
      config.talon_mode = mode;
      config.talon_small_read_threshold = 3;
      auto origin = std::make_shared<RecordingFileSystem>("origin");
      const auto error = arrow::Status::IOError("Talon requires origin metadata");
      origin->input_file->metadata_result = error;
      ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
      ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
      auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
      ASSERT_NE(async_file, nullptr);
      uint8_t out[4] = {};
      for (const auto& status : {file->ReadAt(0, nbytes, out).status(), file->ReadAt(0, nbytes).status(),
                                 file->ReadAsync(0, nbytes).result().status(),
                                 async_file->ReadAtAsyncInto(0, nbytes, out).result().status()}) {
        if (config.talon_mode == TalonMode::SmallReads && nbytes == 4) {
          EXPECT_TRUE(status.ok()) << status;
        } else {
          EXPECT_EQ(status, error);
        }
      }
    }
  }
}

TEST_F(TalonFileSystemServiceFreeTest, MetricsOnlyCountReadsRoutedThroughTalon) {
  auto config = Config();
  config.talon_mode = TalonMode::SmallReads;
  config.talon_small_read_threshold = 3;
  config.talon_coordinator = "127.0.0.1:0";
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, std::make_shared<RecordingFileSystem>("origin")));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  ASSERT_AND_ASSIGN(const auto metrics, FindMetricsSource(fs, kTalonMetricsSource));
  ASSERT_NE(metrics, nullptr);
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);
  uint8_t out[4] = {};
  int64_t expected_count = 0;
  for (const int64_t nbytes : {4, 3, 4}) {
    ASSERT_STATUS_OK(file->ReadAt(0, nbytes, out).status());
    ASSERT_STATUS_OK(file->ReadAt(0, nbytes).status());
    ASSERT_STATUS_OK(file->ReadAsync(0, nbytes).result().status());
    ASSERT_STATUS_OK(async_file->ReadAtAsyncInto(0, nbytes, out).result().status());
    if (nbytes == 3) {
      expected_count += 4;
    }
    EXPECT_EQ(metrics->GetReadCount(), expected_count);
  }
  // Only the four threshold-sized requests attempted Talon and fell back.
  EXPECT_EQ(metrics->GetReadCount(), 4);
  EXPECT_EQ(metrics->GetFailedCount(), 4);
  EXPECT_EQ(metrics->GetReadBytes(), 0);
}

TEST_F(TalonFileSystemServiceFreeTest, LargeReadRoutingPrecedesEofClampingAndPreservesOriginErrors) {
  auto config = Config();
  config.talon_mode = TalonMode::SmallReads;
  config.talon_small_read_threshold = 3;
  for (const bool warm_metadata : {false, true}) {
    SCOPED_TRACE(warm_metadata);
    auto origin = std::make_shared<RecordingFileSystem>("origin");
    const auto error = MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "origin denied", "provider detail");
    origin->input_file->read_status = error;
    ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
    if (warm_metadata) {
      ASSERT_AND_ASSIGN(const auto size, file->GetSize());
      ASSERT_EQ(size, 7);
    }
    auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
    ASSERT_NE(async_file, nullptr);
    uint8_t out[4] = {};
    // Four requested bytes exceed the threshold even though only one remains.
    for (const auto& status :
         {file->ReadAt(6, 4, out).status(), file->ReadAt(6, 4).status(), file->ReadAsync(6, 4).result().status(),
          async_file->ReadAtAsyncInto(6, 4, out).result().status()}) {
      EXPECT_EQ(status, error);
      EXPECT_EQ(status.detail(), error.detail());
    }
  }
}

TEST_F(TalonFileSystemServiceFreeTest, LargeReadsPreserveNativeAsyncOriginAndCallerPool) {
  for (const bool allocating : {false, true}) {
    SCOPED_TRACE(allocating);
    auto config = Config();
    config.talon_mode = TalonMode::SmallReads;
    config.talon_small_read_threshold = 2;
    auto origin = std::make_shared<RecordingFileSystem>("origin");
    auto origin_file = std::make_shared<AsyncOriginInputFile>();
    origin->fallback_file = origin_file;
    ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
    auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
    ASSERT_NE(async_file, nullptr);
    arrow::ProxyMemoryPool pool(arrow::default_memory_pool());
    uint8_t out[3] = {};
    auto future =
        allocating
            ? file->ReadAsync(arrow::io::IOContext(&pool), 1, 3).Then([](const std::shared_ptr<arrow::Buffer>& buffer) {
                EXPECT_EQ(buffer->ToString(), "ayl");
                return buffer->size();
              })
            : async_file->ReadAtAsyncInto(1, 3, out);
    EXPECT_TRUE(origin_file->started.is_finished());
    EXPECT_FALSE(future.is_finished());
    origin_file->completion.MarkFinished(3);
    ASSERT_AND_ASSIGN(const auto bytes_read, future.result());
    EXPECT_EQ(bytes_read, 3);
    if (allocating) {
      EXPECT_EQ(origin_file->async_pool, &pool);
    } else {
      EXPECT_EQ(std::string(reinterpret_cast<char*>(out), 3), "ayl");
    }
    ASSERT_STATUS_OK(file->Close());
    EXPECT_TRUE(origin_file->closed());
  }
}

TEST_F(TalonFileSystemServiceFreeTest, LargeReadsSupportBlockingOrigin) {
  auto config = Config();
  config.talon_mode = TalonMode::SmallReads;
  config.talon_small_read_threshold = 2;
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);

  uint8_t out[3] = {};
  // Origins without native async reads may complete synchronously.
  auto future = async_file->ReadAtAsyncInto(1, 3, out);
  ASSERT_AND_ASSIGN(const auto bytes_read, future.result());
  EXPECT_EQ(bytes_read, 3);
  EXPECT_EQ(std::string(reinterpret_cast<char*>(out), 3), "ayl");
  EXPECT_EQ(origin->input_file->read_calls.load(), 1);
}

TEST(TalonConfigTest, MaxIdlePerAddressProperty) {
  constexpr auto key = PROPERTY_FS_TALON_MAX_IDLE_PER_ADDR;
  api::Properties properties;
  ASSERT_AND_ASSIGN(const auto default_limit, api::GetValue<uint32_t>(properties, key));
  EXPECT_EQ(default_limit, 256U);
  ASSERT_EQ(api::SetValue(properties, key, "32"), std::nullopt);
  ASSERT_AND_ASSIGN(const auto explicit_limit, api::GetValue<uint32_t>(properties, key));
  EXPECT_EQ(explicit_limit, 32U);
  for (const char* invalid : {"0", "-1", "4294967296", "invalid"}) {
    EXPECT_NE(api::SetValue(properties, key, invalid), std::nullopt);
  }
}

TEST(TalonConfigTest, ChecksTypedPoolLimitOnlyWhenEnabled) {
  api::Properties properties;
  properties[PROPERTY_FS_STORAGE_TYPE] = std::string("remote");
  properties[PROPERTY_FS_TALON_COORDINATOR] = std::string("127.0.0.1:7000");
  properties[PROPERTY_FS_TALON_MAX_IDLE_PER_ADDR] = uint32_t{0};
  ArrowFileSystemConfig config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties, config));

  properties[PROPERTY_FS_TALON_MODE] = static_cast<uint32_t>(TalonMode::Full);
  const auto status = ArrowFileSystemConfig::create_file_system_config(properties, config);
  EXPECT_TRUE(status.IsInvalid()) << status;
  EXPECT_NE(status.message().find(PROPERTY_FS_TALON_MAX_IDLE_PER_ADDR), std::string::npos);
}

TEST(TalonConfigTest, ParsesExplicitProperties) {
  api::Properties properties;
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_STORAGE_TYPE, "remote"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_MODE, "1"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_COORDINATOR, "127.0.0.1:7000"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_BLOCK_SIZE, "8388608"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_MAX_IDLE_PER_ADDR, "32"), std::nullopt);

  ArrowFileSystemConfig config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties, config));
  EXPECT_EQ(config.talon_mode, TalonMode::Full);
  EXPECT_EQ(config.talon_coordinator, "127.0.0.1:7000");
  EXPECT_EQ(config.talon_block_size, 8388608U);
  EXPECT_EQ(config.talon_max_idle_per_addr, 32U);
}

TEST(TalonConfigTest, EnabledRequiresRemoteCoordinatorAndBlockSize) {
  for (const auto& [storage_type, coordinator, block_size] :
       std::vector<std::tuple<std::string, std::string, std::string>>{
           {"local", "127.0.0.1:7000", "8388608"}, {"remote", "", "8388608"}, {"remote", "127.0.0.1:7000", "0"}}) {
    api::Properties properties;
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_STORAGE_TYPE, storage_type.c_str()), std::nullopt);
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_MODE, "1"), std::nullopt);
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_COORDINATOR, coordinator.c_str()), std::nullopt);
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_BLOCK_SIZE, block_size.c_str()), std::nullopt);
    ArrowFileSystemConfig config;
    EXPECT_FALSE(ArrowFileSystemConfig::create_file_system_config(properties, config).ok());
  }
}

TEST(TalonConfigTest, CacheKeyIgnoresTalonSettingsWhenDisabled) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.cloud_provider = kCloudProviderAWS;
  base.address = "127.0.0.1:9000";
  base.bucket_name = "test-bucket";

  auto other_coordinator = base;
  other_coordinator.talon_coordinator = "127.0.0.1:7000";
  EXPECT_EQ(base.GetCacheKey(), other_coordinator.GetCacheKey());

  auto other_block_size = base;
  other_block_size.talon_block_size = 16777216;
  EXPECT_EQ(base.GetCacheKey(), other_block_size.GetCacheKey());

  auto other_pool_size = base;
  other_pool_size.talon_max_idle_per_addr = 32;
  EXPECT_EQ(base.GetCacheKey(), other_pool_size.GetCacheKey());

  auto other_threshold = base;
  other_threshold.talon_small_read_threshold = 1;
  EXPECT_EQ(base.GetCacheKey(), other_threshold.GetCacheKey());
}

TEST(TalonConfigTest, CacheKeyIncludesOnlyEffectiveReadRouting) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.talon_mode = TalonMode::Full;
  auto other = config;
  other.talon_small_read_threshold = 1;
  EXPECT_EQ(config.GetCacheKey(), other.GetCacheKey());
  other.talon_mode = TalonMode::SmallReads;
  EXPECT_NE(config.GetCacheKey(), other.GetCacheKey());
  config.talon_mode = TalonMode::SmallReads;
  EXPECT_NE(config.GetCacheKey(), other.GetCacheKey());
}

TEST(TalonConfigTest, CacheKeyIncludesTalonRouting) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.cloud_provider = kCloudProviderAWS;
  base.address = "127.0.0.1:9000";
  base.bucket_name = "test-bucket";

  auto enabled = base;
  enabled.talon_mode = TalonMode::Full;
  enabled.talon_coordinator = "127.0.0.1:7000";
  enabled.talon_block_size = 8388608;
  EXPECT_NE(base.GetCacheKey(), enabled.GetCacheKey());

  auto other_coordinator = enabled;
  other_coordinator.talon_coordinator = "127.0.0.1:7001";
  EXPECT_NE(enabled.GetCacheKey(), other_coordinator.GetCacheKey());

  auto other_block_size = enabled;
  other_block_size.talon_block_size = 16777216;
  EXPECT_NE(enabled.GetCacheKey(), other_block_size.GetCacheKey());

  auto other_pool_size = enabled;
  other_pool_size.talon_max_idle_per_addr = 32;
  EXPECT_NE(enabled.GetCacheKey(), other_pool_size.GetCacheKey());
}

TEST(TalonTestEnvTest, AddsTalonPropertiesWhenEnabled) {
  ScopedEnv storage_type(ENV_VAR_STORAGE_TYPE, "remote");
  ScopedEnv enabled(ENV_VAR_TALON_ENABLED, "true");
  ScopedEnv coordinator(ENV_VAR_TALON_COORDINATOR, "127.0.0.1:7000");
  ScopedEnv block_size(ENV_VAR_TALON_BLOCK_SIZE, "8388608");

  api::Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  EXPECT_TRUE(IsTalonEnv());
  EXPECT_EQ(api::GetValue<uint32_t>(properties, PROPERTY_FS_TALON_MODE).ValueOrDie(),
            static_cast<uint32_t>(TalonMode::Full));
  EXPECT_EQ(api::GetValue<std::string>(properties, PROPERTY_FS_TALON_COORDINATOR).ValueOrDie(), "127.0.0.1:7000");
  EXPECT_EQ(api::GetValue<uint32_t>(properties, PROPERTY_FS_TALON_BLOCK_SIZE).ValueOrDie(), 8388608U);
}

TEST(TalonTestEnvTest, AcceptsOneAndUsesDefaultBlockSize) {
  ScopedEnv storage_type(ENV_VAR_STORAGE_TYPE, "remote");
  ScopedEnv enabled(ENV_VAR_TALON_ENABLED, "1");
  ScopedEnv coordinator(ENV_VAR_TALON_COORDINATOR, "127.0.0.1:7000");
  ScopedEnv block_size(ENV_VAR_TALON_BLOCK_SIZE, nullptr);

  api::Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  EXPECT_TRUE(IsTalonEnv());
  EXPECT_EQ(api::GetValue<uint32_t>(properties, PROPERTY_FS_TALON_BLOCK_SIZE).ValueOrDie(), 268435456U);
}

TEST(TalonTestEnvTest, DoesNotAddTalonPropertiesWhenDisabled) {
  ScopedEnv storage_type(ENV_VAR_STORAGE_TYPE, "remote");
  ScopedEnv enabled(ENV_VAR_TALON_ENABLED, "false");
  ScopedEnv coordinator(ENV_VAR_TALON_COORDINATOR, "127.0.0.1:7000");
  ScopedEnv block_size(ENV_VAR_TALON_BLOCK_SIZE, "8388608");

  api::Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  EXPECT_FALSE(IsTalonEnv());
  EXPECT_FALSE(properties.contains(PROPERTY_FS_TALON_MODE));
  EXPECT_FALSE(properties.contains(PROPERTY_FS_TALON_COORDINATOR));
  EXPECT_FALSE(properties.contains(PROPERTY_FS_TALON_BLOCK_SIZE));
}

TEST(TalonTestEnvTest, RejectsEnabledTalonOutsideRemoteStorage) {
  ScopedEnv storage_type(ENV_VAR_STORAGE_TYPE, "local");
  ScopedEnv enabled(ENV_VAR_TALON_ENABLED, "true");
  ScopedEnv coordinator(ENV_VAR_TALON_COORDINATOR, "127.0.0.1:7000");

  api::Properties properties;
  const auto status = InitTestProperties(properties);
  EXPECT_TRUE(status.IsInvalid()) << status.ToString();
}

TEST(TalonTestEnvTest, RequiresCoordinatorWhenEnabled) {
  ScopedEnv storage_type(ENV_VAR_STORAGE_TYPE, "remote");
  ScopedEnv enabled(ENV_VAR_TALON_ENABLED, "true");
  ScopedEnv coordinator(ENV_VAR_TALON_COORDINATOR, nullptr);

  api::Properties properties;
  const auto status = InitTestProperties(properties);
  EXPECT_TRUE(status.IsInvalid()) << status.ToString();
}

TEST(TalonFileSystemTest, S3CompatibleProviderIdentityIsPartOfEquality) {
  ArrowFileSystemConfig aws;
  aws.storage_type = "remote";
  aws.cloud_provider = kCloudProviderAWS;
  aws.address = "127.0.0.1:9000";
  aws.bucket_name = "test-bucket";
  aws.access_key_id = "access-key";
  aws.access_key_value = "secret-key";
  aws.region = "us-east-1";
  aws.use_ssl = false;

  auto aliyun = aws;
  aliyun.cloud_provider = kCloudProviderAliyun;

  ASSERT_AND_ASSIGN(auto aws_fs, CreateArrowFileSystem(aws));
  ASSERT_AND_ASSIGN(auto aliyun_fs, CreateArrowFileSystem(aliyun));
  const auto aws_proxy = std::dynamic_pointer_cast<FileSystemProxy>(aws_fs);
  const auto aliyun_proxy = std::dynamic_pointer_cast<FileSystemProxy>(aliyun_fs);
  ASSERT_NE(aws_proxy, nullptr);
  ASSERT_NE(aliyun_proxy, nullptr);
  EXPECT_EQ(aws_proxy->base_path(), "test-bucket/");
  EXPECT_EQ(aliyun_proxy->base_path(), "test-bucket/");
  EXPECT_EQ(std::dynamic_pointer_cast<FileSystemProxy>(aws_proxy->base_fs()), nullptr);
  EXPECT_EQ(std::dynamic_pointer_cast<FileSystemProxy>(aliyun_proxy->base_fs()), nullptr);
  EXPECT_FALSE(aws_fs->Equals(*aliyun_fs));
  EXPECT_FALSE(aliyun_fs->Equals(*aws_fs));
}

class TalonProviderEqualityTest : public ::testing::TestWithParam<const char*> {};

TEST_P(TalonProviderEqualityTest, ComparesRealProviderAndTalonSymmetrically) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = GetParam();
  config.address = config.cloud_provider == kCloudProviderAzure ? "core.windows.net" : "127.0.0.1:9000";
  config.bucket_name = "test-bucket";
  config.access_key_id = "testaccount";
  config.access_key_value = "c2VjcmV0";
  config.region = "us-east-1";

  ASSERT_AND_ASSIGN(auto disabled, CreateArrowFileSystem(config));
  ASSERT_AND_ASSIGN(auto same_disabled, CreateArrowFileSystem(config));
  EXPECT_TRUE(disabled->Equals(*same_disabled));
  EXPECT_TRUE(same_disabled->Equals(*disabled));

  config.talon_mode = TalonMode::Full;
  config.talon_coordinator = "127.0.0.1:7000";
  ASSERT_AND_ASSIGN(auto enabled, CreateArrowFileSystem(config));
  ASSERT_EQ(disabled->type_name(), enabled->type_name());
  EXPECT_FALSE(enabled->Equals(*disabled));
  EXPECT_FALSE(disabled->Equals(*enabled));

  const auto disabled_proxy = std::dynamic_pointer_cast<FileSystemProxy>(disabled);
  const auto enabled_proxy = std::dynamic_pointer_cast<FileSystemProxy>(enabled);
  ASSERT_NE(disabled_proxy, nullptr);
  ASSERT_NE(enabled_proxy, nullptr);
  EXPECT_FALSE(enabled_proxy->base_fs()->Equals(*disabled_proxy->base_fs()));
  EXPECT_FALSE(disabled_proxy->base_fs()->Equals(*enabled_proxy->base_fs()));
  // A raw provider must also reject its bucket-rooted wrapper despite the matching type name.
  EXPECT_FALSE(disabled_proxy->base_fs()->Equals(*disabled));

  ASSERT_AND_ASSIGN(auto same_enabled, CreateArrowFileSystem(config));
  EXPECT_TRUE(enabled->Equals(*same_enabled));
  EXPECT_TRUE(same_enabled->Equals(*enabled));

  config.talon_mode = TalonMode::Disabled;
  config.access_key_id = "differentaccount";
  ASSERT_AND_ASSIGN(auto different_account, CreateArrowFileSystem(config));
  EXPECT_FALSE(disabled->Equals(*different_account));
  EXPECT_FALSE(different_account->Equals(*disabled));
}

INSTANTIATE_TEST_SUITE_P(CloudProviders,
                         TalonProviderEqualityTest,
                         ::testing::Values(kCloudProviderAWS, kCloudProviderAzure));

TEST(TalonBridgeErrorTest, PreservesClassificationAndDiagnostics) {
  const std::string message =
      "all replicas failed after refresh; last worker 10.0.0.8:9000: worker error: Timeout: backend deadline exceeded";
  for (const auto& [talon_code, storage_code] : {
           std::pair{talon::ffi::TalonErrorCode::Timeout, ExtendStatusCode::StorageTransientTimeout},
           std::pair{talon::ffi::TalonErrorCode::Unavailable, ExtendStatusCode::StorageTransientService},
           std::pair{talon::ffi::TalonErrorCode::RateLimited, ExtendStatusCode::StorageTransientThrottling},
           std::pair{talon::ffi::TalonErrorCode::VersionMismatch, ExtendStatusCode::AwsErrorPreConditionFailed},
           std::pair{talon::ffi::TalonErrorCode::NonRetryable, ExtendStatusCode::AwsErrorNonRetryable},
       }) {
    const auto status =
        talon::internal::ToArrowIoResult("read", static_cast<int32_t>(talon_code), 0, message.c_str()).status();
    const auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), storage_code);
    EXPECT_EQ(detail->retryable(), talon_code == talon::ffi::TalonErrorCode::Timeout ||
                                       talon_code == talon::ffi::TalonErrorCode::Unavailable ||
                                       talon_code == talon::ffi::TalonErrorCode::RateLimited);
    EXPECT_NE(status.message().find(message), std::string::npos);
  }
}

TEST(TalonBridgeErrorTest, HandlesInvalidMissingAndUnknownErrors) {
  EXPECT_TRUE(talon::internal::ToArrowIoResult(
                  "read", static_cast<int32_t>(talon::ffi::TalonErrorCode::InvalidArgument), 0, "bad range")
                  .status()
                  .IsInvalid());
  const auto missing = talon::internal::ToArrowIoResult(
                           "stat", static_cast<int32_t>(talon::ffi::TalonErrorCode::NotFound), 0, "missing object")
                           .status();
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(missing), ENOENT);
  const auto unknown = talon::internal::ToArrowIoResult("read", 99, 0, nullptr).status();
  EXPECT_TRUE(unknown.IsIOError());
  EXPECT_EQ(unknown.detail(), nullptr);
  EXPECT_NE(unknown.message().find("unknown Talon error"), std::string::npos);
  EXPECT_EQ(talon::internal::ToArrowIoResult("read", 0, 42, nullptr).ValueOrDie(), 42);
  EXPECT_TRUE(
      talon::internal::ToArrowIoResult("stat", 0, std::numeric_limits<uint64_t>::max(), nullptr).status().IsIOError());
}

TEST(TalonBridgeTest, MapsClientCreationErrorAndPreservesTalonMessage) {
  ArrowFileSystemConfig config;
  config.talon_coordinator = "127.0.0.1:7000";
  config.talon_block_size = 0;
  config.talon_max_idle_per_addr = 8;
  const auto result = talon::TalonClient::Make(config);

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsIOError()) << result.status().ToString();
  EXPECT_NE(result.status().message().find("Failed to create Talon client"), std::string::npos);
  EXPECT_NE(result.status().message().find("block_size must be non-zero"), std::string::npos);
}

TEST(TalonBridgeTest, MapsOpenErrorAndPreservesTalonMessage) {
  ArrowFileSystemConfig config;
  config.talon_coordinator = "127.0.0.1:7000";
  config.talon_block_size = 8U * 1024U * 1024U;
  config.talon_max_idle_per_addr = 8;
  ASSERT_AND_ASSIGN(auto client, talon::TalonClient::Make(config));

  const auto result = client->OpenObject("aws", "test-bucket", "", std::nullopt);

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsIOError()) << result.status().ToString();
  EXPECT_NE(result.status().message().find("Failed to open Talon object"), std::string::npos);
  EXPECT_NE(result.status().message().find("bucket and object key must be non-empty"), std::string::npos);
}

TEST(TalonBridgeTest, OpensVolcengineObjectWithKnownMetadata) {
  ArrowFileSystemConfig config;
  config.talon_coordinator = "127.0.0.1:7000";
  config.talon_block_size = 8U * 1024U * 1024U;
  config.talon_max_idle_per_addr = 8;
  ASSERT_AND_ASSIGN(auto client, talon::TalonClient::Make(config));
  ASSERT_AND_ASSIGN(auto reader,
                    client->OpenObject("volcengine", "test-bucket", "path/a", talon::TalonObjectStat{7, "etag"}));
  EXPECT_EQ(reader.KnownSize(), 7);
}

TEST(TalonBridgeTest, MapsAsyncErrorAndPreservesTalonMessage) {
  ArrowFileSystemConfig config;
  config.talon_coordinator = "127.0.0.1:7000";
  config.talon_block_size = 8U * 1024U * 1024U;
  config.talon_max_idle_per_addr = 8;
  ASSERT_AND_ASSIGN(auto client, talon::TalonClient::Make(config));
  ASSERT_AND_ASSIGN(auto reader, client->OpenObject("aws", "test-bucket", "path/a", talon::TalonObjectStat{0, ""}));

  const auto result = reader.ReadAtAsync(0, std::numeric_limits<uint64_t>::max(), nullptr).result();

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsInvalid()) << result.status().ToString();
  EXPECT_NE(result.status().message().find("Failed to read Talon object"), std::string::npos);
  EXPECT_NE(result.status().message().find("Talon read length exceeds isize::MAX"), std::string::npos);
}

TEST(TalonFileSystemTest, PreservesOuterSubtreeAndExtensionInterfaces) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAWS;
  config.address = "http://127.0.0.1:9000";
  config.bucket_name = "test-bucket";
  config.access_key_id = "minioadmin";
  config.access_key_value = "minioadmin";
  config.region = "us-east-1";
  config.s3_crt_async_read = false;
  config.talon_mode = TalonMode::Full;
  config.talon_coordinator = "127.0.0.1:7000";
  config.talon_block_size = 8388608;

  ASSERT_AND_ASSIGN(auto fs, CreateArrowFileSystem(config));
  auto proxy = std::dynamic_pointer_cast<FileSystemProxy>(fs);
  ASSERT_NE(proxy, nullptr);
  EXPECT_EQ(proxy->base_path(), "test-bucket/");
  EXPECT_EQ(std::dynamic_pointer_cast<FileSystemProxy>(proxy->base_fs()), nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<UploadConditional>(fs), nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<UploadSizable>(fs), nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<Observable>(fs), nullptr);
}

TEST_F(TalonFileSystemServiceFreeTest, ProducerAcceptsRawProviderFilesystem) {
  const auto config = Config();
  auto origin = std::make_shared<RecordingFileSystem>("origin");

  ASSERT_AND_ASSIGN(auto talon_fs, TalonFileSystemProducer(config, origin).Make());
  EXPECT_EQ(talon_fs->type_name(), "recording");
  EXPECT_EQ(std::dynamic_pointer_cast<FileSystemProxy>(talon_fs), nullptr);
}

TEST_F(TalonFileSystemServiceFreeTest, LazyMetadataIsSharedByConcurrentSizeAndReadRequests) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  const auto origin_file = origin->input_file;
  origin_file->async_result = arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>>::Make();
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  ASSERT_EQ(origin_file->metadata_read_calls, 0);
  EXPECT_EQ(origin_file->async_metadata_read_calls.load(), 0);
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);

  std::vector<std::future<arrow::Future<int64_t>>> submissions;
  for (int i = 0; i < 16; ++i) {
    submissions.push_back(std::async(std::launch::async, [async_file] { return async_file->GetSizeAsync(); }));
  }
  EXPECT_TRUE(origin_file->metadata_requested.Wait(5));
  for (auto& submitted : submissions) {
    EXPECT_EQ(submitted.wait_for(std::chrono::milliseconds(100)), std::future_status::ready);
  }
  auto buffer = file->ReadAsync({}, 7, 64 * 1024 * 1024);
  uint8_t out = 0;
  auto read_into = async_file->ReadAtAsyncInto(7, 16, &out);
  EXPECT_FALSE(buffer.is_finished());
  EXPECT_FALSE(read_into.is_finished());
  EXPECT_EQ(origin_file->async_metadata_read_calls.load(), 1);
  origin_file->async_result.MarkFinished(origin_file->metadata_result);

  for (auto& submitted : submissions) {
    auto future = submitted.get();
    ASSERT_TRUE(future.Wait(5));
    ASSERT_AND_ASSIGN(const auto size, future.result());
    EXPECT_EQ(size, 7);
  }
  ASSERT_TRUE(buffer.Wait(5));
  ASSERT_AND_ASSIGN(const auto bytes, buffer.result());
  EXPECT_EQ(bytes->size(), 0);
  ASSERT_TRUE(read_into.Wait(5));
  ASSERT_AND_ASSIGN(const auto bytes_read, read_into.result());
  EXPECT_EQ(bytes_read, 0);
  EXPECT_EQ(origin_file->metadata_read_calls, 0);
  EXPECT_EQ(origin_file->async_metadata_read_calls.load(), 1);
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, PendingLazyInitializationOutlivesClosedFileAndFilesystem) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  auto destroyed = arrow::Future<std::thread::id>::Make();
  origin->input_file->on_destroy = [destroyed]() mutable { destroyed.MarkFinished(std::this_thread::get_id()); };
  auto metadata = arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>>::Make();
  origin->input_file->async_result = metadata;
  const auto metadata_result = origin->input_file->metadata_result;
  const auto requested = origin->input_file->metadata_requested;
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  ASSERT_EQ(origin->input_file->metadata_read_calls, 0);
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);
  auto size = async_file->GetSizeAsync();
  EXPECT_TRUE(requested.Wait(5));
  ASSERT_STATUS_OK(file->Close());
  EXPECT_TRUE(file->GetSize().status().IsInvalid());
  file.reset();
  fs.reset();
  origin.reset();
  EXPECT_FALSE(destroyed.is_finished());

  // Stand in for the CRT callback thread. No final origin ownership may be
  // released here, even if the initialization continuation finishes quickly.
  metadata.MarkFinished(metadata_result);
  ASSERT_TRUE(size.Wait(5));
  ASSERT_AND_ASSIGN(const auto resolved_size, size.result());
  EXPECT_EQ(resolved_size, 7);
  ASSERT_TRUE(destroyed.Wait(5));
  EXPECT_NE(destroyed.result().ValueOrDie(), std::this_thread::get_id());
}

TEST_F(TalonFileSystemServiceFreeTest, LazyInitializationRetriesOriginalMetadataErrorOnSameHandle) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  const auto metadata = origin->input_file->metadata_result;
  const auto error = MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "metadata denied", "provider detail");
  origin->input_file->metadata_result = error;
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  EXPECT_EQ(file->GetSize().status(), error);
  origin->input_file->metadata_result = metadata;
  ASSERT_AND_ASSIGN(const auto size, file->GetSize());
  EXPECT_EQ(size, 7);
  EXPECT_EQ(origin->input_file->metadata_read_calls, 0);
  EXPECT_EQ(origin->input_file->async_metadata_read_calls.load(), 2);
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, LazyInitializationCanRetryFromErrorContinuation) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  auto metadata = arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>>::Make();
  origin->input_file->async_result = metadata;
  const auto error = MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "metadata denied", "provider detail");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);
  auto recovered = async_file->GetSizeAsync().Then(
      [](const int64_t size) { return arrow::Future<int64_t>::MakeFinished(size); },
      [origin_file = origin->input_file, async_file, error](const arrow::Status& status) {
        EXPECT_EQ(status, error);
        origin_file->async_result = {};
        // Retry while the previous initialization is still completing its
        // callbacks, before that attempt's ownership handle is released.
        return async_file->GetSizeAsync();
      });
  EXPECT_TRUE(origin->input_file->metadata_requested.Wait(5));
  metadata.MarkFinished(error);
  ASSERT_TRUE(recovered.Wait(5));
  ASSERT_AND_ASSIGN(const auto size, recovered.result());
  EXPECT_EQ(size, 7);
  EXPECT_EQ(origin->input_file->async_metadata_read_calls.load(), 2);
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, InvalidAndZeroLengthReadsDoNotInitializeReader) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);
  EXPECT_TRUE(async_file->ReadAtAsyncInto(-1, 1, nullptr).result().status().IsInvalid());
  EXPECT_TRUE(async_file->ReadAtAsyncInto(0, -1, nullptr).result().status().IsInvalid());
  EXPECT_TRUE(async_file->ReadAtAsyncInto(0, 1, nullptr).result().status().IsInvalid());
  ASSERT_AND_ASSIGN(const auto bytes_read, async_file->ReadAtAsyncInto(0, 0, nullptr).result());
  EXPECT_EQ(bytes_read, 0);
  ASSERT_AND_ASSIGN(const auto buffer, file->ReadAsync({}, 0, 0).result());
  EXPECT_EQ(buffer->size(), 0);
  ASSERT_STATUS_OK(file->Seek(1));
  ASSERT_AND_ASSIGN(const auto position, file->Tell());
  EXPECT_EQ(position, 1);
  ASSERT_STATUS_OK(file->Close());
  EXPECT_EQ(origin->input_file->metadata_read_calls, 0);
  EXPECT_EQ(origin->input_file->async_metadata_read_calls.load(), 0);
}

TEST_F(TalonFileSystemServiceFreeTest, CachesObjectMetadataAcrossOpens) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  ASSERT_AND_ASSIGN(auto first, fs->OpenInputFile("prefix/object"));
  ASSERT_EQ(origin->input_open_calls, 1);
  EXPECT_EQ(origin->input_file->metadata_read_calls, 0);
  EXPECT_EQ(origin->input_file->async_metadata_read_calls.load(), 0);
  ASSERT_AND_ASSIGN(const auto size, first->GetSize());
  EXPECT_EQ(size, 7);
  EXPECT_EQ(origin->input_file->async_metadata_read_calls.load(), 1);
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(first.get());
  ASSERT_NE(async_file, nullptr);
  ASSERT_AND_ASSIGN(const auto async_size, async_file->GetSizeAsync().result());
  EXPECT_EQ(async_size, 7);
  ASSERT_STATUS_OK(first->Close());

  // Each reader owns an origin file, but cached metadata needs no new HEAD.
  const auto metadata_error = arrow::Status::IOError("origin metadata unavailable");
  const auto second_origin = std::make_shared<MetadataInputFile>(99);
  second_origin->metadata_result = metadata_error;
  origin->input_file = second_origin;
  arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
  ASSERT_AND_ASSIGN(auto second, fs->OpenInputFile(info));
  ASSERT_AND_ASSIGN(const auto cached_size, second->GetSize());
  EXPECT_EQ(cached_size, 7);
  EXPECT_EQ(origin->input_open_calls, 2);
  EXPECT_EQ(second_origin->metadata_read_calls, 0);
  EXPECT_EQ(second_origin->async_metadata_read_calls.load(), 0);
  origin->input_file = std::make_shared<MetadataInputFile>();
  ASSERT_AND_ASSIGN(auto stream, fs->OpenInputStream("prefix/object"));
  EXPECT_EQ(origin->input_open_calls, 3);
  EXPECT_EQ(origin->input_file->metadata_read_calls, 0);
  ASSERT_STATUS_OK(second->Close());
  EXPECT_TRUE(second_origin->closed());
  EXPECT_FALSE(origin->input_file->closed());
  ASSERT_STATUS_OK(stream->Close());
  EXPECT_TRUE(origin->input_file->closed());

  origin->input_file = std::make_shared<MetadataInputFile>();
  origin->input_file->metadata_result = metadata_error;
  ASSERT_AND_ASSIGN(auto other_file, fs->OpenInputFile("prefix/other"));
  EXPECT_EQ(other_file->GetSize().status(), metadata_error);
  auto other_origin = std::make_shared<RecordingFileSystem>("other-origin");
  other_origin->input_file->metadata_result = metadata_error;
  ASSERT_AND_ASSIGN(auto other_fs, Wrap(Config(), other_origin));
  ASSERT_AND_ASSIGN(auto separate_file, other_fs->OpenInputFile("prefix/object"));
  EXPECT_EQ(separate_file->GetSize().status(), metadata_error);
}

TEST_F(TalonFileSystemServiceFreeTest, FailedMetadataLookupCanBeRetried) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  const auto error =
      MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "origin metadata denied", "provider detail");
  origin->input_file->metadata_result = error;
  ASSERT_AND_ASSIGN(auto failed_file, fs->OpenInputFile("prefix/object"));
  EXPECT_EQ(failed_file->GetSize().status(), error);
  ASSERT_STATUS_OK(failed_file->Close());

  origin->input_file = std::make_shared<MetadataInputFile>();
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  EXPECT_EQ(origin->input_open_calls, 2);
  ASSERT_AND_ASSIGN(const auto size, file->GetSize());
  EXPECT_EQ(size, 7);
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, RejectsMissingOrEmptyETag) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  for (const auto& metadata : std::vector<std::shared_ptr<const arrow::KeyValueMetadata>>{
           nullptr, arrow::key_value_metadata({"owner"}, {"test-owner"}), arrow::key_value_metadata({"ETag"}, {""}),
           arrow::key_value_metadata({"ETag"}, {"\"\""})}) {
    origin->input_file = std::make_shared<MetadataInputFile>();
    origin->input_file->metadata_result = metadata;
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
    const auto result = file->GetSize();
    ASSERT_FALSE(result.ok());
    EXPECT_TRUE(result.status().IsIOError()) << result.status();
    EXPECT_NE(result.status().message().find("ETag"), std::string::npos);
  }
}

TEST_F(TalonFileSystemServiceFreeTest, EvictsLeastRecentlyUsedObjectMetadata) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  for (int i = 0; i < 4096; ++i) {
    origin->input_file = std::make_shared<MetadataInputFile>();
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/" + std::to_string(i)));
    ASSERT_AND_ASSIGN(const auto size, file->GetSize());
    EXPECT_EQ(size, 7);
    ASSERT_STATUS_OK(file->Close());
  }
  ASSERT_EQ(origin->input_open_calls, 4096);
  origin->input_file = std::make_shared<MetadataInputFile>();
  ASSERT_AND_ASSIGN(auto recent, fs->OpenInputFile("prefix/0"));
  ASSERT_STATUS_OK(recent->Close());
  origin->input_file = std::make_shared<MetadataInputFile>();
  ASSERT_AND_ASSIGN(auto extra, fs->OpenInputFile("prefix/extra"));
  ASSERT_AND_ASSIGN(const auto extra_size, extra->GetSize());
  EXPECT_EQ(extra_size, 7);
  ASSERT_STATUS_OK(extra->Close());

  const auto metadata_error = arrow::Status::IOError("origin metadata unavailable");
  origin->input_file = std::make_shared<MetadataInputFile>();
  origin->input_file->metadata_result = metadata_error;
  ASSERT_AND_ASSIGN(auto retained, fs->OpenInputFile("prefix/0"));
  ASSERT_AND_ASSIGN(const auto retained_size, retained->GetSize());
  EXPECT_EQ(retained_size, 7);
  ASSERT_STATUS_OK(retained->Close());
  origin->input_file = std::make_shared<MetadataInputFile>();
  origin->input_file->metadata_result = metadata_error;
  ASSERT_AND_ASSIGN(auto evicted, fs->OpenInputFile("prefix/1"));
  EXPECT_EQ(evicted->GetSize().status(), metadata_error);
}

TEST_F(TalonFileSystemServiceFreeTest, OwnsConfigWithNormalizedBucket) {
  auto config = Config();
  config.bucket_name = "test-bucket///";
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  origin->input_file = std::make_shared<MetadataInputFile>(0);
  ASSERT_AND_ASSIGN(auto fs, TalonFileSystemProducer(config, origin).Make());

  config.bucket_name = "different-bucket";
  config.cloud_provider = "unsupported";
  arrow::fs::FileInfo info("test-bucket/prefix/object", arrow::fs::FileType::File);
  info.set_size(0);
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));
  ASSERT_AND_ASSIGN(const auto size, file->GetSize());
  EXPECT_EQ(size, 0);
  EXPECT_EQ(origin->input_open_calls, 1);
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, ForwardsPoolLimitAndPreservesTalonValidationError) {
  auto config = Config();
  config.talon_max_idle_per_addr = 0;
  const auto result = Wrap(config, std::make_shared<RecordingFileSystem>("origin"));
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsIOError()) << result.status();
  EXPECT_NE(result.status().message().find("max_idle_per_addr must be non-zero"), std::string::npos);
}

TEST_F(TalonFileSystemServiceFreeTest, BrokerOriginIsValidatedOnRequest) {
  auto config = Config();
  config.credential_endpoint = "https://broker/assume-role";
  config.region = "us-west-2";
  config.role_arn = "arn:aws:iam::123456789012:role/reader";
  config.address = "https://storage.example/path-not-an-endpoint";
  ASSERT_AND_ASSIGN(const auto client, talon::TalonClient::Make(config));
  const auto result = client->Load({"s3://bucket/empty", 0, ""});
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().message().find("origin endpoint"), std::string::npos);
  EXPECT_EQ(result.status().message().find("path-not-an-endpoint"), std::string::npos);
}

TEST_F(TalonFileSystemServiceFreeTest, NamespaceIsolatesFilesystemIdentityWithoutBroker) {
  auto config = Config();
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto unset_namespace, Wrap(config, origin));
  config.talon_namespace = "default";
  ASSERT_AND_ASSIGN(auto explicit_namespace, Wrap(config, origin));
  EXPECT_FALSE(unset_namespace->Equals(*explicit_namespace));
  config.talon_namespace = "tenant-a";
  ASSERT_AND_ASSIGN(auto tenant_a, Wrap(config, origin));
  config.talon_namespace = "tenant-b";
  ASSERT_AND_ASSIGN(auto tenant_b, Wrap(config, origin));
  EXPECT_FALSE(tenant_a->Equals(*tenant_b));
  EXPECT_FALSE(tenant_a->Equals(*unset_namespace));
}

TEST_F(TalonFileSystemServiceFreeTest, BrokerEndpointsDoNotShareFilesystemIdentity) {
  auto config = Config();
  config.credential_endpoint = "https://broker/assume-role";
  config.region = "us-west-2";
  config.role_arn = "arn:aws:iam::123456789012:role/reader";
  config.address = "https://storage.example";
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto first, Wrap(config, origin));
  config.credential_endpoint = "https://other-broker/assume-role";
  ASSERT_AND_ASSIGN(auto second, Wrap(config, origin));
  EXPECT_FALSE(first->Equals(*second));
}

TEST_F(TalonFileSystemServiceFreeTest, OpenWithFileInfoRejectsNonFiles) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  for (const auto type : {arrow::fs::FileType::NotFound, arrow::fs::FileType::Directory}) {
    SCOPED_TRACE(static_cast<int>(type));
    const arrow::fs::FileInfo info("prefix/object", type);
    const auto file = fs->OpenInputFile(info);
    const auto stream = fs->OpenInputStream(info);
    for (const auto& status : {file.status(), stream.status()}) {
      EXPECT_TRUE(status.IsIOError()) << status;
      EXPECT_NE(status.message().find("test-bucket/prefix/object"), std::string::npos);
      if (type == arrow::fs::FileType::NotFound) {
        EXPECT_EQ(arrow::internal::ErrnoFromStatus(status), ENOENT);
      } else {
        EXPECT_NE(status.message().find("Not a regular file"), std::string::npos);
      }
    }
  }
  EXPECT_EQ(origin->input_open_calls, 0);
}

TEST_F(TalonFileSystemServiceFreeTest, OpenWithFileInfoAcceptsFileAndUnknown) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  for (const auto type : {arrow::fs::FileType::File, arrow::fs::FileType::Unknown}) {
    for (const int64_t size : {-1, 0, 7}) {
      SCOPED_TRACE(::testing::Message() << "type=" << static_cast<int>(type) << ", size=" << size);
      arrow::fs::FileInfo info("prefix/object", type);
      info.set_size(size);
      origin->input_file = std::make_shared<MetadataInputFile>();
      ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));
      origin->input_file = std::make_shared<MetadataInputFile>();
      ASSERT_AND_ASSIGN(auto stream, fs->OpenInputStream(info));
      ASSERT_AND_ASSIGN(const auto known_size, file->GetSize());
      EXPECT_EQ(known_size, size >= 0 ? size : 7);
      ASSERT_STATUS_OK(file->Close());
      ASSERT_STATUS_OK(stream->Close());
    }
  }
  EXPECT_EQ(origin->input_open_calls, 12);
}

TEST_F(TalonFileSystemServiceFreeTest, KnownSizeSkipsMetadataAndReachesOrigin) {
  for (const auto type : {arrow::fs::FileType::File, arrow::fs::FileType::Unknown}) {
    for (const int64_t size : {0, 7}) {
      SCOPED_TRACE(::testing::Message() << "type=" << static_cast<int>(type) << ", size=" << size);
      auto origin = std::make_shared<RecordingFileSystem>("origin");
      origin->input_file = std::make_shared<MetadataInputFile>(size);
      origin->input_file->metadata_result = arrow::Status::IOError("HEAD must not be called");
      auto config = Config();
      config.talon_coordinator = "127.0.0.1:0";
      ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
      arrow::fs::FileInfo info("prefix/object", type);
      info.set_size(size);
      ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));
      EXPECT_EQ(origin->last_input_size.load(), size);
      EXPECT_EQ(origin->last_input_path, "test-bucket/prefix/object");
      ASSERT_AND_ASSIGN(const auto known_size, file->GetSize());
      EXPECT_EQ(known_size, size);
      auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
      ASSERT_NE(async_file, nullptr);
      ASSERT_AND_ASSIGN(const auto async_size, async_file->GetSizeAsync().result());
      EXPECT_EQ(async_size, size);
      ASSERT_AND_ASSIGN(const auto buffer, file->ReadAsync({}, 0, 16).result());
      EXPECT_EQ(buffer->ToString(), size == 0 ? "" : "payload");
      EXPECT_EQ(origin->input_file->metadata_read_calls, 0);
      EXPECT_EQ(origin->input_file->async_metadata_read_calls.load(), 0);
      ASSERT_STATUS_OK(file->Close());

      origin->input_file = std::make_shared<MetadataInputFile>(size);
      origin->input_file->metadata_result = arrow::Status::IOError("HEAD must not be called");
      ASSERT_AND_ASSIGN(auto stream, fs->OpenInputStream(info));
      ASSERT_AND_ASSIGN(const auto bytes, stream->Read(16));
      EXPECT_EQ(bytes->ToString(), size == 0 ? "" : "payload");
      EXPECT_EQ(origin->input_file->metadata_read_calls, 0);
      EXPECT_EQ(origin->input_file->async_metadata_read_calls.load(), 0);
      ASSERT_STATUS_OK(stream->Close());
    }
  }
}

TEST_F(TalonFileSystemServiceFreeTest, EquivalentTalonFilesystemsCompareEqualSymmetrically) {
  const auto config = Config();
  ASSERT_AND_ASSIGN(auto left, Wrap(config, std::make_shared<RecordingFileSystem>("origin")));
  ASSERT_AND_ASSIGN(auto right, Wrap(config, std::make_shared<RecordingFileSystem>("origin")));
  const auto left_proxy = std::dynamic_pointer_cast<FileSystemProxy>(left);
  const auto right_proxy = std::dynamic_pointer_cast<FileSystemProxy>(right);
  ASSERT_NE(left_proxy, nullptr);
  ASSERT_NE(right_proxy, nullptr);
  const auto left_talon = left_proxy->base_fs();
  const auto right_talon = right_proxy->base_fs();

  EXPECT_TRUE(left_talon->Equals(*right_talon));
  EXPECT_TRUE(right_talon->Equals(*left_talon));
}

TEST_F(TalonFileSystemServiceFreeTest, PreservesProviderTypeName) {
  const auto config = Config();
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, std::make_shared<RecordingFileSystem>("origin")));

  EXPECT_EQ(fs->type_name(), "recording");
}

TEST_F(TalonFileSystemServiceFreeTest, TalonFilesystemComparesUnequalToUndecoratedOrigin) {
  const auto config = Config();
  ASSERT_AND_ASSIGN(auto talon_fs, Wrap(config, std::make_shared<RecordingFileSystem>("origin")));
  auto raw_origin = std::make_shared<RecordingFileSystem>("origin");

  const auto talon_proxy = std::dynamic_pointer_cast<FileSystemProxy>(talon_fs);
  ASSERT_NE(talon_proxy, nullptr);
  const auto talon_base = talon_proxy->base_fs();
  EXPECT_FALSE(talon_base->Equals(*raw_origin));
  EXPECT_FALSE(raw_origin->Equals(*talon_base));
}

TEST_F(TalonFileSystemServiceFreeTest, EqualityIncludesTalonRouting) {
  const auto config = Config();
  ASSERT_AND_ASSIGN(auto base, Wrap(config, std::make_shared<RecordingFileSystem>("origin")));
  const auto base_proxy = std::dynamic_pointer_cast<FileSystemProxy>(base);
  ASSERT_NE(base_proxy, nullptr);
  const auto base_talon = base_proxy->base_fs();

  auto other_coordinator = config;
  other_coordinator.talon_coordinator = "127.0.0.1:7001";
  ASSERT_AND_ASSIGN(auto coordinator_fs, Wrap(other_coordinator, std::make_shared<RecordingFileSystem>("origin")));
  const auto coordinator_proxy = std::dynamic_pointer_cast<FileSystemProxy>(coordinator_fs);
  ASSERT_NE(coordinator_proxy, nullptr);
  const auto coordinator_talon = coordinator_proxy->base_fs();
  EXPECT_FALSE(base_talon->Equals(*coordinator_talon));
  EXPECT_FALSE(coordinator_talon->Equals(*base_talon));

  auto other_block_size = config;
  other_block_size.talon_block_size *= 2;
  ASSERT_AND_ASSIGN(auto block_size_fs, Wrap(other_block_size, std::make_shared<RecordingFileSystem>("origin")));
  const auto block_size_proxy = std::dynamic_pointer_cast<FileSystemProxy>(block_size_fs);
  ASSERT_NE(block_size_proxy, nullptr);
  const auto block_size_talon = block_size_proxy->base_fs();
  EXPECT_FALSE(base_talon->Equals(*block_size_talon));
  EXPECT_FALSE(block_size_talon->Equals(*base_talon));

  auto other_pool_size = config;
  other_pool_size.talon_max_idle_per_addr = 32;
  ASSERT_AND_ASSIGN(auto pool_size_fs, Wrap(other_pool_size, std::make_shared<RecordingFileSystem>("origin")));
  const auto pool_size_proxy = std::dynamic_pointer_cast<FileSystemProxy>(pool_size_fs);
  ASSERT_NE(pool_size_proxy, nullptr);
  const auto pool_size_talon = pool_size_proxy->base_fs();
  EXPECT_FALSE(base_talon->Equals(*pool_size_talon));
  EXPECT_FALSE(pool_size_talon->Equals(*base_talon));

  auto other_threshold = config;
  other_threshold.talon_small_read_threshold = 1;
  ASSERT_AND_ASSIGN(auto threshold_fs, Wrap(other_threshold, std::make_shared<RecordingFileSystem>("origin")));
  EXPECT_TRUE(base->Equals(*threshold_fs));
  EXPECT_TRUE(threshold_fs->Equals(*base));

  auto small_reads = config;
  small_reads.talon_mode = TalonMode::SmallReads;
  ASSERT_AND_ASSIGN(auto small_reads_fs, Wrap(small_reads, std::make_shared<RecordingFileSystem>("origin")));
  EXPECT_FALSE(base->Equals(*small_reads_fs));
  EXPECT_FALSE(small_reads_fs->Equals(*base));

  small_reads.talon_small_read_threshold = 1;
  ASSERT_AND_ASSIGN(auto small_threshold_fs, Wrap(small_reads, std::make_shared<RecordingFileSystem>("origin")));
  EXPECT_FALSE(small_reads_fs->Equals(*small_threshold_fs));
  EXPECT_FALSE(small_threshold_fs->Equals(*small_reads_fs));
}

TEST_F(TalonFileSystemServiceFreeTest, ForwardsStreamingFileInfoGeneratorWithFullSelector) {
  const auto config = Config();
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));

  arrow::fs::FileSelector selector;
  selector.base_dir = "prefix";
  selector.allow_not_found = true;
  selector.recursive = true;
  selector.max_recursion = 3;
  auto generator = fs->GetFileInfoGenerator(selector);

  EXPECT_EQ(origin->synchronous_listing_calls(), 0);
  EXPECT_EQ(origin->generator_calls(), 1);
  EXPECT_EQ(origin->last_generator_selector().base_dir, "test-bucket/prefix");
  EXPECT_TRUE(origin->last_generator_selector().allow_not_found);
  EXPECT_TRUE(origin->last_generator_selector().recursive);
  EXPECT_EQ(origin->last_generator_selector().max_recursion, 3);

  ASSERT_AND_ASSIGN(auto first, generator().result());
  ASSERT_EQ(first.size(), 1);
  EXPECT_EQ(first[0].path(), "prefix/first");

  ASSERT_AND_ASSIGN(auto second, generator().result());
  ASSERT_EQ(second.size(), 1);
  EXPECT_EQ(second[0].path(), "prefix/second");

  ASSERT_AND_ASSIGN(auto end, generator().result());
  EXPECT_TRUE(end.empty());
}

TEST_F(TalonFileSystemServiceFreeTest, ReadMetadataReusesOriginFileFromCacheMiss) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
  info.set_size(7);
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));
  ASSERT_AND_ASSIGN(const auto size, file->GetSize());
  EXPECT_EQ(size, 7);
  EXPECT_EQ(origin->input_open_calls, 1);

  ASSERT_AND_ASSIGN(auto metadata, file->ReadMetadata());
  ASSERT_NE(metadata, nullptr);
  ASSERT_AND_ASSIGN(const auto etag, metadata->Get("ETag"));
  ASSERT_AND_ASSIGN(const auto content_length, metadata->Get("Content-Length"));
  ASSERT_AND_ASSIGN(const auto owner, metadata->Get("owner"));
  EXPECT_EQ(etag, "\"origin-etag\"");
  EXPECT_EQ(content_length, "7");
  EXPECT_EQ(owner, "test-owner");
  EXPECT_EQ(origin->last_input_path, "test-bucket/prefix/object");
  ASSERT_AND_ASSIGN(auto repeated, file->ReadMetadata());
  EXPECT_EQ(repeated, metadata);
  ASSERT_AND_ASSIGN(auto async_metadata, file->ReadMetadataAsync().result());
  EXPECT_EQ(async_metadata, metadata);
  EXPECT_EQ(origin->input_open_calls, 1);

  ASSERT_STATUS_OK(file->Close());
  EXPECT_TRUE(origin->input_file->closed());
  EXPECT_TRUE(file->ReadMetadata().status().IsInvalid());
  EXPECT_TRUE(file->ReadMetadataAsync().result().status().IsInvalid());
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, ReadMetadataAsyncForwardsProviderFutureAndContext) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputStream("prefix/object"));
  arrow::ProxyMemoryPool pool(arrow::default_memory_pool());
  origin->input_file->async_result = arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>>::Make();

  auto future = file->ReadMetadataAsync(arrow::io::IOContext(&pool));
  EXPECT_FALSE(future.is_finished());
  EXPECT_EQ(origin->input_file->async_pool, &pool);
  EXPECT_EQ(origin->last_input_path, "test-bucket/prefix/object");
  origin->input_file->async_result.MarkFinished(origin->input_file->metadata_result);
  ASSERT_AND_ASSIGN(auto metadata, future.result());
  ASSERT_NE(metadata, nullptr);
  ASSERT_AND_ASSIGN(const auto etag, metadata->Get("ETag"));
  EXPECT_EQ(etag, "\"origin-etag\"");
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, ReadMetadataPreservesOriginErrorsAndNullMetadata) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  ASSERT_AND_ASSIGN(auto first, fs->OpenInputFile("prefix/object"));
  ASSERT_STATUS_OK(first->Close());
  origin->input_file = std::make_shared<MetadataInputFile>();
  const auto error =
      MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "origin metadata denied", "provider detail");
  origin->input_open_status = error;
  EXPECT_EQ(fs->OpenInputFile("prefix/object").status(), error);

  origin->input_open_status = arrow::Status::OK();
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  origin->input_file->metadata_result = error;
  EXPECT_EQ(file->ReadMetadata().status(), error);
  EXPECT_EQ(file->ReadMetadataAsync().result().status(), error);

  origin->input_file->metadata_result = std::shared_ptr<const arrow::KeyValueMetadata>{};
  ASSERT_AND_ASSIGN(auto metadata, file->ReadMetadata());
  EXPECT_EQ(metadata, nullptr);
  ASSERT_AND_ASSIGN(auto async_metadata, file->ReadMetadataAsync().result());
  EXPECT_EQ(async_metadata, nullptr);
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, AllocatingReadsClampKnownSizeBeforeAllocation) {
  for (const bool async : {false, true}) {
    SCOPED_TRACE(async);
    RejectingMemoryPool pool;
    const arrow::io::IOContext io_context(&pool);
    auto origin = std::make_shared<RecordingFileSystem>("origin", io_context);
    origin->input_file = std::make_shared<MetadataInputFile>(8192);
    ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
    arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
    info.set_size(8192);
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));

    const auto result =
        async ? file->ReadAsync(io_context, 4096, 64 * 1024 * 1024).result() : file->ReadAt(4096, 64 * 1024 * 1024);
    EXPECT_TRUE(result.status().IsOutOfMemory()) << result.status();
    EXPECT_EQ(pool.last_requested_size, 4096);
    ASSERT_STATUS_OK(file->Close());
  }
}

TEST_F(TalonFileSystemServiceFreeTest, AllocatingReadsAtKnownEofDoNotAllocatePayload) {
  for (const bool async : {false, true}) {
    for (const int64_t size : {0, 8192}) {
      SCOPED_TRACE(::testing::Message() << "async=" << async << ", size=" << size);
      arrow::ProxyMemoryPool pool(arrow::default_memory_pool());
      const arrow::io::IOContext io_context(&pool);
      auto origin = std::make_shared<RecordingFileSystem>("origin", io_context);
      origin->input_file = std::make_shared<MetadataInputFile>(size);
      ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
      arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
      info.set_size(size);
      ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));

      ASSERT_AND_ASSIGN(auto buffer, async ? file->ReadAsync(io_context, size, 64 * 1024 * 1024).result()
                                           : file->ReadAt(size, 64 * 1024 * 1024));
      EXPECT_EQ(buffer->size(), 0);
      EXPECT_EQ(pool.max_memory(), 0);
      ASSERT_STATUS_OK(file->Close());
    }
  }
}

TEST_F(TalonFileSystemServiceFreeTest, AllocatingReadsValidateBeforeAllocation) {
  for (const bool async : {false, true}) {
    SCOPED_TRACE(async);
    RejectingMemoryPool pool;
    const arrow::io::IOContext io_context(&pool);
    auto origin = std::make_shared<RecordingFileSystem>("origin", io_context);
    origin->input_file = std::make_shared<MetadataInputFile>(8192);
    ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
    arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
    info.set_size(8192);
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));

    const std::vector<std::tuple<int64_t, int64_t, arrow::StatusCode>> cases{
        {-1, 64 * 1024 * 1024, arrow::StatusCode::Invalid},
        {0, -1, arrow::StatusCode::Invalid},
        {8193, 64 * 1024 * 1024, arrow::StatusCode::IOError}};
    for (const auto& [position, nbytes, expected_code] : cases) {
      const auto result =
          async ? file->ReadAsync(io_context, position, nbytes).result() : file->ReadAt(position, nbytes);
      EXPECT_EQ(result.status().code(), expected_code);
    }
    ASSERT_STATUS_OK(file->Close());
    const auto closed =
        async ? file->ReadAsync(io_context, 0, 64 * 1024 * 1024).result() : file->ReadAt(0, 64 * 1024 * 1024);
    EXPECT_TRUE(closed.status().IsInvalid());
    EXPECT_EQ(pool.last_requested_size, -1);
  }
}

TEST_F(TalonFileSystemServiceFreeTest, OpenWithoutKnownSizeResolvesBeforeAllocatingReads) {
  for (const bool async : {false, true}) {
    SCOPED_TRACE(async);
    RejectingMemoryPool pool;
    const arrow::io::IOContext io_context(&pool);
    auto origin = std::make_shared<RecordingFileSystem>("origin", io_context);
    origin->input_file = std::make_shared<MetadataInputFile>(8192);
    ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));

    const auto result =
        async ? file->ReadAsync(io_context, 4096, 64 * 1024 * 1024).result() : file->ReadAt(4096, 64 * 1024 * 1024);
    EXPECT_TRUE(result.status().IsOutOfMemory()) << result.status();
    EXPECT_EQ(pool.last_requested_size, 4096);
    EXPECT_EQ(origin->input_open_calls, 1);
    ASSERT_STATUS_OK(file->Close());
  }
}

TEST_F(TalonFileSystemServiceFreeTest, OriginMetadataResolvesSizeWhenTalonUnavailable) {
  auto config = Config();
  // Size and ETag now come from origin metadata, without a Talon stat RPC.
  config.talon_coordinator = "127.0.0.1:0";
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  ASSERT_AND_ASSIGN(const auto size, file->GetSize());
  EXPECT_EQ(size, 7);
  EXPECT_EQ(origin->last_input_path, "test-bucket/prefix/object");
}

TEST_F(TalonFileSystemServiceFreeTest, TalonFallbackReusesOriginFileWithMetadata) {
  for (const bool metadata_first : {false, true}) {
    SCOPED_TRACE(metadata_first);
    auto config = Config();
    config.talon_coordinator = "127.0.0.1:0";
    auto origin = std::make_shared<RecordingFileSystem>("origin");
    ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
    if (metadata_first) {
      ASSERT_AND_ASSIGN(auto metadata, file->ReadMetadata());
      ASSERT_NE(metadata, nullptr);
    }
    ASSERT_AND_ASSIGN(const auto size, file->GetSize());
    EXPECT_EQ(size, 7);
    for (int i = 0; i < 3; ++i) {
      ASSERT_AND_ASSIGN(auto buffer, file->ReadAt(1, 3));
      EXPECT_EQ(buffer->ToString(), "ayl");
    }
    ASSERT_AND_ASSIGN(auto metadata, file->ReadMetadata());
    ASSERT_NE(metadata, nullptr);
    EXPECT_EQ(origin->input_open_calls, 1);
  }
}

TEST_F(TalonFileSystemServiceFreeTest, TalonOpenFailureDoesNotFallBack) {
  auto config = Config();
  // A local provider-mapping error must not be hidden by a usable origin.
  config.cloud_provider = "unsupported-by-talon";
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
  arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
  info.set_size(7);
  for (const bool known_size : {false, true}) {
    SCOPED_TRACE(known_size);
    origin->input_file = std::make_shared<MetadataInputFile>();
    const auto opens_before = origin->input_open_calls;
    const auto result = known_size ? fs->OpenInputFile(info) : fs->OpenInputFile("prefix/object");
    // A known size initializes the Talon reader immediately; unknown sizes
    // still defer initialization until the first metadata-dependent request.
    EXPECT_EQ(result.ok(), !known_size);
    const auto status = result.ok() ? result.ValueOrDie()->GetSize().status() : result.status();
    ASSERT_FALSE(status.ok());
    EXPECT_NE(status.message().find("Talon does not support cloud provider"), std::string::npos);
    EXPECT_EQ(origin->input_open_calls, opens_before + 1);
    if (result.ok()) {
      EXPECT_FALSE(result.ValueOrDie()->ReadAsync(0, 1).result().ok());
    }
  }
  ASSERT_AND_ASSIGN(const auto metrics, FindMetricsSource(fs, kTalonMetricsSource));
  ASSERT_NE(metrics, nullptr);
  EXPECT_EQ(metrics->GetReadCount(), 0);
  EXPECT_EQ(metrics->GetReadBytes(), 0);
  EXPECT_EQ(metrics->GetFailedCount(), 0);
}

TEST_F(TalonFileSystemServiceFreeTest, TalonReadFailureFallsBackForEveryReadApi) {
  auto config = Config();
  config.talon_coordinator = "127.0.0.1:0";
  for (const bool known_size : {false, true}) {
    SCOPED_TRACE(known_size);
    auto origin = std::make_shared<RecordingFileSystem>("origin");
    ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
    arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
    if (known_size) {
      info.set_size(7);
    }
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));
    ASSERT_AND_ASSIGN(auto buffer, file->ReadAt(1, 3));
    EXPECT_EQ(buffer->ToString(), "ayl");
    ASSERT_AND_ASSIGN(auto async_buffer, file->ReadAsync(4, 8).result());
    EXPECT_EQ(async_buffer->ToString(), "oad");

    auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
    ASSERT_NE(async_file, nullptr);
    uint8_t out[4] = {'x', 'x', 'x', 'x'};
    ASSERT_AND_ASSIGN(const auto bytes_read, async_file->ReadAtAsyncInto(2, 4, out).result());
    EXPECT_EQ(bytes_read, 4);
    EXPECT_EQ(std::string(reinterpret_cast<char*>(out), 4), "yloa");

    ASSERT_STATUS_OK(file->Seek(3));
    ASSERT_AND_ASSIGN(const auto sequential_bytes, file->Read(4, out));
    EXPECT_EQ(sequential_bytes, 4);
    EXPECT_EQ(std::string(reinterpret_cast<char*>(out), 4), "load");
    ASSERT_AND_ASSIGN(const auto position, file->Tell());
    EXPECT_EQ(position, 7);
    EXPECT_EQ(origin->last_input_path, "test-bucket/prefix/object");
    ASSERT_AND_ASSIGN(const auto metrics, FindMetricsSource(fs, kTalonMetricsSource));
    ASSERT_NE(metrics, nullptr);
    EXPECT_EQ(metrics->GetReadCount(), 4);
    EXPECT_EQ(metrics->GetReadBytes(), 0);
    EXPECT_EQ(metrics->GetFailedCount(), 4);
    ASSERT_AND_ASSIGN(const auto origin_metrics, FindMetricsSource(fs, kOriginMetricsSource));
    EXPECT_EQ(origin_metrics, origin->metrics);
  }
}

TEST_F(TalonFileSystemServiceFreeTest, MetricsAreSharedByReadersAndIsolatedByFilesystem) {
  auto config = Config();
  config.talon_coordinator = "127.0.0.1:0";
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
  ASSERT_AND_ASSIGN(auto other_fs, Wrap(config, std::make_shared<RecordingFileSystem>("other")));
  ASSERT_AND_ASSIGN(const auto metrics, FindMetricsSource(fs, kTalonMetricsSource));
  ASSERT_AND_ASSIGN(const auto other_metrics, FindMetricsSource(other_fs, kTalonMetricsSource));
  ASSERT_NE(metrics, nullptr);
  ASSERT_NE(other_metrics, nullptr);
  EXPECT_NE(metrics, other_metrics);
  std::vector<std::shared_ptr<arrow::io::RandomAccessFile>> files;
  for (int i = 0; i < 2; ++i) {
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
    files.push_back(std::move(file));
  }
  // The recording origin shares its input file; keep both wrappers open until all reads finish.
  for (const auto& file : files) {
    const std::vector<arrow::io::ReadRange> ranges = {{0, 2}, {2, 0}, {5, 8}};
    auto futures = file->ReadManyAsync({}, ranges);
    ASSERT_EQ(futures.size(), ranges.size());
    for (auto& future : futures) {
      ASSERT_STATUS_OK(future.result().status());
    }
  }
  EXPECT_EQ(metrics->GetReadCount(), 4);
  EXPECT_EQ(metrics->GetReadBytes(), 0);
  EXPECT_EQ(metrics->GetFailedCount(), 4);
  EXPECT_EQ(other_metrics->GetReadCount(), 0);
  EXPECT_EQ(other_metrics->GetFailedCount(), 0);
}

TEST_F(TalonFileSystemServiceFreeTest, MetricsSourcesFfiSeparatesOriginAndTalon) {
  auto config = Config();
  config.talon_coordinator = "127.0.0.1:0";
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  ASSERT_AND_ASSIGN(auto buffer, file->ReadAt(0, 3));
  // Distinct values verify that each source keeps its own snapshot.
  origin->metrics->IncrementReadBytes(17);
  FileSystemWrapper wrapper(fs);
  const auto handle = reinterpret_cast<FileSystemHandle>(&wrapper);
  LoonFilesystemMetricsSources sources = {};
  auto result = loon_filesystem_get_metrics_sources(handle, &sources);
  ASSERT_TRUE(loon_ffi_is_success(&result)) << loon_ffi_get_errmsg(&result);
  loon_ffi_free_result(&result);
  ASSERT_EQ(sources.count, 2);
  bool found_origin = false;
  bool found_talon = false;
  for (uint32_t i = 0; i < sources.count; ++i) {
    const auto& entry = sources.entries[i];
    EXPECT_EQ(entry.display_key, nullptr);
    if (std::strcmp(entry.source, kOriginMetricsSource) == 0) {
      found_origin = true;
      EXPECT_EQ(entry.metrics.read_count, 0);
      EXPECT_EQ(entry.metrics.read_bytes, 17);
    } else if (std::strcmp(entry.source, kTalonMetricsSource) == 0) {
      found_talon = true;
      EXPECT_EQ(entry.metrics.read_count, 1);
      EXPECT_EQ(entry.metrics.read_bytes, 0);
      EXPECT_EQ(entry.metrics.failed_count, 1);
    } else {
      ADD_FAILURE() << "Unexpected source: " << entry.source;
    }
  }
  EXPECT_TRUE(found_origin);
  EXPECT_TRUE(found_talon);
  loon_filesystem_free_metrics_sources(&sources);
}

TEST_F(TalonFileSystemServiceFreeTest, MetricsSourcesFfiListsAllSourcesAcrossCachedFilesystems) {
  auto& cache = FilesystemCache::getInstance();
  cache.clean();
  // Also clear the global cache when a fatal assertion exits this test early.
  const std::shared_ptr<FilesystemCache> cleanup(&cache, [](FilesystemCache* value) { value->clean(); });
  api::Properties properties;
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_STORAGE_TYPE, "remote"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_CLOUD_PROVIDER, kCloudProviderAWS), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_ADDRESS, "127.0.0.1:9000"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_BUCKET_NAME, "metrics-test-bucket"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_ACCESS_KEY_ID, "test-access-key"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_ACCESS_KEY_VALUE, "test-secret-key"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_REGION, "us-east-1"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_USE_SSL, "false"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_USE_IAM, "false"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_MODE, "0"), std::nullopt);
  ASSERT_AND_ASSIGN(auto origin_fs, cache.get(properties));
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_MODE, "1"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_COORDINATOR, "127.0.0.1:0"), std::nullopt);
  ASSERT_AND_ASSIGN(auto talon_fs, cache.get(properties));
  ASSERT_NE(origin_fs, talon_fs);
  ASSERT_EQ(cache.size(), 2);

  // Distinct values verify the flattened list preserves both grouping and snapshots.
  std::map<std::pair<std::string, std::string>, int64_t> expected;
  int64_t bytes = 1;
  for (const auto& [display_key, fs] : cache.list()) {
    const auto observable = std::dynamic_pointer_cast<Observable>(fs);
    ASSERT_NE(observable, nullptr);
    ASSERT_AND_ASSIGN(const auto sources, observable->GetMetricsSources());
    ASSERT_EQ(sources.size(), fs == talon_fs ? 2 : 1);
    for (const auto& [source, metrics] : sources) {
      ASSERT_NE(metrics, nullptr);
      metrics->IncrementReadBytes(bytes);
      EXPECT_TRUE(expected.emplace(std::make_pair(display_key, source), bytes).second);
      ++bytes;
    }
  }
  ASSERT_EQ(expected.size(), 3);
  LoonFilesystemMetricsSources sources = {};
  auto result = loon_filesystem_list_metrics_sources(&sources);
  const bool success = loon_ffi_is_success(&result);
  const std::string error = success ? "" : loon_ffi_get_errmsg(&result);
  loon_ffi_free_result(&result);
  ASSERT_TRUE(success) << error;
  // Snapshots and names must remain valid after the cache releases its references.
  cache.clean();
  EXPECT_EQ(sources.count, 3);
  for (uint32_t i = 0; i < sources.count; ++i) {
    const auto& entry = sources.entries[i];
    EXPECT_NE(entry.display_key, nullptr);
    EXPECT_NE(entry.source, nullptr);
    if (entry.display_key == nullptr || entry.source == nullptr) {
      continue;
    }
    const auto found = expected.find({entry.display_key, entry.source});
    EXPECT_NE(found, expected.end());
    if (found != expected.end()) {
      EXPECT_EQ(entry.metrics.read_bytes, found->second);
      expected.erase(found);
    }
  }
  EXPECT_TRUE(expected.empty());
  loon_filesystem_free_metrics_sources(&sources);
}

TEST_F(TalonFileSystemServiceFreeTest, TalonFallbackPreservesOriginErrorDetails) {
  auto config = Config();
  config.talon_coordinator = "127.0.0.1:0";
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  const auto error = MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "origin denied", "provider detail");
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  ASSERT_AND_ASSIGN(const auto size, file->GetSize());
  EXPECT_EQ(size, 7);
  origin->input_file->read_status = error;
  uint8_t out = 0;
  for (const auto& status : {file->ReadAt(0, 1, &out).status(), file->ReadAsync(0, 1).result().status()}) {
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), error.code());
    EXPECT_EQ(status.detail(), error.detail());
    EXPECT_NE(status.message().find("Talon"), std::string::npos);
    EXPECT_NE(status.message().find("origin denied"), std::string::npos);
  }
  ASSERT_AND_ASSIGN(const auto metrics, FindMetricsSource(fs, kTalonMetricsSource));
  ASSERT_NE(metrics, nullptr);
  EXPECT_EQ(metrics->GetReadCount(), 2);
  EXPECT_EQ(metrics->GetReadBytes(), 0);
  EXPECT_EQ(metrics->GetFailedCount(), 2);
}

TEST_F(TalonFileSystemServiceFreeTest, TalonFallbackUsesNativeAsyncOrigin) {
  for (const bool allocating : {false, true}) {
    SCOPED_TRACE(allocating);
    auto config = Config();
    config.talon_coordinator = "127.0.0.1:0";
    auto origin = std::make_shared<RecordingFileSystem>("origin");
    auto origin_file = std::make_shared<AsyncOriginInputFile>();
    origin->fallback_file = origin_file;
    ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
    auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
    ASSERT_NE(async_file, nullptr);
    uint8_t out[3] = {};
    auto future = allocating ? file->ReadAsync(1, 3).Then([](const std::shared_ptr<arrow::Buffer>& buffer) {
      EXPECT_EQ(buffer->ToString(), "ayl");
      return buffer->size();
    })
                             : async_file->ReadAtAsyncInto(1, 3, out);
    EXPECT_TRUE(origin_file->started.Wait(5.0));
    EXPECT_FALSE(future.is_finished());
    origin_file->completion.MarkFinished(3);
    ASSERT_AND_ASSIGN(const auto result, future.result());
    EXPECT_EQ(result, 3);
    if (!allocating) {
      EXPECT_EQ(std::string(reinterpret_cast<char*>(out), 3), "ayl");
    }
    ASSERT_STATUS_OK(file->Close());
    EXPECT_TRUE(origin_file->closed());
  }
}

TEST_F(TalonFileSystemServiceFreeTest, ConcurrentTalonFallbackReadsReuseOriginFile) {
  auto config = Config();
  config.talon_coordinator = "127.0.0.1:0";
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  ASSERT_AND_ASSIGN(auto metadata, file->ReadMetadata());
  ASSERT_NE(metadata, nullptr);
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);
  std::vector<std::vector<uint8_t>> outputs(8, std::vector<uint8_t>(3));
  std::vector<arrow::Future<int64_t>> futures;
  for (auto& out : outputs) {
    futures.push_back(async_file->ReadAtAsyncInto(2, 3, out.data()));
  }
  for (const auto& future : futures) {
    future.Wait();
  }
  for (size_t i = 0; i < futures.size(); ++i) {
    ASSERT_AND_ASSIGN(const auto bytes_read, futures[i].result());
    EXPECT_EQ(bytes_read, 3);
    EXPECT_EQ(std::string(outputs[i].begin(), outputs[i].end()), "ylo");
  }
  EXPECT_EQ(origin->input_open_calls, 1);
  ASSERT_STATUS_OK(file->Close());
  EXPECT_TRUE(origin->input_file->closed());
}

TEST_F(TalonFileSystemServiceFreeTest, TalonFallbackPreservesAsyncOriginErrors) {
  for (const bool allocating : {false, true}) {
    SCOPED_TRACE(allocating);
    auto config = Config();
    config.talon_coordinator = "127.0.0.1:0";
    auto origin = std::make_shared<RecordingFileSystem>("origin");
    auto origin_file = std::make_shared<AsyncOriginInputFile>();
    origin->fallback_file = origin_file;
    ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
    auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
    ASSERT_NE(async_file, nullptr);
    uint8_t out = 0;
    auto future =
        allocating
            ? file->ReadAsync(0, 1).Then([](const std::shared_ptr<arrow::Buffer>& buffer) { return buffer->size(); })
            : async_file->ReadAtAsyncInto(0, 1, &out);
    EXPECT_TRUE(origin_file->started.Wait(5.0));
    const auto error = MakeExtendError(ExtendStatusCode::StorageTransientNetwork, "origin I/O failed", "detail");
    origin_file->completion.MarkFinished(arrow::Result<int64_t>(error));
    const auto& result = future.result();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), error.code());
    EXPECT_EQ(result.status().detail(), error.detail());
    EXPECT_NE(result.status().message().find("Talon"), std::string::npos);
    EXPECT_NE(result.status().message().find("origin I/O failed"), std::string::npos);
  }
}

TEST_F(TalonFileSystemServiceFreeTest, InvalidReadsAndKnownEofDoNotFallBack) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
  info.set_size(7);
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);
  uint8_t out = 0;
  EXPECT_TRUE(async_file->ReadAtAsyncInto(-1, 1, &out).status().IsInvalid());
  EXPECT_TRUE(async_file->ReadAtAsyncInto(0, -1, &out).status().IsInvalid());
  EXPECT_TRUE(async_file->ReadAtAsyncInto(0, 1, nullptr).status().IsInvalid());
  EXPECT_TRUE(async_file->ReadAtAsyncInto(8, 1, &out).status().IsIOError());
  ASSERT_AND_ASSIGN(const auto bytes_read, async_file->ReadAtAsyncInto(7, 1, &out).result());
  EXPECT_EQ(bytes_read, 0);
  ASSERT_STATUS_OK(file->Close());
  EXPECT_TRUE(async_file->ReadAtAsyncInto(0, 1, &out).status().IsInvalid());
  EXPECT_EQ(origin->input_open_calls, 1);
  EXPECT_EQ(origin->input_file->read_calls.load(), 0);
  ASSERT_AND_ASSIGN(const auto metrics, FindMetricsSource(fs, kTalonMetricsSource));
  ASSERT_NE(metrics, nullptr);
  EXPECT_EQ(metrics->GetReadCount(), 0);
  EXPECT_EQ(metrics->GetReadBytes(), 0);
  EXPECT_EQ(metrics->GetFailedCount(), 0);
}

TEST_F(TalonIntegrationTest, WriteOriginReadThroughTalon) {
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path_));
  ASSERT_AND_ASSIGN(auto buffer, file->ReadAt(0, static_cast<int64_t>(expected_.size())));
  ASSERT_EQ(buffer->size(), static_cast<int64_t>(expected_.size()));
  EXPECT_EQ(std::memcmp(buffer->data(), expected_.data(), expected_.size()), 0);
}

TEST_F(TalonIntegrationTest, FallsBackToConfiguredOriginWhenTalonUnavailable) {
  // Run with S3/MinIO or Azure/Azurite through TEST_ENV_CLOUD_PROVIDER.
  // This test requires only the origin service: Talon must fail deterministically.
  // Set TEST_ENV_TALON_ENABLED=true and TEST_ENV_TALON_COORDINATOR=127.0.0.1:0.
  // The AWS configuration exercises both the SDK and CRT origin read paths.
  ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties_));
  const auto crt_modes =
      config.cloud_provider == kCloudProviderAWS ? std::vector<bool>{false, true} : std::vector<bool>{false};
  for (const bool use_crt : crt_modes) {
    SCOPED_TRACE(::testing::Message() << "provider=" << config.cloud_provider << ", CRT=" << use_crt);
    auto properties = properties_;
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_COORDINATOR, "127.0.0.1:0"), std::nullopt);
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_S3_CRT_ASYNC_READ, use_crt ? "true" : "false"), std::nullopt);
    ASSERT_AND_ASSIGN(auto fs, GetFileSystem(properties));
    ASSERT_AND_ASSIGN(const auto talon_metrics, FindMetricsSource(fs, kTalonMetricsSource));
    ASSERT_AND_ASSIGN(const auto origin_metrics, FindMetricsSource(fs, kOriginMetricsSource));
    ASSERT_NE(talon_metrics, nullptr);
    ASSERT_NE(origin_metrics, nullptr);
    const auto talon_before = talon_metrics->GetSnapshot();
    const auto origin_before = origin_metrics->GetSnapshot();
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(path_));
    auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
    ASSERT_NE(async_file, nullptr);

    ASSERT_AND_ASSIGN(const auto size, async_file->GetSizeAsync().result());
    EXPECT_EQ(size, static_cast<int64_t>(expected_.size()));
    constexpr int64_t kOffset = 1024;
    constexpr int64_t kLength = 4096;
    ASSERT_AND_ASSIGN(auto buffer, file->ReadAt(kOffset, kLength));
    ASSERT_EQ(buffer->size(), kLength);
    EXPECT_EQ(std::memcmp(buffer->data(), expected_.data() + kOffset, kLength), 0);

    std::vector<uint8_t> out(kLength);
    ASSERT_AND_ASSIGN(const auto bytes_read, async_file->ReadAtAsyncInto(kOffset, kLength, out.data()).result());
    EXPECT_EQ(bytes_read, kLength);
    EXPECT_TRUE(std::equal(out.begin(), out.end(), expected_.begin() + kOffset));

    ASSERT_AND_ASSIGN(auto tail, file->ReadAsync(size - 5, 16).result());
    ASSERT_EQ(tail->size(), 5);
    EXPECT_EQ(std::memcmp(tail->data(), expected_.data() + expected_.size() - 5, 5), 0);
    ASSERT_AND_ASSIGN(auto metadata, file->ReadMetadata());
    ASSERT_NE(metadata, nullptr);
    ASSERT_STATUS_OK(file->Close());
    EXPECT_EQ(talon_metrics->GetReadCount() - talon_before.read_count, 3);
    EXPECT_EQ(talon_metrics->GetFailedCount() - talon_before.failed_count, 3);
    EXPECT_EQ(talon_metrics->GetReadBytes() - talon_before.read_bytes, 0);
    EXPECT_EQ(origin_metrics->GetReadBytes() - origin_before.read_bytes, 2 * kLength + 5);
  }
}

TEST_P(TalonCloudMetadataTest, ReadsETagAndReusesMetadataWithoutOriginRequests) {
  const std::string content = "Talon ETag and size cache validation";
  ASSERT_AND_ASSIGN(auto output, origin_fs_->OpenOutputStream(path_));
  const auto write_status = output->Write(content);
  ASSERT_STATUS_OK(write_status);
  const auto close_status = output->Close();
  ASSERT_STATUS_OK(close_status);

  ASSERT_AND_ASSIGN(auto origin_file, origin_fs_->OpenInputFile(path_));
  ASSERT_AND_ASSIGN(const auto metadata, origin_file->ReadMetadata());
  ASSERT_NE(metadata, nullptr);
  ASSERT_AND_ASSIGN(const auto etag, metadata->Get("ETag"));
  ASSERT_FALSE(etag.empty());
  RecordProperty("etag", etag);
  ASSERT_AND_ASSIGN(const auto size, origin_file->GetSize());
  ASSERT_EQ(size, content.size());
  ASSERT_AND_ASSIGN(const auto data, origin_file->ReadAt(0, size));
  ASSERT_EQ(data->ToString(), content);

  auto talon_config = config_;
  talon_config.talon_mode = TalonMode::Full;
  // Opening a versioned reader must not require a coordinator RPC.
  talon_config.talon_coordinator = "127.0.0.1:1";
  ASSERT_AND_ASSIGN(auto talon_fs, CreateArrowFileSystem(talon_config));
  arrow::fs::FileInfo info(path_, arrow::fs::FileType::File);
  ASSERT_AND_ASSIGN(auto talon_file, talon_fs->OpenInputFile(info));
  ASSERT_AND_ASSIGN(const auto talon_size, talon_file->GetSize());
  EXPECT_EQ(talon_size, size);
  ASSERT_AND_ASSIGN(const auto talon_metadata, talon_file->ReadMetadata());
  ASSERT_AND_ASSIGN(const auto talon_etag, talon_metadata->Get("ETag"));
  EXPECT_EQ(talon_etag, etag);

  // Once the test object is gone, another HEAD would fail. Success below
  // therefore proves both the provider file cache and the filesystem cache.
  ASSERT_STATUS_OK(origin_fs_->DeleteFile(path_));
  ASSERT_AND_ASSIGN(auto uncached_file, origin_fs_->OpenInputFile(path_));
  ASSERT_FALSE(uncached_file->ReadMetadata().ok());
  ASSERT_STATUS_OK(uncached_file->Close());
  ASSERT_AND_ASSIGN(const auto cached_size, origin_file->GetSize());
  EXPECT_EQ(cached_size, size);
  ASSERT_AND_ASSIGN(const auto cached_metadata, origin_file->ReadMetadata());
  EXPECT_EQ(cached_metadata, metadata);
  ASSERT_STATUS_OK(origin_file->Close());
  ASSERT_STATUS_OK(talon_file->Close());

  ASSERT_AND_ASSIGN(auto reopened, talon_fs->OpenInputFile(path_));
  ASSERT_AND_ASSIGN(const auto reopened_size, reopened->GetSize());
  EXPECT_EQ(reopened_size, size);
  ASSERT_STATUS_OK(reopened->Close());
  RecordProperty("deleted_object_cache_checks", "passed");
}

INSTANTIATE_TEST_SUITE_P(CloudReaders, TalonCloudMetadataTest, ::testing::Bool());

TEST_F(TalonIntegrationTest, EvictionReadsReplacementAtSamePath) {
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path_));
  ASSERT_AND_ASSIGN(auto warmed, file->ReadAt(0, static_cast<int64_t>(expected_.size())));
  ASSERT_EQ(warmed->size(), expected_.size());
  ASSERT_EQ(std::memcmp(warmed->data(), expected_.data(), expected_.size()), 0);
  ASSERT_STATUS_OK(file->Close());

  // Model final collection release, followed by replacing the external object.
  // Keep the size unchanged: a size-only check must not hide a stale ETag.
  auto& cache = FilesystemCache::getInstance();
  ASSERT_STATUS_OK(cache.evict(properties_));
  auto replacement = expected_;
  for (auto& byte : replacement) {
    byte ^= 0xff;
  }
  ASSERT_AND_ASSIGN(auto output, fs_->OpenOutputStream(path_));
  ASSERT_STATUS_OK(output->Write(replacement.data(), static_cast<int64_t>(replacement.size())));
  ASSERT_STATUS_OK(output->Close());

  // The retained old filesystem still reads the warmed old-version blocks.
  // This also proves the test is exercising Talon, not just origin fallback.
  ASSERT_AND_ASSIGN(auto stale_reader, fs_->OpenInputFile(path_));
  ASSERT_AND_ASSIGN(auto stale, stale_reader->ReadAt(0, static_cast<int64_t>(expected_.size())));
  ASSERT_EQ(stale->size(), expected_.size());
  ASSERT_EQ(std::memcmp(stale->data(), expected_.data(), expected_.size()), 0);
  ASSERT_STATUS_OK(stale_reader->Close());

  ASSERT_AND_ASSIGN(auto recreated_fs, cache.get(properties_));
  ASSERT_NE(recreated_fs, fs_);
  ASSERT_AND_ASSIGN(auto reopened, recreated_fs->OpenInputFile(path_));
  ASSERT_AND_ASSIGN(auto actual, reopened->ReadAt(0, static_cast<int64_t>(replacement.size())));
  ASSERT_EQ(actual->size(), replacement.size());
  EXPECT_EQ(std::memcmp(actual->data(), replacement.data(), replacement.size()), 0);
  ASSERT_STATUS_OK(reopened->Close());
}

TEST_F(TalonIntegrationTest, RawPrewarmThenNormalReader) {
  ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties_));
  const std::string scheme = config.cloud_provider == kCloudProviderAzure ? "az" : "s3";
  const std::string uri = scheme + "://" + config.bucket_name + "/" + path_;
  RecordProperty("raw_uri", uri);
  RecordProperty("object_path", path_);
  ASSERT_AND_ASSIGN(const auto client, talon::TalonClient::Make(config));
  RecordProperty("prewarm_start_unix_ns", std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                             std::chrono::system_clock::now().time_since_epoch())
                                                             .count()));
  // Normal known-size readers use the unchecked cache identity, so prewarm must match it.
  ASSERT_AND_ASSIGN(const auto load, client->Load({uri, expected_.size(), ""}));
  ASSERT_EQ(load.size, expected_.size());
  RecordProperty("prewarm_end_unix_ns", std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                           std::chrono::system_clock::now().time_since_epoch())
                                                           .count()));
  ASSERT_AND_ASSIGN(const auto metrics, FindMetricsSource(fs_, kTalonMetricsSource));
  ASSERT_NE(metrics, nullptr);
  const auto before = metrics->GetSnapshot();
  RecordProperty("read_phase_start_unix_ns", std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                std::chrono::system_clock::now().time_since_epoch())
                                                                .count()));
  arrow::fs::FileInfo info(path_, arrow::fs::FileType::File);
  info.set_size(static_cast<int64_t>(expected_.size()));
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(info));
  ASSERT_AND_ASSIGN(const auto data, file->ReadAt(0, info.size()));
  ASSERT_EQ(data->size(), info.size());
  EXPECT_EQ(std::memcmp(data->data(), expected_.data(), expected_.size()), 0);
  EXPECT_EQ(metrics->GetReadCount() - before.read_count, 1);
  EXPECT_EQ(metrics->GetReadBytes() - before.read_bytes, expected_.size());
  EXPECT_EQ(metrics->GetFailedCount() - before.failed_count, 0);
  ASSERT_STATUS_OK(file->Close());
  RecordProperty("read_phase_end_unix_ns", std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                              std::chrono::system_clock::now().time_since_epoch())
                                                              .count()));
}

TEST_F(TalonIntegrationTest, RawPrewarmReadAndPartialFailure) {
  ASSERT_AND_ASSIGN(const auto config, GetFileSystemConfig(properties_));
  const std::string scheme = config.cloud_provider == kCloudProviderAzure ? "az" : "s3";
  const std::string uri = scheme + "://" + config.bucket_name + "/" + path_;
  ASSERT_AND_ASSIGN(auto client, talon::TalonClient::Make(config));
  ASSERT_AND_ASSIGN(const auto stat, client->Stat(uri));
  EXPECT_EQ(stat.size, expected_.size());
  ASSERT_FALSE(stat.version.empty());
  ASSERT_AND_ASSIGN(const auto load, client->Load({uri, stat.size, stat.version}));
  EXPECT_EQ(load.size, expected_.size());
  EXPECT_EQ(load.blocks, 1);
  ASSERT_AND_ASSIGN(const auto unchecked, client->Load({uri, stat.size, ""}));
  EXPECT_EQ(unchecked.size, expected_.size());
  std::vector<uint8_t> output(expected_.size());
  ASSERT_AND_ASSIGN(const auto read, client->Read(uri, 0, output.size(), output.data(), stat));
  EXPECT_EQ(read, expected_.size());
  EXPECT_EQ(output, expected_);

  const auto failed = client->Load({uri, stat.size, "missing-version"});
  ASSERT_FALSE(failed.ok());
  const auto failed_detail = ExtendStatusDetail::UnwrapStatus(failed.status());
  ASSERT_NE(failed_detail, nullptr) << failed.status();
  EXPECT_EQ(failed_detail->code(), ExtendStatusCode::AwsErrorNonRetryable);
  EXPECT_NE(failed.status().message().find("worker"), std::string::npos) << failed.status();
  EXPECT_NE(failed.status().message().find("offset"), std::string::npos) << failed.status();
  ASSERT_AND_ASSIGN(const auto partial, client->BatchLoad({{uri, stat.size, stat.version},
                                                           {uri, stat.size, "missing-version"},
                                                           {uri, 0, ""},
                                                           {uri, stat.size, "another-missing-version"}}));
  EXPECT_FALSE(partial.status.ok());
  const auto partial_detail = ExtendStatusDetail::UnwrapStatus(partial.status);
  ASSERT_NE(partial_detail, nullptr) << partial.status;
  EXPECT_EQ(partial_detail->code(), ExtendStatusCode::AwsErrorNonRetryable);
  EXPECT_NE(partial.status.message().find("batch load incomplete for 2 file(s):"), std::string::npos) << partial.status;
  EXPECT_NE(partial.status.message().find("worker"), std::string::npos) << partial.status;
  EXPECT_TRUE(partial.results.empty());
  ASSERT_EQ(partial.failed_files.size(), 2);
  EXPECT_EQ(partial.failed_files[0].index, 1);
  EXPECT_EQ(partial.failed_files[1].index, 3);
  EXPECT_FALSE(partial.failed_files[0].uncertain);
  EXPECT_FALSE(partial.failed_files[1].uncertain);
  EXPECT_FALSE(partial.failed_files[0].error.empty());
  EXPECT_FALSE(partial.failed_files[1].error.empty());

  // Requests and client ownership can disappear immediately after submission.
  arrow::Future<talon::TalonBatchLoadResult> batch;
  {
    std::vector<talon::TalonLoadRequest> requests{
        {uri, stat.size, stat.version}, {uri, 0, ""}, {uri, stat.size, stat.version}};
    batch = client->BatchLoadAsync(requests);
  }
  auto read_future = client->ReadAsync(uri, 4096, 8192, output.data());
  client.reset();
  // Drain both operations before assertions release the caller-owned buffer.
  read_future.Wait();
  batch.Wait();
  ASSERT_AND_ASSIGN(const auto copied_batch, batch.result());
  ASSERT_TRUE(copied_batch.status.ok()) << copied_batch.status;
  ASSERT_EQ(copied_batch.results.size(), 3);
  EXPECT_EQ(copied_batch.results[0].size, expected_.size());
  EXPECT_EQ(copied_batch.results[1].size, 0);
  EXPECT_EQ(copied_batch.results[1].blocks, 0);
  EXPECT_EQ(copied_batch.results[2].size, expected_.size());
  ASSERT_AND_ASSIGN(const auto copied_read, read_future.result());
  EXPECT_EQ(copied_read, 8192);
  EXPECT_TRUE(std::equal(output.begin(), output.begin() + 8192, expected_.begin() + 4096));
}

TEST_F(TalonIntegrationTest, OpenWithKnownSize) {
  ASSERT_AND_ASSIGN(const auto metrics, FindMetricsSource(fs_, kTalonMetricsSource));
  ASSERT_NE(metrics, nullptr);
  const auto before = metrics->GetSnapshot();
  // Supply the writer-known size so this test does not issue a HEAD itself.
  arrow::fs::FileInfo info(path_, arrow::fs::FileType::File);
  info.set_size(static_cast<int64_t>(expected_.size()));
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(info));
  ASSERT_AND_ASSIGN(const auto size, file->GetSize());
  EXPECT_EQ(size, static_cast<int64_t>(expected_.size()));
  ASSERT_AND_ASSIGN(auto buffer, file->ReadAt(0, static_cast<int64_t>(expected_.size())));
  ASSERT_EQ(buffer->size(), static_cast<int64_t>(expected_.size()));
  EXPECT_EQ(std::memcmp(buffer->data(), expected_.data(), expected_.size()), 0);
  EXPECT_EQ(metrics->GetReadCount() - before.read_count, 1);
  EXPECT_EQ(metrics->GetReadBytes() - before.read_bytes, static_cast<int64_t>(expected_.size()));
  EXPECT_EQ(metrics->GetFailedCount() - before.failed_count, 0);
  ASSERT_AND_ASSIGN(auto async_buffer, file->ReadAsync(info.size() - 3, 10).result());
  EXPECT_EQ(async_buffer->size(), 3);
  EXPECT_EQ(metrics->GetReadCount() - before.read_count, 2);
  EXPECT_EQ(metrics->GetReadBytes() - before.read_bytes, static_cast<int64_t>(expected_.size()) + 3);
  EXPECT_EQ(metrics->GetFailedCount() - before.failed_count, 0);
}

TEST_F(TalonIntegrationTest, OpenWithoutKnownSize) {
  ASSERT_AND_ASSIGN(const auto metrics, FindMetricsSource(fs_, kTalonMetricsSource));
  ASSERT_NE(metrics, nullptr);
  const auto before = metrics->GetSnapshot();
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path_));
  ASSERT_AND_ASSIGN(const auto size, file->GetSize());
  ASSERT_EQ(size, static_cast<int64_t>(expected_.size()));

  constexpr int64_t kOffset = 4096;
  constexpr int64_t kLength = 4096;
  ASSERT_AND_ASSIGN(auto buffer, file->ReadAt(kOffset, kLength));
  ASSERT_EQ(buffer->size(), kLength);
  EXPECT_EQ(std::memcmp(buffer->data(), expected_.data() + kOffset, kLength), 0);
  EXPECT_EQ(metrics->GetReadCount() - before.read_count, 1);
  EXPECT_EQ(metrics->GetReadBytes() - before.read_bytes, kLength);
  EXPECT_EQ(metrics->GetFailedCount() - before.failed_count, 0);
}

TEST_F(TalonIntegrationTest, AsyncReadAtIntoCallerBuffer) {
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path_));
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);

  constexpr int64_t kOffset = 1024;
  std::vector<uint8_t> output(4096);
  auto future = async_file->ReadAtAsyncInto(kOffset, static_cast<int64_t>(output.size()), output.data());
  ASSERT_AND_ASSIGN(const auto bytes_read, future.result());
  ASSERT_EQ(bytes_read, static_cast<int64_t>(output.size()));
  EXPECT_TRUE(std::equal(output.begin(), output.end(), expected_.begin() + kOffset));
}

TEST_F(TalonIntegrationTest, ConcurrentAsyncRanges) {
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path_));
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);

  constexpr size_t kRangeCount = 16;
  constexpr size_t kRangeSize = 64 * 1024;
  std::vector<std::vector<uint8_t>> outputs(kRangeCount, std::vector<uint8_t>(kRangeSize));
  std::vector<arrow::Future<int64_t>> futures;
  futures.reserve(kRangeCount);
  for (size_t i = 0; i < kRangeCount; ++i) {
    futures.push_back(async_file->ReadAtAsyncInto(static_cast<int64_t>(i * kRangeSize),
                                                  static_cast<int64_t>(kRangeSize), outputs[i].data()));
  }

  // Drain all reads before a fatal assertion can release buffers still used by Rust.
  // Each future retains its result for the assertions below, including failures.
  for (const auto& future : futures) {
    future.Wait();
  }

  for (size_t i = 0; i < kRangeCount; ++i) {
    ASSERT_AND_ASSIGN(const auto bytes_read, futures[i].result());
    ASSERT_EQ(bytes_read, static_cast<int64_t>(kRangeSize));
    EXPECT_TRUE(std::equal(outputs[i].begin(), outputs[i].end(), expected_.begin() + i * kRangeSize));
  }
}

TEST_F(TalonIntegrationTest, EofAndClosedFileSemantics) {
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path_));
  std::vector<uint8_t> output(4096);
  ASSERT_AND_ASSIGN(const auto bytes_read, file->ReadAt(static_cast<int64_t>(expected_.size()),
                                                        static_cast<int64_t>(output.size()), output.data()));
  EXPECT_EQ(bytes_read, 0);

  ASSERT_STATUS_OK(file->Close());
  const auto closed_read = file->ReadAt(0, static_cast<int64_t>(output.size()), output.data());
  ASSERT_FALSE(closed_read.ok());
  EXPECT_TRUE(closed_read.status().IsInvalid()) << closed_read.status().ToString();
}

TEST_F(TalonIntegrationTest, ParquetReadThroughNormalStorageApi) {
  ASSERT_STATUS_OK(CreateTestDir(fs_, parquet_path_));
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto batch, CreateTestData(schema));
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema));

  auto writer = api::Writer::create(parquet_path_, schema, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);
  ASSERT_STATUS_OK(writer->write(batch));
  ASSERT_AND_ASSIGN(auto column_groups, writer->close());

  auto reader = api::Reader::create(column_groups, schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto batch_reader, reader->get_record_batch_reader());
  std::shared_ptr<arrow::RecordBatch> result;
  ASSERT_STATUS_OK(batch_reader->ReadNext(&result));
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->num_rows(), batch->num_rows());
  ASSERT_STATUS_OK(ValidateRowAlignment(result));
  ASSERT_STATUS_OK(batch_reader->ReadNext(&result));
  EXPECT_EQ(result, nullptr);
}

}  // namespace milvus_storage::test
