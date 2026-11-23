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

// GCP filesystems on one (endpoint, bucket) keep their own identities: each
// SDK client is bound to the provider of the Make() call that built it and
// attaches that provider's token itself, the same provider authenticates its
// CRT and native transport requests, and a filesystem is built even when
// another identity is registered for its bucket.

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <arrow/filesystem/filesystem.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <aws/core/Aws.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/URI.h>
#include <aws/core/http/curl/CurlHttpClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/gcp/gcp_credential_provider.h"
#include "milvus-storage/filesystem/gcp/gcp_credential_registry.h"
#include "milvus-storage/filesystem/gcp/gcp_filesystem_producer.h"

namespace milvus_storage {

namespace {

namespace http = boost::beast::http;
using Tcp = boost::asio::ip::tcp;

// An empty token stands for a provider that cannot get one.
class StaticBearerProvider final : public GcpCredentialProvider {
  public:
  explicit StaticBearerProvider(std::string token, std::string identity = "")
      : token_(std::move(token)), identity_(identity.empty() ? "static:" + token_ : std::move(identity)) {}

  std::optional<std::pair<std::string, std::string>> AuthorizationHeader() override {
    ++calls_;
    if (token_.empty()) {
      return std::nullopt;
    }
    return std::make_pair(std::string("Authorization"), "Bearer " + token_);
  }

  arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>&) override {
    return arrow::Status::OK();
  }

  bool IssuesBearerToken() const override { return true; }

  std::string Identity() const override { return identity_; }

  // How often a request asked this provider for a header.
  int calls() const { return calls_; }

  private:
  std::string token_;
  std::string identity_;
  std::atomic<int> calls_{0};
};

// HMAC-style provider: the AWS SDK signs with the config's access keys.
class StaticHmacProvider final : public GcpCredentialProvider {
  public:
  std::optional<std::pair<std::string, std::string>> AuthorizationHeader() override { return std::nullopt; }

  arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>&) override {
    ++calls_;
    return arrow::Status::OK();
  }

  bool IssuesBearerToken() const override { return false; }

  // How often a request reached this provider's GOOG4 re-sign step.
  int calls() const { return calls_; }

  std::string Identity() const override { return "hmac:binding-test"; }

  private:
  std::atomic<int> calls_{0};
};

// Serves a 9-byte object to HEAD and ranged GET requests and records each
// request's method and Authorization header in arrival order.
class RecordingServer {
  public:
  RecordingServer()
      : acceptor_(context_, {boost::asio::ip::address_v4::loopback(), 0}),
        port_(acceptor_.local_endpoint().port()),
        thread_([this] { Serve(); }) {}

  ~RecordingServer() {
    stopping_ = true;
    // Unblock the pending accept.
    boost::asio::io_context wake_context;
    Tcp::socket wake(wake_context);
    boost::system::error_code ignored;
    wake.connect({boost::asio::ip::address_v4::loopback(), port_}, ignored);
    thread_.join();
  }

  RecordingServer(const RecordingServer&) = delete;
  RecordingServer& operator=(const RecordingServer&) = delete;

  uint16_t port() const { return port_; }

  std::vector<std::string> Authorizations() const {
    std::lock_guard<std::mutex> lock(mu_);
    return authorizations_;
  }

  std::vector<std::string> Methods() const {
    std::lock_guard<std::mutex> lock(mu_);
    return methods_;
  }

  private:
  void Serve() {
    while (!stopping_) {
      Tcp::socket socket(context_);
      boost::system::error_code ec;
      acceptor_.accept(socket, ec);
      if (ec || stopping_) {
        return;
      }
      boost::beast::flat_buffer buffer;
      http::request<http::empty_body> request;
      http::read(socket, buffer, request, ec);
      if (ec) {
        continue;
      }
      const auto authorization = request[http::field::authorization];
      {
        std::lock_guard<std::mutex> lock(mu_);
        authorizations_.emplace_back(authorization.data(), authorization.size());
        methods_.emplace_back(request.method_string());
      }
      if (request.method() == http::verb::get) {
        WriteRange(socket, request);
      } else {
        http::response<http::empty_body> response{http::status::ok, request.version()};
        response.content_length(kObject.size());
        response.set(http::field::etag, kEtag);
        response.set(http::field::last_modified, "Wed, 21 Oct 2015 07:28:00 GMT");
        response.keep_alive(false);
        http::write(socket, response, ec);
      }
      socket.shutdown(Tcp::socket::shutdown_both, ec);
    }
  }

  static constexpr std::string_view kObject = "abcdefghi";
  static constexpr const char* kEtag = "\"8aa99b1f439ff71293e95357bac6fd94\"";

  // Answers "Range: bytes=<first>-<last>" with that slice of the object.
  static void WriteRange(Tcp::socket& socket, const http::request<http::empty_body>& request) {
    const std::string range(request[http::field::range]);
    size_t first = 0;
    size_t last = kObject.size() - 1;
    if (range.starts_with("bytes=")) {
      const auto dash = range.find('-');
      first = std::stoul(range.substr(6, dash - 6));
      last = std::min(last, static_cast<size_t>(std::stoul(range.substr(dash + 1))));
    }
    http::response<http::string_body> response{http::status::partial_content, request.version()};
    response.set(http::field::content_range,
                 "bytes " + std::to_string(first) + "-" + std::to_string(last) + "/" + std::to_string(kObject.size()));
    response.set(http::field::accept_ranges, "bytes");
    response.set(http::field::etag, kEtag);
    response.body() = std::string(kObject.substr(first, last - first + 1));
    response.prepare_payload();
    response.keep_alive(false);
    boost::system::error_code ec;
    http::write(socket, response, ec);
  }

  boost::asio::io_context context_;
  Tcp::acceptor acceptor_;
  uint16_t port_;
  std::atomic<bool> stopping_{false};
  mutable std::mutex mu_;
  std::vector<std::string> authorizations_;
  std::vector<std::string> methods_;
  // Last member: the thread starts after everything it uses.
  std::thread thread_;
};

ArrowFileSystemConfig LocalGcsConfig(uint16_t port, const std::string& bucket, const std::string& alias) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderGCP;
  config.address = "127.0.0.1:" + std::to_string(port);
  config.bucket_name = bucket;
  config.region = "auto";
  config.use_ssl = false;
  config.s3_use_crt = false;
  config.alias = alias;
  return config;
}

constexpr const char* kForeignFactoryTag = "RegistryLookupHttpClientFactory";

// Stands in for an embedder that initializes the AWS SDK with its own GCP
// factory before milvus-storage, as Milvus does on a GCP QueryNode: plain
// curl clients, and a Bearer header looked up in GcpCredentialRegistry by
// request URI.
class RegistryLookupHttpClientFactory final : public Aws::Http::HttpClientFactory {
  public:
  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& config) const override {
    return Aws::MakeShared<Aws::Http::CurlHttpClient>(kForeignFactoryTag, config);
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::String& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& stream_factory) const override {
    return CreateHttpRequest(Aws::Http::URI(uri), method, stream_factory);
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::Http::URI& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& stream_factory) const override {
    auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(kForeignFactoryTag, uri, method);
    request->SetResponseStreamFactory(stream_factory);
    auto& registry = GcpCredentialRegistry::Instance();
    auto provider = registry.Lookup(uri);
    if (provider != nullptr && provider->IssuesBearerToken() && registry.MayCarryBearerToken(uri)) {
      if (auto header = provider->AuthorizationHeader()) {
        request->SetHeaderValue(header->first.c_str(), header->second.c_str());
      }
    }
    return request;
  }

  void InitStaticState() override { Aws::Http::CurlHttpClient::InitGlobalState(); }

  void CleanupStaticState() override { Aws::Http::CurlHttpClient::CleanupGlobalState(); }
};

// The SignedHeaders list of a SigV4 Authorization header value.
std::string SignedHeaders(const std::string& authorization) {
  const auto start = authorization.find("SignedHeaders=");
  if (start == std::string::npos) {
    return "";
  }
  const auto end = authorization.find(',', start);
  return authorization.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

std::string Join(const std::vector<std::string>& values) {
  std::string joined;
  for (const auto& value : values) {
    joined += "[" + value + "]";
  }
  return joined;
}

// Runs the scenario in a fresh process, so the GCP producer initializes the
// AWS SDK with its HTTP client factory and the registry starts empty. The
// scenario returns an empty string on success, or what went wrong.
void ExpectInFreshProcess(const std::function<std::string()>& scenario) {
#if defined(_WIN32)
  GTEST_SKIP() << "Test requires POSIX process APIs.";
#else
  const auto original_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto run = [&scenario]() -> int {
    const auto error = scenario();
    if (!error.empty()) {
      std::cerr << error << std::endl;
      return 1;
    }
    return 0;
  };
  EXPECT_EXIT((::alarm(60), ::_exit(run())), ::testing::ExitedWithCode(0), "");
  GTEST_FLAG_SET(death_test_style, original_style);
#endif
}

}  // namespace

class GcpClientBindingTest : public ::testing::Test {
  protected:
  static arrow::Result<ArrowFileSystemPtr> Make(const ArrowFileSystemConfig& config,
                                                std::shared_ptr<GcpCredentialProvider> provider) {
    return GcpFileSystemProducer(config).MakeWithProvider(std::move(provider));
  }
};

TEST_F(GcpClientBindingTest, ExternalKeepsItsIdentityWhenThePrimaryLaterUsesTheSameBucket) {
  ExpectInFreshProcess([]() -> std::string {
    RecordingServer server;
    // The order of a refresh: explore the external source, then open the
    // primary filesystem to commit the manifest.
    auto external = Make(LocalGcsConfig(server.port(), "shared-bucket", "external"),
                         std::make_shared<StaticBearerProvider>("external-token"));
    if (!external.ok()) {
      return "external Make: " + external.status().ToString();
    }
    auto primary = Make(LocalGcsConfig(server.port(), "shared-bucket", ""),
                        std::make_shared<StaticBearerProvider>("primary-token"));
    if (!primary.ok()) {
      return "primary Make: " + primary.status().ToString();
    }
    auto external_info = (*external)->GetFileInfo("shared-bucket/object");
    if (!external_info.ok()) {
      return "external GetFileInfo: " + external_info.status().ToString();
    }
    auto primary_info = (*primary)->GetFileInfo("shared-bucket/object");
    if (!primary_info.ok()) {
      return "primary GetFileInfo: " + primary_info.status().ToString();
    }
    const std::vector<std::string> expected = {"Bearer external-token", "Bearer primary-token"};
    if (server.Authorizations() != expected) {
      return "authorizations " + Join(server.Authorizations()) + ", expected " + Join(expected);
    }
    return "";
  });
}

TEST_F(GcpClientBindingTest, ExternalsWithDifferentIdentitiesShareABucket) {
  ExpectInFreshProcess([]() -> std::string {
    RecordingServer server;
    // The primary filesystem makes this endpoint operator configuration, so
    // Bearer tokens may go to it over plain HTTP.
    auto primary = Make(LocalGcsConfig(server.port(), "primary-bucket", ""),
                        std::make_shared<StaticBearerProvider>("primary-token"));
    if (!primary.ok()) {
      return "primary Make: " + primary.status().ToString();
    }
    auto provider_a = std::make_shared<StaticBearerProvider>("token-a");
    auto external_a = Make(LocalGcsConfig(server.port(), "shared-bucket", "a"), provider_a);
    if (!external_a.ok()) {
      return "external a Make: " + external_a.status().ToString();
    }
    // A second identity on the same bucket, such as a rotated target
    // service account, while the first filesystem is still cached.
    auto external_b =
        Make(LocalGcsConfig(server.port(), "shared-bucket", "b"), std::make_shared<StaticBearerProvider>("token-b"));
    if (!external_b.ok()) {
      return "external b Make: " + external_b.status().ToString();
    }
    // The first identity again, as after a cache eviction: it reuses the
    // registered provider and its cached token.
    auto external_a_again = Make(LocalGcsConfig(server.port(), "shared-bucket", "a-again"),
                                 std::make_shared<StaticBearerProvider>("token-a-again", "static:token-a"));
    if (!external_a_again.ok()) {
      return "external a again Make: " + external_a_again.status().ToString();
    }
    for (const auto* fs : {&*external_a, &*external_b, &*external_a_again}) {
      auto info = (*fs)->GetFileInfo("shared-bucket/object");
      if (!info.ok()) {
        return "GetFileInfo: " + info.status().ToString();
      }
    }
    const std::vector<std::string> expected = {"Bearer token-a", "Bearer token-b", "Bearer token-a"};
    if (server.Authorizations() != expected) {
      return "authorizations " + Join(server.Authorizations()) + ", expected " + Join(expected);
    }
    // Bearer filesystems all give the SDK anonymous credentials, so only their
    // identities tell them apart.
    if ((*external_a)->Equals(**external_b) || !(*external_a)->Equals(**external_a_again)) {
      return "Equals does not follow the filesystems' identities";
    }
    // b's request must not ask a's provider, which may be failing after a
    // rotation revoked it.
    if (provider_a->calls() != 2) {
      return "provider a was asked " + std::to_string(provider_a->calls()) + " times, expected 2";
    }
    return "";
  });
}

TEST_F(GcpClientBindingTest, BoundClientSendsNoBearerTokenToAnUntrustedEndpoint) {
  ExpectInFreshProcess([]() -> std::string {
    RecordingServer server;
    // An external filesystem's endpoint comes from a user spec. Without a
    // primary filesystem on it, plain HTTP to a non-Google host is untrusted.
    auto external = Make(LocalGcsConfig(server.port(), "external-bucket", "external"),
                         std::make_shared<StaticBearerProvider>("external-token"));
    if (!external.ok()) {
      return "external Make: " + external.status().ToString();
    }
    auto info = (*external)->GetFileInfo("external-bucket/object");
    if (!info.ok()) {
      return "GetFileInfo: " + info.status().ToString();
    }
    // The same gate applies to CRT and native transport requests.
    auto crt_config = LocalGcsConfig(server.port(), "crt-bucket", "crt");
    crt_config.s3_use_crt = true;
    auto crt = Make(crt_config, std::make_shared<StaticBearerProvider>("crt-token"));
    if (!crt.ok()) {
      return "CRT external Make: " + crt.status().ToString();
    }
    arrow::fs::FileInfo crt_info("crt-bucket/object", arrow::fs::FileType::File);
    crt_info.set_size(9);
    auto input = (*crt)->OpenInputFile(crt_info);
    if (!input.ok()) {
      return "CRT OpenInputFile: " + input.status().ToString();
    }
    auto bytes = (*input)->ReadAt(2, 4);
    if (!bytes.ok()) {
      return "CRT ReadAt: " + bytes.status().ToString();
    }
    auto metadata = (*input)->ReadMetadataAsync({}).result();
    if (!metadata.ok()) {
      return "CRT ReadMetadataAsync: " + metadata.status().ToString();
    }
    const std::vector<std::string> expected = {"", "", ""};
    if (server.Authorizations() != expected) {
      return "authorizations " + Join(server.Authorizations()) + ", expected " + Join(expected);
    }
    return "";
  });
}

TEST_F(GcpClientBindingTest, BoundClientWithoutATokenSendsNoOtherIdentitysToken) {
  ExpectInFreshProcess([]() -> std::string {
    RecordingServer server;
    auto primary = Make(LocalGcsConfig(server.port(), "shared-bucket", ""),
                        std::make_shared<StaticBearerProvider>("primary-token"));
    if (!primary.ok()) {
      return "primary Make: " + primary.status().ToString();
    }
    // The registry entry holds the primary's identity. Its token must not
    // reach this filesystem's request, even when there is no token of its own.
    auto external = Make(LocalGcsConfig(server.port(), "shared-bucket", "external"),
                         std::make_shared<StaticBearerProvider>("", "static:no-token"));
    if (!external.ok()) {
      return "external Make: " + external.status().ToString();
    }
    auto info = (*external)->GetFileInfo("shared-bucket/object");
    if (!info.ok()) {
      return "GetFileInfo: " + info.status().ToString();
    }
    const std::vector<std::string> expected = {""};
    if (server.Authorizations() != expected) {
      return "authorizations " + Join(server.Authorizations()) + ", expected " + Join(expected);
    }
    return "";
  });
}

TEST_F(GcpClientBindingTest, BearerExternalSharesTheHmacPrimaryBucket) {
  ExpectInFreshProcess([]() -> std::string {
    RecordingServer server;
    auto primary_config = LocalGcsConfig(server.port(), "shared-bucket", "");
    primary_config.access_key_id = "GOOGBINDINGTESTKEY";
    primary_config.access_key_value = "binding-test-secret";
    auto hmac_provider = std::make_shared<StaticHmacProvider>();
    auto primary = Make(primary_config, hmac_provider);
    if (!primary.ok()) {
      return "primary Make: " + primary.status().ToString();
    }
    auto external = Make(LocalGcsConfig(server.port(), "shared-bucket", "external"),
                         std::make_shared<StaticBearerProvider>("external-token"));
    if (!external.ok()) {
      return "external Make: " + external.status().ToString();
    }
    for (const auto* fs : {&*primary, &*external}) {
      auto info = (*fs)->GetFileInfo("shared-bucket/object");
      if (!info.ok()) {
        return "GetFileInfo: " + info.status().ToString();
      }
    }
    const auto authorizations = server.Authorizations();
    if (authorizations.size() != 2 ||
        !authorizations[0].starts_with("AWS4-HMAC-SHA256 Credential=GOOGBINDINGTESTKEY/") ||
        authorizations[1] != "Bearer external-token") {
      return "authorizations " + Join(authorizations) +
             ", expected the primary's SigV4 signature and the external Bearer token";
    }
    // The primary's bound delegator still runs its GOOG4 re-sign step.
    if (hmac_provider->calls() != 1) {
      return "the HMAC provider's re-sign step ran " + std::to_string(hmac_provider->calls()) + " times, expected 1";
    }
    return "";
  });
}

TEST_F(GcpClientBindingTest, HmacExternalsKeepTheirSignatureOnABearerBucket) {
  ExpectInFreshProcess([]() -> std::string {
    RecordingServer server;
    auto hmac_config = LocalGcsConfig(server.port(), "shared-bucket", "hmac");
    hmac_config.access_key_id = "GOOGBINDINGTESTKEY";
    hmac_config.access_key_value = "binding-test-secret";
    auto hmac = Make(hmac_config, std::make_shared<StaticHmacProvider>());
    if (!hmac.ok()) {
      return "HMAC external Make: " + hmac.status().ToString();
    }
    // The primary's trusted registration replaces the HMAC entry.
    auto primary = Make(LocalGcsConfig(server.port(), "shared-bucket", ""),
                        std::make_shared<StaticBearerProvider>("primary-token"));
    if (!primary.ok()) {
      return "primary Make: " + primary.status().ToString();
    }
    auto info = (*hmac)->GetFileInfo("shared-bucket/object");
    if (!info.ok()) {
      return "GetFileInfo: " + info.status().ToString();
    }
    const auto authorizations = server.Authorizations();
    if (authorizations.size() != 1 ||
        !authorizations[0].starts_with("AWS4-HMAC-SHA256 Credential=GOOGBINDINGTESTKEY/")) {
      return "authorizations " + Join(authorizations) + ", expected the external's SigV4 signature";
    }
    // SigV4 signs every header present when it runs. A Bearer header attached
    // by URI before signing would be in SignedHeaders, and GCS would reject
    // the signature.
    if (SignedHeaders(authorizations[0]).find("authorization") != std::string::npos) {
      return "SigV4 signed an Authorization header: " + authorizations[0];
    }
    // An HMAC external built after the primary is refused an entry on the
    // primary's bucket, and still signs with its own keys.
    auto late_config = LocalGcsConfig(server.port(), "shared-bucket", "late-hmac");
    late_config.access_key_id = "GOOGBINDINGTESTLATE";
    late_config.access_key_value = "binding-test-late-secret";
    auto late = Make(late_config, std::make_shared<StaticHmacProvider>());
    if (!late.ok()) {
      return "late HMAC external Make: " + late.status().ToString();
    }
    auto late_info = (*late)->GetFileInfo("shared-bucket/object");
    if (!late_info.ok()) {
      return "late GetFileInfo: " + late_info.status().ToString();
    }
    const auto late_authorizations = server.Authorizations();
    if (late_authorizations.size() != 2 ||
        !late_authorizations[1].starts_with("AWS4-HMAC-SHA256 Credential=GOOGBINDINGTESTLATE/") ||
        SignedHeaders(late_authorizations[1]).find("authorization") != std::string::npos) {
      return "authorizations " + Join(late_authorizations) + ", expected the late external's SigV4 signature";
    }
    // The same with CRT: neither the CRT GET nor the native HEAD gets the
    // primary's Bearer token.
    auto crt_config = LocalGcsConfig(server.port(), "shared-bucket", "crt-hmac");
    crt_config.access_key_id = "GOOGBINDINGTESTCRT";
    crt_config.access_key_value = "binding-test-crt-secret";
    crt_config.s3_use_crt = true;
    auto crt = Make(crt_config, std::make_shared<StaticHmacProvider>());
    if (!crt.ok()) {
      return "CRT HMAC external Make: " + crt.status().ToString();
    }
    arrow::fs::FileInfo crt_info("shared-bucket/object", arrow::fs::FileType::File);
    crt_info.set_size(9);
    auto crt_input = (*crt)->OpenInputFile(crt_info);
    if (!crt_input.ok()) {
      return "CRT OpenInputFile: " + crt_input.status().ToString();
    }
    auto crt_bytes = (*crt_input)->ReadAt(2, 4);
    if (!crt_bytes.ok()) {
      return "CRT ReadAt: " + crt_bytes.status().ToString();
    }
    auto crt_metadata = (*crt_input)->ReadMetadataAsync({}).result();
    if (!crt_metadata.ok()) {
      return "CRT ReadMetadataAsync: " + crt_metadata.status().ToString();
    }
    const auto crt_authorizations = server.Authorizations();
    if (crt_authorizations.size() != 4) {
      return "authorizations " + Join(crt_authorizations) + ", expected two more CRT requests";
    }
    for (size_t i = 2; i < 4; ++i) {
      if (!crt_authorizations[i].starts_with("AWS4-HMAC-SHA256 Credential=GOOGBINDINGTESTCRT/") ||
          SignedHeaders(crt_authorizations[i]).find("authorization") != std::string::npos) {
        return "authorizations " + Join(crt_authorizations) + ", expected the CRT external's SigV4 signatures";
      }
    }
    return "";
  });
}

TEST_F(GcpClientBindingTest, CrtFilesystemsKeepTheirIdentities) {
  ExpectInFreshProcess([]() -> std::string {
    RecordingServer server;
    const auto crt_config = [&server](const std::string& alias) {
      auto config = LocalGcsConfig(server.port(), "shared-bucket", alias);
      config.s3_use_crt = true;
      return config;
    };
    auto external_a = Make(crt_config("a"), std::make_shared<StaticBearerProvider>("token-a"));
    if (!external_a.ok()) {
      return "external a Make: " + external_a.status().ToString();
    }
    // The primary's trusted registration replaces a's entry.
    auto primary = Make(LocalGcsConfig(server.port(), "shared-bucket", ""),
                        std::make_shared<StaticBearerProvider>("primary-token"));
    if (!primary.ok()) {
      return "primary Make: " + primary.status().ToString();
    }
    // Another identity on the bucket.
    auto external_b = Make(crt_config("b"), std::make_shared<StaticBearerProvider>("token-b"));
    if (!external_b.ok()) {
      return "external b Make: " + external_b.status().ToString();
    }
    arrow::fs::FileInfo info("shared-bucket/object", arrow::fs::FileType::File);
    info.set_size(9);
    // Open one by FileInfo and one by path; each overload passes the
    // authenticator on.
    for (int i = 0; i < 2; ++i) {
      auto input = i == 0 ? (*external_a)->OpenInputFile(info) : (*external_b)->OpenInputFile(info.path());
      if (!input.ok()) {
        return "OpenInputFile: " + input.status().ToString();
      }
      // A CRT meta request, then a native transport HEAD.
      auto bytes = (*input)->ReadAt(2, 4);
      if (!bytes.ok() || (*bytes)->ToString() != "cdef") {
        return "ReadAt: " + (bytes.ok() ? (*bytes)->ToString() : bytes.status().ToString());
      }
      auto metadata = (*input)->ReadMetadataAsync({}).result();
      if (!metadata.ok()) {
        return "ReadMetadataAsync: " + metadata.status().ToString();
      }
    }
    const std::vector<std::string> expected = {"Bearer token-a", "Bearer token-a", "Bearer token-b", "Bearer token-b"};
    const std::vector<std::string> methods = {"GET", "HEAD", "GET", "HEAD"};
    if (server.Authorizations() != expected || server.Methods() != methods) {
      return "requests " + Join(server.Methods()) + " " + Join(server.Authorizations()) + ", expected " +
             Join(methods) + " " + Join(expected);
    }
    return "";
  });
}

TEST_F(GcpClientBindingTest, ConcurrentMakeCallsBindTheirOwnProviders) {
  ExpectInFreshProcess([]() -> std::string {
    RecordingServer server;
    auto primary = Make(LocalGcsConfig(server.port(), "primary-bucket", ""),
                        std::make_shared<StaticBearerProvider>("primary-token"));
    if (!primary.ok()) {
      return "primary Make: " + primary.status().ToString();
    }
    constexpr int kThreads = 8;
    std::vector<ArrowFileSystemPtr> filesystems(kThreads);
    std::vector<std::string> errors(kThreads);
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
      threads.emplace_back([&, i] {
        auto made = Make(LocalGcsConfig(server.port(), "shared-bucket", "external-" + std::to_string(i)),
                         std::make_shared<StaticBearerProvider>("token-" + std::to_string(i)));
        if (made.ok()) {
          filesystems[i] = std::move(made).ValueOrDie();
        } else {
          errors[i] = made.status().ToString();
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    std::vector<std::string> expected;
    for (int i = 0; i < kThreads; ++i) {
      if (!errors[i].empty()) {
        return "external " + std::to_string(i) + " Make: " + errors[i];
      }
      auto info = filesystems[i]->GetFileInfo("shared-bucket/object");
      if (!info.ok()) {
        return "GetFileInfo: " + info.status().ToString();
      }
      expected.push_back("Bearer token-" + std::to_string(i));
    }
    if (server.Authorizations() != expected) {
      return "authorizations " + Join(server.Authorizations()) + ", expected " + Join(expected);
    }
    return "";
  });
}

TEST_F(GcpClientBindingTest, FactoryInstalledOutsideMilvusStorageKeepsOneIdentityPerBucket) {
  ExpectInFreshProcess([]() -> std::string {
    // The SDK ignores milvus-storage's later InitAPI, so this factory serves
    // every client and nothing is bound.
    Aws::SDKOptions options;
    options.httpOptions.httpClientFactory_create_fn = [] {
      return Aws::MakeShared<RegistryLookupHttpClientFactory>(kForeignFactoryTag);
    };
    Aws::InitAPI(options);

    RecordingServer server;
    const auto endpoint = "http://127.0.0.1:" + std::to_string(server.port()) + "/";
    auto& registry = GcpCredentialRegistry::Instance();
    auto primary = Make(LocalGcsConfig(server.port(), "primary-bucket", ""),
                        std::make_shared<StaticBearerProvider>("primary-token"));
    if (!primary.ok()) {
      return "primary Make: " + primary.status().ToString();
    }
    auto external_provider = std::make_shared<StaticBearerProvider>("external-token");
    auto external = Make(LocalGcsConfig(server.port(), "external-bucket", "external"), external_provider);
    if (!external.ok()) {
      return "external Make: " + external.status().ToString();
    }
    if (registry.Lookup(Aws::Http::URI((endpoint + "external-bucket/object").c_str())) != external_provider) {
      return "an external filesystem with unbound clients is not registered";
    }
    auto info = (*external)->GetFileInfo("external-bucket/object");
    if (!info.ok()) {
      return "GetFileInfo: " + info.status().ToString();
    }
    const std::vector<std::string> expected = {"Bearer external-token"};
    if (server.Authorizations() != expected) {
      return "authorizations " + Join(server.Authorizations()) + ", expected " + Join(expected);
    }
    // Unbound clients find credentials by URI, so another identity on a
    // registered bucket is refused as before.
    auto intruder = Make(LocalGcsConfig(server.port(), "primary-bucket", "intruder"),
                         std::make_shared<StaticBearerProvider>("intruder-token"));
    if (intruder.ok() || intruder.status().message().find("own storage") == std::string::npos) {
      return "another identity on the primary's bucket: " + intruder.status().ToString();
    }
    return "";
  });
}

}  // namespace milvus_storage
