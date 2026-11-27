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

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <arrow/result.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/endpoint/EndpointProviderBase.h>

namespace Aws {
class AmazonWebServiceRequest;
}

namespace Aws::Client {
class AWSClient;
}  // namespace Aws::Client

namespace milvus_storage {

struct ArrowFileSystemConfig;
struct S3Options;

class GcpCredentialProvider : public Aws::Auth::AWSCredentialsProvider {
  public:
  // Configure one client before publication. True means its direct CRT reads
  // need bearer authorization before native submission.
  static bool ConfigS3HttpClient(const S3Options& options, Aws::Client::AWSClient& client);
  // Direct reads bypass the SDK transmit interceptor. Validate the SDK-resolved
  // destination before attaching this client's bearer to the modeled request.
  static arrow::Result<bool> AuthorizeRead(const std::shared_ptr<Aws::Auth::AWSCredentialsProvider>& credentials,
                                           Aws::AmazonWebServiceRequest& model,
                                           Aws::Endpoint::ResolveEndpointOutcome endpoint,
                                           const std::string& key,
                                           Aws::Http::HttpMethod method);

  // Bearer providers set the Authorization header in AuthorizeRequest(). Empty credentials
  // tell the AWS SDK to skip SigV4 signing. The HMAC provider overrides this with its key pair.
  Aws::Auth::AWSCredentials GetAWSCredentials() override { return {}; }

  protected:
  explicit GcpCredentialProvider(const ArrowFileSystemConfig& config);

  private:
  friend struct S3Options;
  friend class GcpFileSystemProducer;
  friend class NativeS3Transport;
  class RequestInterceptor;

  static void TrustEndpoint(const ArrowFileSystemConfig& config);
  static bool MayCarryBearerToken(const Aws::Http::URI& uri);

  // True when GCP supplied the final Authorization header. False preserves
  // the SDK signature (ordinary HMAC). An unavailable bearer leaves the request unsigned.
  virtual arrow::Result<bool> AuthorizeRequest(const std::shared_ptr<Aws::Http::HttpRequest>& request) = 0;

  arrow::Result<bool> AuthorizeScopedRequest(const std::shared_ptr<Aws::Http::HttpRequest>& request, bool bearer);
  const Aws::Http::URI endpoint_;
  const std::string bucket_;
  const Aws::String virtual_host_;
  const std::string config_key_;
  std::once_flag untrusted_warning_;
};

arrow::Result<std::shared_ptr<GcpCredentialProvider>> BuildGcpProviderFromConfig(const ArrowFileSystemConfig& config);

}  // namespace milvus_storage
