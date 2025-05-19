
/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/client/AWSErrorMarshaller.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/AmazonWebServiceResult.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/internal/AWSHttpResourceClient.h>
#include <memory>
#include <mutex>

namespace Aws {
namespace Http {
class HttpClient;
class HttpRequest;
enum class HttpResponseCode;
}  // namespace Http

namespace Internal {
/**
 * To support retrieving credentials from STS.
 * Note that STS accepts request with protocol of queryxml. Calling GetResource() will trigger
 * a query request using AWSHttpResourceClient under the hood.
 */
class AWS_CORE_API TencentCloudSTSCredentialsClient : public AWSHttpResourceClient {
  public:
  /**
   * Initializes the provider to retrieve credentials from STS when it expires.
   */
  explicit TencentCloudSTSCredentialsClient(const Client::ClientConfiguration& clientConfiguration);

  TencentCloudSTSCredentialsClient& operator=(TencentCloudSTSCredentialsClient& rhs) = delete;
  TencentCloudSTSCredentialsClient(const TencentCloudSTSCredentialsClient& rhs) = delete;
  TencentCloudSTSCredentialsClient& operator=(TencentCloudSTSCredentialsClient&& rhs) = delete;
  TencentCloudSTSCredentialsClient(const TencentCloudSTSCredentialsClient&& rhs) = delete;

  // If you want to make an AssumeRoleWithWebIdentity call to sts. use these classes to pass data to and get info from
  // TencentCloudSTSCredentialsClient client. If you want to make an AssumeRole call to sts, define the request/result
  // members class/struct like this.
  struct STSAssumeRoleWithWebIdentityRequest {
    Aws::String region;
    Aws::String providerId;
    Aws::String webIdentityToken;
    Aws::String roleArn;
    Aws::String roleSessionName;
  };

  struct STSAssumeRoleWithWebIdentityResult {
    Aws::Auth::AWSCredentials creds;
  };

  STSAssumeRoleWithWebIdentityResult GetAssumeRoleWithWebIdentityCredentials(
      const STSAssumeRoleWithWebIdentityRequest& request);

  private:
  Aws::String m_endpoint;
};
}  // namespace Internal
}  // namespace Aws
