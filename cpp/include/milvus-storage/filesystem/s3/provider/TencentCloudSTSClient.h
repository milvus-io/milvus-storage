// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

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
#include <aws/core/utils/json/JsonSerializer.h>
#include <memory>
#include <mutex>

namespace milvus_storage {

/**
 * Retrieves credentials through Tencent's JSON STS API. Web identity calls
 * establish the caller; signed AssumeRole calls select an explicit data role.
 */
class AWS_CORE_API TencentCloudSTSCredentialsClient : public Aws::Internal::AWSHttpResourceClient {
  public:
  /**
   * Initializes the provider to retrieve credentials from STS when it expires.
   */
  explicit TencentCloudSTSCredentialsClient(const Aws::Client::ClientConfiguration& clientConfiguration);

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

  struct STSAssumeRoleRequest {
    Aws::Auth::AWSCredentials callerCredentials;
    Aws::String region;
    Aws::String roleArn;
    Aws::String roleSessionName;
    Aws::String externalId;
  };

  Aws::Auth::AWSCredentials GetAssumeRoleCredentials(const STSAssumeRoleRequest& request);

  private:
  Aws::Auth::AWSCredentials SendRequest(const Aws::String& action,
                                        const Aws::String& region,
                                        const Aws::Utils::Json::JsonValue& payload,
                                        const Aws::Auth::AWSCredentials* caller = nullptr);
  Aws::String m_endpoint;
};

}  // namespace milvus_storage
