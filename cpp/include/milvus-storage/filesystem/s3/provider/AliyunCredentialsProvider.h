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
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/internal/AWSHttpResourceClient.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <memory>
#include "AliyunSTSClient.h"

namespace milvus_storage {

/**
 * To support retrieving credentials of STS AssumeRole with web identity.
 * Note that STS accepts request with protocol of queryxml. Calling GetAWSCredentials() will trigger (if expired)
 * a query request using AWSHttpResourceClient under the hood.
 */
class AWS_CORE_API AliyunSTSAssumeRoleWebIdentityCredentialsProvider : public ::Aws::Auth::AWSCredentialsProvider {
  public:
  // Reads role_arn, session_name, and OIDC token file from ALIBABA_CLOUD_* env
  // vars, with profile-config fallback.
  AliyunSTSAssumeRoleWebIdentityCredentialsProvider();

  // Per-tenant ctor. role_arn and session_name come from the caller; OIDC
  // token file and provider ARN are still read from process env (machine
  // identity). No profile-config fallback — the caller is authoritative.
  AliyunSTSAssumeRoleWebIdentityCredentialsProvider(const ::Aws::String& role_arn, const ::Aws::String& session_name);

  /**
   * Retrieves the credentials if found, otherwise returns empty credential set.
   */
  ::Aws::Auth::AWSCredentials GetAWSCredentials() override;

  protected:
  void Reload() override;

  private:
  void InitializeClient();
  void RefreshIfExpired();
  ::Aws::String CalculateQueryString() const;

  ::Aws::UniquePtr<AliyunSTSCredentialsClient> m_client;
  ::Aws::Auth::AWSCredentials m_credentials;
  ::Aws::String m_roleArn;
  ::Aws::String m_tokenFile;
  ::Aws::String m_sessionName;
  ::Aws::String m_token;
  bool m_initialized;
  bool ExpiresSoon() const;
};

}  // namespace milvus_storage
