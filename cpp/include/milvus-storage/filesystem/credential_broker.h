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

namespace milvus_storage::credential_broker {

// Credential Broker HTTP request fields. These names are part of the wire protocol.
inline constexpr char kCsp[] = "csp";
inline constexpr char kCredentialType[] = "credentialType";
inline constexpr char kRegion[] = "region";
inline constexpr char kBucket[] = "bucket";
inline constexpr char kDurationSeconds[] = "durationSeconds";
inline constexpr char kRoleArn[] = "roleArn";
inline constexpr char kExternalId[] = "externalId";
inline constexpr char kGcpServiceAccountEmail[] = "gcpServiceAccountEmail";
inline constexpr char kAzureAccountName[] = "azureAccountName";
inline constexpr char kAzureClientId[] = "azureClientId";
inline constexpr char kAzureTenantId[] = "azureTenantId";

// IAM uses the Broker's own identity; omitting credentialType retains target-identity mode.
inline constexpr char kIam[] = "iam";

// Credential Broker HTTP response fields consumed by the Azure SAS provider.
inline constexpr char kSuccess[] = "success";
inline constexpr char kCredentials[] = "credentials";
inline constexpr char kTempAk[] = "tempAk";
inline constexpr char kSessionToken[] = "sessionToken";
inline constexpr char kExpiredAt[] = "expiredAt";

// Talon request descriptor: {url, request: {...}, origin: {...}}.
inline constexpr char kUrl[] = "url";
inline constexpr char kRequest[] = "request";
inline constexpr char kOrigin[] = "origin";

// Talon origin fields; region shares kRegion with the Broker request.
inline constexpr char kEndpoint[] = "endpoint";
inline constexpr char kUseVirtualHost[] = "use_virtual_host";
inline constexpr char kRequestTimeoutMs[] = "request_timeout_ms";

}  // namespace milvus_storage::credential_broker
