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
inline constexpr char kDurationSeconds[] = "durationSeconds";
inline constexpr char kRoleArn[] = "roleArn";
inline constexpr char kExternalId[] = "externalId";

// IAM uses the Broker's own identity; omitting credentialType retains target-identity mode.
inline constexpr char kIam[] = "iam";

// Broker CSP aliases differ from storage provider names for Aliyun and Tencent.
inline constexpr char kCspAws[] = "aws";
inline constexpr char kCspAliyun[] = "ali";
inline constexpr char kCspTencent[] = "tc";

// Credential Broker HTTP response fields shared by credential providers.
inline constexpr char kSuccess[] = "success";
inline constexpr char kMsg[] = "msg";
inline constexpr char kDetail[] = "detail";
inline constexpr char kCredentials[] = "credentials";
inline constexpr char kTempAk[] = "tempAk";
inline constexpr char kTempSk[] = "tempSk";
inline constexpr char kSessionToken[] = "sessionToken";
inline constexpr char kExpiredAt[] = "expiredAt";

}  // namespace milvus_storage::credential_broker
