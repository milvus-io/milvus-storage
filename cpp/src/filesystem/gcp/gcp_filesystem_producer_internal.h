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
#include <string>

#include <aws/core/http/HttpClientFactory.h>

namespace milvus_storage::gcp_internal {

// Builds the registry-backed HTTP client factory that
// GcpFileSystemProducer::InitS3Compat installs process-wide. Exposed from the
// source tree (not the public include directory) so unit tests can drive the
// factory's Bearer delivery gate directly: CreateHttpRequest adds an
// Authorization header only for a registered provider that issues Bearer
// tokens and a URI that GcpCredentialRegistry::MayCarryBearerToken accepts.
std::shared_ptr<Aws::Http::HttpClientFactory> MakeGoogleHttpClientFactory(std::string tls_min_version);

}  // namespace milvus_storage::gcp_internal
