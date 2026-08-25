// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>

#include <arrow/status.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>

namespace milvus_storage::gcp_internal {

// Convert a request-local token failure into the synthetic response consumed
// by the S3 SDK. Kept source-private, but named so the marshalling boundary can
// be exercised directly in tests.
std::shared_ptr<Aws::Http::HttpResponse> MakeTokenErrorResponse(const std::shared_ptr<Aws::Http::HttpRequest>& request,
                                                                const arrow::Status& token_status);

}  // namespace milvus_storage::gcp_internal
