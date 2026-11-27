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

#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/core/utils/stream/ResponseStream.h>

namespace milvus_storage {

// Credential providers call STS while they hold their reload lock, so every
// request signed by the provider waits for the call, including its retries.
//
// Each attempt has to connect within kSTSConnectTimeoutMs, the SDK default, and
// finish within kSTSAttemptTimeoutMs, TLS and the response included. DNS counts
// against both limits, but curl cannot interrupt a lookup in progress, so an
// attempt lasts at least as long as its lookup. The SDK defaults have no limit
// on the whole attempt, and their no-progress limit also counts the uploaded
// body, so a stalled attempt ran for about 9 s, and one whose response body
// trickled ran for as long as the trickle lasted. The cost of the shorter
// limit: an STS that takes 5 to 9 s to answer used to succeed and now fails
// the attempt.
//
// The second-hop and AWS role_arn STS clients retry a call that fails with an
// error the SDK treats as retryable up to kSTSMaxRetries times, as the first
// hops already did. With the SDK's default retry strategy (10 retries), the
// backoff alone took about 25.6 s when STS kept failing.
inline constexpr long kSTSConnectTimeoutMs = 1000;
inline constexpr long kSTSAttemptTimeoutMs = 5000;
inline constexpr long kSTSMaxRetries = 3;

inline void ApplySTSTimeouts(Aws::Client::ClientConfiguration& config) {
  config.connectTimeoutMs = kSTSConnectTimeoutMs;
  config.httpRequestTimeoutMs = kSTSAttemptTimeoutMs;
}

// SetResendableBody sets body as the content of request and makes every attempt
// send all of it. AWSHttpResourceClient retries by sending the same request
// again, and the HTTP client reads the body from the stream's current
// position, so after an attempt that uploaded the body, a retry without a
// rewind goes out with the original Content-Length and no body, and the server
// waits for a body that never arrives. The SDK's curl client creates the
// response stream at the start of each attempt, before it uploads the body, so
// the body is rewound there; clear() resets the end-of-stream flags that the
// previous upload left, without which seekg() does nothing.
inline void SetResendableBody(Aws::Http::HttpRequest& request, const Aws::String& body, const char* allocation_tag) {
  auto stream = Aws::MakeShared<Aws::StringStream>(allocation_tag);
  *stream << body;
  request.AddContentBody(stream);
  request.SetContentLength(std::to_string(body.size()));
  Aws::IOStreamFactory response_stream_factory = request.GetResponseStreamFactory();
  if (!response_stream_factory) {
    response_stream_factory = Aws::Utils::Stream::DefaultResponseStreamFactoryMethod;
  }
  request.SetResponseStreamFactory([stream, response_stream_factory] {
    stream->clear();
    stream->seekg(0);
    return response_stream_factory();
  });
}

}  // namespace milvus_storage
