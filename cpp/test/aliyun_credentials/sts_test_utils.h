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

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <aws/core/Aws.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/http/standard/StandardHttpResponse.h>
#include <aws/core/utils/UUID.h>
#include <glog/logging.h>

namespace milvus_storage {
namespace {
class ScopedEnvironment {
  public:
  ScopedEnvironment(const char* name, const std::string& value) : name_(name) {
    const auto* old = std::getenv(name);
    if (old)
      old_ = old;
    setenv(name, value.c_str(), 1);
  }
  ~ScopedEnvironment() {
    if (old_)
      setenv(name_, old_->c_str(), 1);
    else
      unsetenv(name_);
  }

  private:
  const char* name_;
  std::optional<std::string> old_;
};

class STSHttpClient : public Aws::Http::HttpClient {
  public:
  void BeforeNextResponse(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    before_response_ = std::move(callback);
  }
  void Enqueue(std::string body,
               Aws::Http::HttpResponseCode code = Aws::Http::HttpResponseCode::OK,
               Aws::Http::HeaderValueCollection headers = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    responses_.push({std::move(body), code, std::move(headers)});
  }
  std::shared_ptr<Aws::Http::HttpResponse> MakeRequest(const std::shared_ptr<Aws::Http::HttpRequest>& request,
                                                       Aws::Utils::RateLimits::RateLimiterInterface*,
                                                       Aws::Utils::RateLimits::RateLimiterInterface*) const override {
    std::unique_lock<std::mutex> lock(mutex_);
    ++requests_;
    auto callback = std::exchange(before_response_, {});
    if (callback) {
      lock.unlock();
      callback();
      lock.lock();
    }
    auto response = Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>("AliyunTest", request);
    if (responses_.empty()) {
      ADD_FAILURE() << "Unexpected STS request to " << request->GetURIString();
      response->SetResponseCode(Aws::Http::HttpResponseCode::FORBIDDEN);
      return response;
    }
    auto spec = std::move(responses_.front());
    responses_.pop();
    response->SetResponseCode(spec.code);
    for (const auto& [name, value] : spec.headers) response->AddHeader(name, value);
    response->GetResponseBody() << spec.body;
    return response;
  }
  size_t Requests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_;
  }
  size_t Pending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return responses_.size();
  }

  private:
  mutable std::mutex mutex_;
  struct Response {
    std::string body;
    Aws::Http::HttpResponseCode code;
    Aws::Http::HeaderValueCollection headers;
  };
  mutable std::queue<Response> responses_;
  mutable size_t requests_ = 0;
  mutable std::function<void()> before_response_;
};

class STSHttpFactory : public Aws::Http::HttpClientFactory {
  public:
  explicit STSHttpFactory(std::shared_ptr<STSHttpClient> client) : client_(std::move(client)) {}
  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(const Aws::Client::ClientConfiguration&) const override {
    return client_;
  }
  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::String& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& factory) const override {
    return CreateHttpRequest(Aws::Http::URI(uri), method, factory);
  }
  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::Http::URI& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& factory) const override {
    auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("AliyunTest", uri, method);
    request->SetResponseStreamFactory(factory);
    return request;
  }

  private:
  std::shared_ptr<STSHttpClient> client_;
};

std::string ExpirationAfter(std::chrono::seconds duration) {
  return (Aws::Utils::DateTime::Now() + std::chrono::duration_cast<std::chrono::milliseconds>(duration))
      .ToGmtString(Aws::Utils::DateFormat::ISO_8601);
}

std::string CredentialsXML(const std::string& expiration, const std::string& prefix = "old") {
  return "<AssumeRoleWithOIDCResponse><Credentials><AccessKeyId>" + prefix + "_ak</AccessKeyId><AccessKeySecret>" +
         prefix + "_sk</AccessKeySecret><SecurityToken>" + prefix + "_token</SecurityToken><Expiration>" + expiration +
         "</Expiration></Credentials></AssumeRoleWithOIDCResponse>";
}

class CapturedLogs : public google::LogSink {
  public:
  CapturedLogs() : old_verbosity_(FLAGS_v) {
    FLAGS_v = 6;
    google::AddLogSink(this);
  }
  ~CapturedLogs() override {
    google::RemoveLogSink(this);
    FLAGS_v = old_verbosity_;
  }
  void send(google::LogSeverity,
            const char*,
            const char*,
            int,
            const google::LogMessageTime&,
            const char* message,
            size_t length) override {
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.append(message, length);
    messages_ += '\n';
  }
  bool Contains(const std::string& text) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return messages_.find(text) != std::string::npos;
  }

  private:
  int old_verbosity_;
  mutable std::mutex mutex_;
  std::string messages_;
};

}  // namespace
}  // namespace milvus_storage
