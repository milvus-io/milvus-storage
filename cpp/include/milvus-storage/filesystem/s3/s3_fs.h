// Copyright 2024 Zilliz
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

#include <aws/core/Aws.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/curl/CurlHttpClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/util/uri.h>
#include <google/cloud/internal/oauth2_credentials.h>
#include <google/cloud/internal/oauth2_google_credentials.h>
#include <google/cloud/storage/oauth2/compute_engine_credentials.h>
#include <google/cloud/storage/oauth2/google_credentials.h>
#include <google/cloud/status_or.h>
#include <cstdlib>
#include "milvus-storage/common/macro.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"

namespace milvus_storage {

static std::unordered_map<std::string, arrow::fs::S3LogLevel> LogLevel_Map = {
    {"off", arrow::fs::S3LogLevel::Off},     {"fatal", arrow::fs::S3LogLevel::Fatal},
    {"error", arrow::fs::S3LogLevel::Error}, {"warn", arrow::fs::S3LogLevel::Warn},
    {"info", arrow::fs::S3LogLevel::Info},   {"debug", arrow::fs::S3LogLevel::Debug},
    {"trace", arrow::fs::S3LogLevel::Trace}};

class S3FileSystemProducer : public FileSystemProducer {
  public:
  S3FileSystemProducer(const ArrowFileSystemConfig& config) : config_(config) {}

  arrow::Result<ArrowFileSystemPtr> Make() override;

  arrow::Result<ExtendedS3Options> CreateS3Options();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateCredentialsProvider();

  void InitS3();

  private:
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateAwsCredentialsProvider();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateAliyunCredentialsProvider();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateTencentCredentialsProvider();

  private:
  const ArrowFileSystemConfig config_;
};

static const char* GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG = "GoogleHttpClientFactory";

class GoogleHttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  explicit GoogleHttpClientFactory(std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials) {
    credentials_ = credentials;
  }

  void SetCredentials(std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials) {
    credentials_ = credentials;
  }

  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& clientConfiguration) const override {
    return Aws::MakeShared<Aws::Http::CurlHttpClient>(GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG, clientConfiguration);
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::String& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& streamFactory) const override {
    return CreateHttpRequest(Aws::Http::URI(uri), method, streamFactory);
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::Http::URI& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& streamFactory) const override {
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(GOOGLE_CLIENT_FACTORY_ALLOCATION_TAG, uri, method);
    request->SetResponseStreamFactory(streamFactory);
    auto auth_header = credentials_->AuthorizationHeader();
    if (!auth_header.ok()) {
      throw std::invalid_argument("GoogleHttpClientFactory: create http request get authorization failed");
    }
    request->SetHeaderValue(auth_header->first.c_str(), auth_header->second.c_str());

    return request;
  }

  private:
  std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials_;
};

}  // namespace milvus_storage
