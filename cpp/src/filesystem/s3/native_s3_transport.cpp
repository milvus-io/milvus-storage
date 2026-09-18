// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#include "filesystem/s3/native_s3_transport.h"
#ifdef WITH_CRT
#include <algorithm>
#include <cctype>
#include <mutex>
#include <utility>
#include <arrow/filesystem/path_util.h>
#include <arrow/util/thread_pool.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/crt/http/HttpRequestResponse.h>
#include <aws/common/uri.h>
#include <aws/s3/s3_client.h>
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/util_internal.h"

namespace milvus_storage {
struct NativeS3Transport::State {
  std::mutex mutex;
  size_t inflight = 0;
  size_t limit = 0;
  std::shared_ptr<S3ClientHolder> holder;

  void Retire(bool network_pending) {
    std::lock_guard lock(mutex);
    if (network_pending)
      --inflight;
  }
};

namespace {
struct Request {
  S3CrtClientLease client_lease;
  std::shared_ptr<NativeS3Transport::State> state;
  arrow::io::IOContext io;
  size_t limit;
  std::shared_ptr<arrow::Buffer> data;
  std::unique_ptr<Aws::Utils::Stream::PreallocatedStreamBuf> streambuf;
  std::shared_ptr<Aws::Http::HttpRequest> http;
  std::shared_ptr<Aws::Crt::Http::HttpRequest> message;
  aws_uri endpoint{};
  NativeS3Response response;
  std::atomic<int> http_status{0};
  bool charged = false;
  bool network_pending = true;
  arrow::Future<NativeS3Response> future = arrow::Future<NativeS3Response>::Make();

  ~Request() {
    message.reset();
    http.reset();
    streambuf.reset();
    data.reset();
    aws_uri_clean_up(&endpoint);
    if (charged)
      state->Retire(network_pending);
  }
  static int Headers(aws_s3_meta_request*, const aws_http_headers* headers, int status, void* user) noexcept {
    auto& r = *static_cast<Request*>(user);
    r.http_status.store(status);
    try {
      for (size_t i = 0; i < aws_http_headers_count(headers); ++i) {
        aws_http_header h{};
        if (aws_http_headers_get_index(headers, i, &h))
          return AWS_OP_ERR;
        Aws::String name(reinterpret_cast<char*>(h.name.ptr), h.name.len);
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
        r.response.headers.emplace(std::move(name), Aws::String(reinterpret_cast<char*>(h.value.ptr), h.value.len));
      }
      return AWS_OP_SUCCESS;
    } catch (...) {
      return aws_raise_error(AWS_ERROR_OOM);
    }
  }
  static int Body(aws_s3_meta_request*, const aws_byte_cursor* data, uint64_t, void* user) noexcept {
    auto& r = *static_cast<Request*>(user);
    if (data->len > r.limit - r.response.body.size()) {
      r.response.status = arrow::Status::CapacityError("S3 response exceeds configured limit");
      return aws_raise_error(AWS_ERROR_S3_CANCELED);
    }
    try {
      r.response.body.append(reinterpret_cast<const char*>(data->ptr), data->len);
      return AWS_OP_SUCCESS;
    } catch (...) {
      return aws_raise_error(AWS_ERROR_OOM);
    }
  }
  static void Telemetry(aws_s3_meta_request*, aws_s3_request_metrics* metrics, void* user) noexcept {
    int status = 0;
    if (!aws_s3_request_metrics_get_response_status_code(metrics, &status) && status) {
      // Request callbacks are serialized for DEFAULT meta requests.
      static_cast<Request*>(user)->http_status.store(status);
    }
  }
  static void Finish(aws_s3_meta_request* meta, const aws_s3_meta_request_result* result, void* user) noexcept {
    auto& r = *static_cast<Request*>(user);
    if (result->response_status)
      r.http_status.store(result->response_status);
    r.response.transport_error = result->error_code;
    // Release our initial reference. Shutdown, not Finish, is the point at which
    // CRT has stopped using all request buffers and invoking request callbacks.
    aws_s3_meta_request_release(meta);
  }
  static void Shutdown(void* user) noexcept {
    S3CrtCallbackScope callback_scope;
    auto* request = static_cast<Request*>(user);
    request->response.http_status = request->http_status.load();
    auto future = request->future;
    auto response = std::move(request->response);
    std::shared_ptr<Request> owned;
    try {
      owned.reset(request);
    } catch (...) {
      // shared_ptr deletes request if control-block allocation fails.
      future.MarkFinished(std::move(response));
      return;
    }
    owned->response = std::move(response);
    auto complete = [owned] {
      // Bound queued responses too. Release admission immediately before user
      // continuations, so a one-slot client can submit the next page/request.
      {
        std::lock_guard lock(owned->state->mutex);
        --owned->state->inflight;
        owned->network_pending = false;
      }
      owned->future.MarkFinished(std::move(owned->response));
    };
    try {
      auto status = owned->io.executor()->Spawn(complete);
      if (status.ok())
        return;
    } catch (...) {
      // Preserve the known HTTP/write result on dispatch failure.
    }
    complete();
  }
};
}  // namespace

bool NativeS3Response::HasHttpStatus(int code) const {
  return status.ok() && http_status == code &&
         (transport_error == 0 || transport_error == AWS_ERROR_S3_INVALID_RESPONSE_STATUS ||
          transport_error == AWS_ERROR_S3_INTERNAL_ERROR || transport_error == AWS_ERROR_S3_SLOW_DOWN);
}

arrow::Status NativeS3Response::ToStatus() const {
  if (!status.ok())
    return status;
  const bool http_error = http_status >= 400 &&
                          (transport_error == 0 || transport_error == AWS_ERROR_S3_INVALID_RESPONSE_STATUS ||
                           transport_error == AWS_ERROR_S3_INTERNAL_ERROR || transport_error == AWS_ERROR_S3_SLOW_DOWN);
  if (transport_error && !http_error) {
    return MakeExtendError(
        ExtendStatusCode::StorageTransientNetwork,
        std::string("Native S3 request failed (outcome may be unknown): ") + aws_error_str(transport_error));
  }
  if (http_status >= 200 && http_status < 300)
    return arrow::Status::OK();
  auto code = ExtendStatusCode::AwsErrorNonRetryable;
  if (http_status == 404)
    code = ExtendStatusCode::AwsErrorNotFound;
  else if (http_status == 401 || http_status == 403)
    code = ExtendStatusCode::AwsErrorAccessDenied;
  else if (http_status == 409)
    code = ExtendStatusCode::AwsErrorConflict;
  else if (http_status == 412)
    code = ExtendStatusCode::AwsErrorPreConditionFailed;
  else if (http_status == 429)
    code = ExtendStatusCode::StorageTransientThrottling;
  else if (http_status >= 500)
    code = ExtendStatusCode::StorageTransientService;
  return MakeExtendError(code, "Native S3 request failed: HTTP " + std::to_string(http_status));
}

arrow::Result<std::shared_ptr<NativeS3Transport>> NativeS3Transport::Make(
    const S3Options& options, std::shared_ptr<S3ClientHolder> holder, std::shared_ptr<S3CrtClientHolder> crt_holder) {
  ARROW_RETURN_NOT_OK(CheckS3Initialized());
  if ((!options.cloud_provider.empty() && options.cloud_provider != "aws") || options.retry_strategy ||
      !options.proxy_options.host.empty() || options.credentials_kind == S3CredentialsKind::Role ||
      options.credentials_kind == S3CredentialsKind::WebIdentity) {
    return arrow::Status::NotImplemented(
        "Native S3 requires AWS/MinIO, explicit/default/anonymous credentials, "
        "no custom retry strategy and no proxy");
  }
  const auto* provider = options.credentials_provider.get();
  const bool standard_provider =
      provider && ((options.credentials_kind == S3CredentialsKind::Explicit &&
                    typeid(*provider) == typeid(Aws::Auth::SimpleAWSCredentialsProvider)) ||
                   (options.credentials_kind == S3CredentialsKind::Anonymous &&
                    typeid(*provider) == typeid(Aws::Auth::AnonymousAWSCredentialsProvider)) ||
                   (options.credentials_kind == S3CredentialsKind::Default &&
                    typeid(*provider) == typeid(Aws::Auth::DefaultAWSCredentialsProviderChain)));
  if (!standard_provider)
    return arrow::Status::NotImplemented("Native S3 custom credentials providers require an adapter");
  if (!options.max_connections || (options.scheme != "http" && options.scheme != "https")) {
    return arrow::Status::Invalid("Invalid native S3 connection options");
  }
  if (!crt_holder)
    return arrow::Status::NotImplemented("Native S3 requires the existing SDK CRT client");
  ARROW_ASSIGN_OR_RAISE(auto lease, crt_holder->Acquire());
  if (!lease.native_client())
    return arrow::Status::Invalid("SDK CRT client has no underlying native client");
  auto state = std::make_shared<State>();
  state->holder = std::move(holder);
  state->limit = options.max_connections;
  return std::shared_ptr<NativeS3Transport>(new NativeS3Transport(std::move(state), std::move(crt_holder)));
}

arrow::Future<NativeS3Response> NativeS3Transport::Send(const Aws::S3::S3Request& model,
                                                        const std::string& key,
                                                        Aws::Http::HttpMethod method,
                                                        const std::string& query,
                                                        const arrow::io::IOContext& io,
                                                        size_t limit,
                                                        std::shared_ptr<arrow::Buffer> data) {
  auto send = [&]() -> arrow::Result<arrow::Future<NativeS3Response>> {
    if (!io.executor())
      return arrow::Status::Invalid("Native S3 needs a caller executor");
    ARROW_RETURN_NOT_OK(io.stop_token().Poll());
    auto r = std::make_unique<Request>();
    r->state = state_;
    r->io = io;
    r->limit = limit;
    ARROW_ASSIGN_OR_RAISE(r->client_lease, holder_->Acquire());
    {
      std::lock_guard lock(state_->mutex);
      if (state_->inflight >= state_->limit)
        return arrow::Status::CapacityError("Native S3 request limit reached");
      ++state_->inflight;
      r->charged = true;
    }
    ARROW_ASSIGN_OR_RAISE(auto lease, state_->holder->Lock());
    auto endpoint = lease->accessEndpointProvider()->ResolveEndpoint(model.GetEndpointContextParams());
    if (!endpoint.IsSuccess())
      return arrow::Status::Invalid(endpoint.GetError().GetMessage().c_str());
    auto& resolved = endpoint.GetResult();
    if (!key.empty())
      resolved.AddPathSegments(key.c_str());
    if (!query.empty())
      resolved.SetQueryString(query.c_str());
    auto uri = resolved.GetURI();
    model.AddQueryStringParameters(uri);
    r->http = Aws::Http::CreateHttpRequest(uri, method, Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
    for (const auto& h : model.GetHeaders()) r->http->SetHeaderValue(h.first, h.second);
    for (const auto& h : model.GetAdditionalCustomHeaders()) r->http->SetHeaderValue(h.first, h.second);
    if (!data) {
      const auto payload = model.SerializePayload();
      if (!payload.empty())
        data = arrow::Buffer::FromString(std::string(payload.data(), payload.size()));
    }
    if (data) {
      r->data = std::move(data);
      r->streambuf = std::make_unique<Aws::Utils::Stream::PreallocatedStreamBuf>(
          reinterpret_cast<unsigned char*>(const_cast<uint8_t*>(r->data->data())), r->data->size());
      r->http->AddContentBody(Aws::MakeShared<Aws::IOStream>("native-s3-body", r->streambuf.get()));
      r->http->SetContentLength(std::to_string(r->data->size()).c_str());
    }
    r->message = r->http->ToCrtHttpRequest();
    const auto url = uri.GetURIString();
    auto cursor = aws_byte_cursor_from_array(url.data(), url.size());
    if (aws_uri_init_parse(&r->endpoint, aws_default_allocator(), &cursor)) {
      return arrow::Status::Invalid("Cannot parse native S3 endpoint");
    }
    aws_s3_meta_request_options options{};
    options.type = AWS_S3_META_REQUEST_TYPE_DEFAULT;
    options.operation_name = aws_byte_cursor_from_c_str(model.GetServiceRequestName());
    options.message = r->message->GetUnderlyingMessage();
    options.endpoint = &r->endpoint;
    options.user_data = r.get();
    options.headers_callback = Request::Headers;
    options.body_callback = Request::Body;
    options.telemetry_callback = Request::Telemetry;
    options.finish_callback = Request::Finish;
    options.shutdown_callback = Request::Shutdown;
    auto future = r->future;
    auto* pending = r.release();
    auto* meta = aws_s3_client_make_meta_request(pending->client_lease.native_client(), &options);
    if (!meta) {
      r.reset(pending);
      return arrow::Status::IOError("CRT rejected S3 request: ", aws_error_str(aws_last_error()));
    }
    return future;
  };
  try {
    auto result = send();
    if (!result.ok())
      return arrow::Future<NativeS3Response>::MakeFinished(result.status());
    return std::move(result).ValueUnsafe();
  } catch (const std::bad_alloc&) {
    return arrow::Future<NativeS3Response>::MakeFinished(arrow::Status::OutOfMemory("Native S3 request allocation"));
  } catch (const std::exception& e) {
    return arrow::Future<NativeS3Response>::MakeFinished(arrow::Status::Invalid(e.what()));
  }
}

}  // namespace milvus_storage
#endif
