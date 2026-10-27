// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#include "filesystem/s3/native_s3_transport.h"
#ifdef WITH_CRT
#include <optional>
#include <utility>
#include <arrow/filesystem/path_util.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/standard/StandardHttpResponse.h>
#include <aws/crt/http/HttpRequestResponse.h>
#include <aws/common/uri.h>
#include <aws/s3/s3_client.h>
#include <aws/s3-crt/S3CrtErrorMarshaller.h>
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/util_internal.h"

namespace milvus_storage {
struct NativeS3Transport::State {
  bool is_gcp = false;
  std::string region;
  std::shared_ptr<S3ClientHolder> holder;
};

namespace {
struct Request {
  S3CrtClientLease client_lease;
  size_t limit;
  std::shared_ptr<Aws::Http::HttpRequest> http;
  std::shared_ptr<Aws::Crt::Http::HttpRequest> message;
  aws_uri endpoint{};
  std::shared_ptr<Aws::Http::HttpResponse> response;
  std::optional<Aws::Client::AWSError<Aws::Client::CoreErrors>> service_error;
  arrow::Status status;
  size_t received = 0;
  std::atomic<int> http_status{0};
  std::shared_ptr<FilesystemMetrics> write_metrics;
  int64_t write_bytes = 0;
  bool write_finished = false;
  arrow::Future<Aws::Client::XmlOutcome> future = arrow::Future<Aws::Client::XmlOutcome>::Make();

  ~Request() {
    message.reset();
    response.reset();
    http.reset();
    aws_uri_clean_up(&endpoint);
  }
  int AddHeaders(const aws_http_headers* headers) {
    for (size_t i = 0; i < aws_http_headers_count(headers); ++i) {
      aws_http_header h{};
      if (aws_http_headers_get_index(headers, i, &h))
        return AWS_OP_ERR;
      response->AddHeader(Aws::String(reinterpret_cast<char*>(h.name.ptr), h.name.len),
                          Aws::String(reinterpret_cast<char*>(h.value.ptr), h.value.len));
    }
    return AWS_OP_SUCCESS;
  }
  int AppendBody(const uint8_t* data, size_t size) {
    if (size > limit - received) {
      status = arrow::Status::CapacityError("S3 response exceeds configured limit");
      return aws_raise_error(AWS_ERROR_S3_CANCELED);
    }
    response->GetResponseBody().write(reinterpret_cast<const char*>(data), size);
    if (!response->GetResponseBody()) {
      status = arrow::Status::IOError("Cannot buffer S3 response");
      return aws_raise_error(AWS_ERROR_S3_CANCELED);
    }
    received += size;
    return AWS_OP_SUCCESS;
  }
  static int Headers(aws_s3_meta_request*, const aws_http_headers* headers, int status, void* user) noexcept {
    auto& r = *static_cast<Request*>(user);
    r.http_status.store(status);
    try {
      return r.AddHeaders(headers);
    } catch (...) {
      r.status = arrow::Status::OutOfMemory("S3 response headers");
      return aws_raise_error(AWS_ERROR_OOM);
    }
  }
  static int Body(aws_s3_meta_request*, const aws_byte_cursor* data, uint64_t, void* user) noexcept {
    auto& r = *static_cast<Request*>(user);
    try {
      return r.AppendBody(data->ptr, data->len);
    } catch (...) {
      r.status = arrow::Status::OutOfMemory("S3 response body");
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
    try {
      // CRT delivers failed responses here rather than through the body callback.
      if (result->error_response_body) {
        r.response = Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>("native-s3-response", r.http);
        r.received = 0;
        r.AppendBody(result->error_response_body->buffer, result->error_response_body->len);
      }
      if (result->error_response_headers && r.AddHeaders(result->error_response_headers) != AWS_OP_SUCCESS)
        r.status = arrow::Status::IOError("Cannot read S3 error response headers");
      r.response->SetResponseCode(static_cast<Aws::Http::HttpResponseCode>(r.http_status.load()));
      // CRT 0.12.6 discards the response body for these recoverable errors when
      // retries are exhausted. Recover the SDK error type from the CRT signal.
      const char* error_name = nullptr;
      switch (result->error_code) {
        case AWS_ERROR_S3_SLOW_DOWN:
          error_name = "SlowDown";
          break;
        case AWS_ERROR_S3_INTERNAL_ERROR:
          error_name = "InternalError";
          break;
        case AWS_ERROR_S3_REQUEST_TIMEOUT:
          error_name = "RequestTimeout";
          break;
        case AWS_ERROR_S3_REQUEST_TIME_TOO_SKEWED:
          error_name = "RequestTimeTooSkewed";
          break;
      }
      if (error_name && !result->error_response_body) {
        r.service_error = Aws::Client::S3CrtErrorMarshaller().FindErrorByName(error_name);
        r.service_error->SetExceptionName(error_name);
        r.service_error->SetMessage(Aws::String("CRT service error (outcome may be unknown): ") +
                                    aws_error_str(result->error_code));
        r.service_error->SetResponseCode(r.response->GetResponseCode());
        r.service_error->SetResponseHeaders(r.response->GetHeaders());
      }
      const bool is_service_error =
          r.service_error.has_value() || result->error_response_body ||
          (r.http_status.load() >= 300 &&
           (result->error_code == AWS_ERROR_S3_INVALID_RESPONSE_STATUS ||
            result->error_code == AWS_ERROR_S3_INTERNAL_ERROR || result->error_code == AWS_ERROR_S3_SLOW_DOWN));
      if (result->error_code && !is_service_error) {
        // An incomplete response cannot establish success or absence, even if
        // headers arrived. Keep the CRT diagnostic when using SDK error parsing.
        r.response->SetResponseCode(Aws::Http::HttpResponseCode::REQUEST_NOT_MADE);
        r.response->SetClientErrorType(Aws::Client::CoreErrors::NETWORK_CONNECTION);
        r.response->SetClientErrorMessage(Aws::String("Native S3 request failed (outcome may be unknown): ") +
                                          aws_error_str(result->error_code));
      }
    } catch (const std::bad_alloc&) {
      r.status = arrow::Status::OutOfMemory("S3 response completion");
    } catch (const std::exception& e) {
      r.status = arrow::Status::IOError(e.what());
    }
    // Release our initial reference. Shutdown, not Finish, is the point at which
    // CRT has stopped using all request buffers and invoking request callbacks.
    aws_s3_meta_request_release(meta);
  }
  arrow::Result<Aws::Client::XmlOutcome> Outcome() {
    if (!status.ok())
      return status;
    if (service_error)
      return Aws::Client::XmlOutcome(std::move(*service_error));
    auto outcome = client_lease->GenerateXmlOutcome(response);
    if (outcome.IsSuccess()) {
      auto root = outcome.GetResult().GetPayload().GetRootElement();
      if (!root.IsNull() && root.GetName() == "Error") {
        // CompleteMultipartUpload can return an error with HTTP 200. The SDK's
        // generic outcome conversion only checks HTTP/client errors.
        response->GetResponseBody().clear();
        response->GetResponseBody().seekg(0);
        auto error = Aws::Client::S3CrtErrorMarshaller().Marshall(*response);
        error.SetResponseHeaders(response->GetHeaders());
        error.SetResponseCode(response->GetResponseCode());
        return Aws::Client::XmlOutcome(std::move(error));
      }
    }
    return outcome;
  }
  static void Shutdown(void* user) noexcept {
    S3CrtCallbackScope callback_scope;
    std::unique_ptr<Request> owned(static_cast<Request*>(user));
    auto result = [&]() -> arrow::Result<Aws::Client::XmlOutcome> {
      try {
        return owned->Outcome();
      } catch (const std::bad_alloc&) {
        return arrow::Status::OutOfMemory("S3 response parsing");
      } catch (const std::exception& e) {
        return arrow::Status::IOError(e.what());
      }
    }();
    if (owned->write_metrics) {
      if (result.ok() && result->IsSuccess())
        owned->write_metrics->IncrementWriteBytes(owned->write_bytes);
      else
        owned->write_metrics->IncrementFailedCount();
      owned->write_finished = true;
    }
    // As with native range reads, let callers select their continuation executor.
    // The request lease remains alive until all inline continuations return.
    owned->future.MarkFinished(std::move(result));
  }
};
}  // namespace

arrow::Result<std::shared_ptr<NativeS3Transport>> NativeS3Transport::Make(
    const S3Options& options, std::shared_ptr<S3ClientHolder> holder, std::shared_ptr<S3CrtClientHolder> crt_holder) {
  ARROW_RETURN_NOT_OK(CheckS3Initialized());
  if (options.retry_strategy || !options.proxy_options.host.empty()) {
    return arrow::Status::NotImplemented("Native S3 requires no custom retry strategy and no proxy");
  }
  if (!options.max_connections || (options.scheme != "http" && options.scheme != "https")) {
    return arrow::Status::Invalid("Invalid native S3 connection options");
  }
  if (!crt_holder)
    return arrow::Status::NotImplemented("Native S3 requires the existing SDK CRT client");
  ARROW_ASSIGN_OR_RAISE(auto lease, crt_holder->Acquire());
  if (!lease->GetUnderlyingS3Client())
    return arrow::Status::Invalid("SDK CRT client has no underlying native client");
  auto state = std::make_shared<State>();
  state->holder = std::move(holder);
  state->is_gcp = options.cloud_provider == "gcp";
  state->region = options.region;
  return std::shared_ptr<NativeS3Transport>(new NativeS3Transport(std::move(state), std::move(crt_holder)));
}

arrow::Future<Aws::Client::XmlOutcome> NativeS3Transport::Send(const Aws::AmazonWebServiceRequest& model,
                                                               const std::string& key,
                                                               Aws::Http::HttpMethod method,
                                                               const std::string& query,
                                                               const arrow::io::IOContext& io,
                                                               size_t limit) {
  auto send = [&]() -> arrow::Result<arrow::Future<Aws::Client::XmlOutcome>> {
    if (method != Aws::Http::HttpMethod::HTTP_HEAD && method != Aws::Http::HttpMethod::HTTP_GET)
      return arrow::Status::NotImplemented("Native S3 transport only supports metadata reads");
    ARROW_RETURN_NOT_OK(io.stop_token().Poll());
    auto r = std::make_unique<Request>();
    r->limit = limit;
    ARROW_ASSIGN_OR_RAISE(r->client_lease, holder_->Acquire());
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
    r->response = Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>("native-s3-response", r->http);
    for (const auto& h : model.GetHeaders()) r->http->SetHeaderValue(h.first, h.second);
    for (const auto& h : model.GetAdditionalCustomHeaders()) r->http->SetHeaderValue(h.first, h.second);
    r->message = r->http->ToCrtHttpRequest();
    // The SDK conversion uses an absolute URI and drops '/' for bucket roots.
    // Direct CRT requests need an origin-form target, especially for virtual
    // hosted bucket listing where providers otherwise reject/ignore the query.
    auto target = uri.GetURLEncodedPath();
    if (target.empty())
      target = "/";
    target += uri.GetQueryString();
    if (!r->message->SetPath(Aws::Crt::ByteCursorFromCString(target.c_str())))
      return arrow::Status::Invalid("Cannot construct native S3 request target");
    const auto url = uri.GetURIString();
    auto cursor = aws_byte_cursor_from_array(url.data(), url.size());
    if (aws_uri_init_parse(&r->endpoint, aws_default_allocator(), &cursor)) {
      return arrow::Status::Invalid("Cannot parse native S3 endpoint");
    }
    aws_s3_meta_request_options options{};
    options.type = AWS_S3_META_REQUEST_TYPE_DEFAULT;
    options.operation_name = aws_byte_cursor_from_c_str(model.GetServiceRequestName());
    options.message = r->message->GetUnderlyingMessage();
    // Inherit the SDK client signing configuration and credentials provider.
    options.endpoint = &r->endpoint;
    options.user_data = r.get();
    options.headers_callback = Request::Headers;
    options.body_callback = Request::Body;
    options.telemetry_callback = Request::Telemetry;
    options.finish_callback = Request::Finish;
    options.shutdown_callback = Request::Shutdown;
    auto future = r->future;
    auto* pending = r.release();
    // CRT queues requests from all streams and schedules available connections.
    // A busy shared client must not turn an admitted stream write into a failure.
    auto* meta = aws_s3_client_make_meta_request(pending->client_lease->GetUnderlyingS3Client(), &options);
    if (!meta) {
      r.reset(pending);
      return arrow::Status::IOError("CRT rejected S3 request: ", aws_error_str(aws_last_error()));
    }
    return future;
  };
  try {
    auto result = send();
    if (!result.ok())
      return arrow::Future<Aws::Client::XmlOutcome>::MakeFinished(result.status());
    return std::move(result).ValueUnsafe();
  } catch (const std::bad_alloc&) {
    return arrow::Future<Aws::Client::XmlOutcome>::MakeFinished(
        arrow::Status::OutOfMemory("Native S3 request allocation"));
  } catch (const std::exception& e) {
    return arrow::Future<Aws::Client::XmlOutcome>::MakeFinished(arrow::Status::Invalid(e.what()));
  }
}

}  // namespace milvus_storage
#endif
