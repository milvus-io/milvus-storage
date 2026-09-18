// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#ifdef WITH_CRT
#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/util/future.h>
#include <aws/core/http/HttpTypes.h>
#include <aws/s3/S3Request.h>
#include "milvus-storage/filesystem/s3/s3_client.h"

namespace milvus_storage {

struct NativeS3Response {
  int http_status = 0;
  int transport_error = 0;
  arrow::Status status;
  std::string body;
  Aws::Http::HeaderValueCollection headers;
  arrow::Status ToStatus() const;
  bool HasHttpStatus(int code) const;
};

class NativeS3Transport {
  public:
  struct State;
  ~NativeS3Transport();
  static arrow::Result<std::shared_ptr<NativeS3Transport>> Make(const S3Options& options,
                                                                std::shared_ptr<S3ClientHolder> holder);
  // DEFAULT meta requests never transform PUT into multipart. All bodies are
  // owned memory; no CRT callback reads a blocking file/iostream data source.
  arrow::Future<NativeS3Response> Send(const Aws::S3::S3Request& request,
                                       const std::string& key,
                                       Aws::Http::HttpMethod method,
                                       const std::string& query,
                                       const arrow::io::IOContext& io_context,
                                       size_t response_limit = 16 * 1024 * 1024,
                                       std::shared_ptr<arrow::Buffer> body = nullptr);

  private:
  explicit NativeS3Transport(std::shared_ptr<State> state) : state_(std::move(state)) {}
  std::shared_ptr<State> state_;
};

// Called before Aws::ShutdownAPI; ordinary transport destruction only initiates
// nonblocking release. This barrier drains requests, completions and native clients.
void FinalizeNativeS3Transports();
}  // namespace milvus_storage
#endif
