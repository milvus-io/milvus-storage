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

#include "milvus-storage/filesystem/metrics/error_classify.h"

#include <arrow/status.h>

namespace milvus_storage {

OpStatus ClassifyHttpStatus(int http_code) {
  switch (http_code) {
    case 404:
      return OpStatus::NotFound;
    case 403:
      return OpStatus::Auth;
    case 429:
    case 503:
      return OpStatus::Throttled;
    case 408:
    case 504:
      return OpStatus::Timeout;
    default:
      break;
  }
  if (http_code >= 500 && http_code < 600) {
    return OpStatus::ServerError;
  }
  if (http_code >= 400 && http_code < 500) {
    return OpStatus::ClientError;
  }
  return OpStatus::Unknown;
}

OpStatus ClassifyArrowStatus(const arrow::Status& s) {
  if (s.ok()) {
    return OpStatus::Ok;
  }
  if (s.IsInvalid() || s.IsTypeError()) {
    return OpStatus::ClientError;
  }
  return OpStatus::Unknown;
}

OpStatus ClassifyS3(int http_code, bool is_throttle, bool is_timeout, bool is_connect) {
  if (is_throttle) {
    return OpStatus::Throttled;
  }
  if (is_timeout) {
    return OpStatus::Timeout;
  }
  if (is_connect) {
    return OpStatus::Network;
  }
  return ClassifyHttpStatus(http_code);
}

}  // namespace milvus_storage
