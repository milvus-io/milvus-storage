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

#include "milvus-storage/filesystem/metrics/op_type.h"

namespace arrow {
class Status;
}

namespace milvus_storage {

// Map an HTTP status code to an OpStatus.
OpStatus ClassifyHttpStatus(int http_code);

// Map an arrow::Status to an OpStatus (arrow's LocalFileSystem loses errno
// detail, so most failures classify as Unknown).
OpStatus ClassifyArrowStatus(const arrow::Status& s);

// Pure S3 mapping, testable without constructing an Aws::S3::S3Error.
// Precedence: throttle > timeout > connect > http-code mapping.
OpStatus ClassifyS3(int http_code, bool is_throttle, bool is_timeout, bool is_connect);

}  // namespace milvus_storage
