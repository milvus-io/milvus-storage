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

#define LOON_SUCCESS 0
#define LOON_INVALID_ARGS 1
#define LOON_MEMORY_ERROR 2
#define LOON_ARROW_ERROR 3
#define LOON_LOGICAL_ERROR 4
#define LOON_GOT_EXCEPTION 5
#define LOON_UNREACHABLE_ERROR 6
#define LOON_INVALID_PROPERTIES 7
#define LOON_FAULT_INJECT_ERROR 8
#define LOON_NOT_SUPPORT 9
#define LOON_FILE_NOT_FOUND 12

// Shared with ExtendStatusCode. Keep values greater than arrow::StatusCode's current max value 45.
#define LOON_AWS_ERROR_NO_SUCH_UPLOAD 50
#define LOON_AWS_ERROR_CONFLICT 51
#define LOON_AWS_ERROR_PRECONDITION_FAILED 52
#define LOON_TRANSIENT_NETWORK 60
#define LOON_TRANSIENT_TIMEOUT 61
#define LOON_TRANSIENT_THROTTLING 62
#define LOON_TRANSIENT_SERVICE 63
#define LOON_TXN_EXHAUSTED_RETRY 70
#define LOON_TXN_RESOLUTION_FAILED 71
