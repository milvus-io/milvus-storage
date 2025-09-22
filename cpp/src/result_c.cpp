// Copyright 2023 Zilliz
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

#include "milvus-storage/result_c.h"
#include "milvus-storage/result_internal.h"

#include <string.h>
#include <cassert>

std::string error_to_string(int code) {
  static std::string error_strings[] = {"Success",                   // NOLINT
                                        "Invalid args",              //
                                        "Memory allocation failed",  //
                                        "Internal error",            //
                                        "Logical error",             //
                                        "Got exception",             //
                                        "Unreachable code"};
  static_assert(sizeof(error_strings) / sizeof((error_strings)[0]) == LOON_ERRORCODE_MAX);

  if (code < LOON_SUCCESS || code >= LOON_ERRORCODE_MAX) {
    return "Unknown error(undefined)";
  }

  return error_strings[code];
}

int IsSuccess(FFIResult* result) {
  assert(result);
  return result->err_code == LOON_SUCCESS;
}

const char* GetErrorMessage(FFIResult* result) {
  assert(result);
  if (IsSuccess(result)) {
    return NULL;
  }
  return result->message;
}

void FreeFFIResult(FFIResult* result) {
  assert(result);
  free(result->message);
}
