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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef RESULT_C
#define RESULT_C

#define LOON_SUCCESS 0
#define LOON_INVALID_ARGS 1
#define LOON_MEMORY_ERROR 2
#define LOON_ARROW_ERROR 3
#define LOON_LOGICAL_ERROR 4
#define LOON_GOT_EXCEPTION 5
#define LOON_UNREACHABLE_ERROR 6
#define LOON_INVALID_PROPERTIES 7
#define LOON_ERRORCODE_MAX 8

// usage example(caller must free the message string):
//
// FFIResult result = SomeFFIFunction(...);
// if (!IsSuccess(&result)) {
//    printf("Error: %s\n", GetErrorMessage(&result));
//    ... // handle error, e.g. log result.message
//    FreeFFIResult(&result); // free the message string
// }
typedef struct ffi_result {
  int err_code;
  char* message;
} FFIResult;

// check result is success
int IsSuccess(FFIResult* result);

// get the error message, return NULL if success
const char* GetErrorMessage(FFIResult* result);

// free the message string inside FFIResult
void FreeFFIResult(FFIResult* result);

#endif  // RESULT_C

#ifdef __cplusplus
}
#endif
