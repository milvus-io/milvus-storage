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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <arrow/c/abi.h>

typedef void* CPackedReader;
typedef void* CArrowArray;
typedef void* CArrowSchema;

int Open(const char* path, struct ArrowSchema* schema, const int64_t buffer_size, struct ArrowArrayStream* out);

int NewPackedReader(const char* path,
                    struct ArrowSchema* schema,
                    const int64_t buffer_size,
                    CPackedReader* c_packed_reader);

int ReadNext(CPackedReader c_packed_reader, CArrowArray* out_array, CArrowSchema* out_schema);

int CloseReader(CPackedReader c_packed_reader);

#ifdef __cplusplus
}
#endif