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

package packed

/*
#include <stdlib.h>
#include "arrow/c/abi.h"
#include "arrow/c/helpers.h"
#include "milvus-storage/packed/reader_c.h"
#include "milvus-storage/packed/writer_c.h"
*/
import "C"

type PackedWriter struct {
	cPackedWriter C.CPackedWriter
}

type PackedReader struct {
	cPackedReader C.CPackedReader
	cSchema       CArrowSchema
}

type (
	CArrowSchema = C.struct_ArrowSchema
	CArrowArray  = C.struct_ArrowArray
)
