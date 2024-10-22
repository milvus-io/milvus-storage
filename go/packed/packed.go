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
#cgo LDFLAGS: -v -lmilvus-storage -Wl,-rpath,@executable_path/

#include <stdlib.h>
#include "milvus-storage/packed/reader_c.h"
#include "arrow/c/abi.h"
#include "arrow/c/helpers.h"
*/
import "C"
import (
	"errors"
	"unsafe"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/arrio"
	"github.com/apache/arrow/go/v12/arrow/cdata"
)

type Reader struct {
	reader arrio.Reader
}

func Open(path string, schema *arrow.Schema, bufferSize int) (arrio.Reader, error) {
	// var cSchemaPtr uintptr
	// cSchema := cdata.SchemaFromPtr(cSchemaPtr)
	var cas cdata.CArrowSchema
	cdata.ExportArrowSchema(schema, &cas)
	casPtr := (*C.struct_ArrowSchema)(unsafe.Pointer(&cas))
	var caasPtr uintptr

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	status := C.Open(cPath, casPtr, C.int64_t(bufferSize), (*C.struct_ArrowArrayStream)(unsafe.Pointer(caasPtr)))
	if status != 0 {
		return nil, errors.New("failed to open")
	}

	reader := cdata.ImportCArrayStream((*cdata.CArrowArrayStream)(unsafe.Pointer(caasPtr)), schema)
	return reader, nil
}

func (r *Reader) Read() (arrow.Record, error) {
	return r.reader.Read()
}
