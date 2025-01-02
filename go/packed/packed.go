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
#include "milvus-storage/packed/reader_c.h"
#include "milvus-storage/packed/writer_c.h"
#include "arrow/c/abi.h"
#include "arrow/c/helpers.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/arrio"
	"github.com/apache/arrow/go/v12/arrow/cdata"
)

type PackedWriter struct {
	cPackedWriter C.CPackedWriter
}

type (
	ColumnOffsetMapping = C.CColumnOffsetMapping
	CArrowSchema        = C.struct_ArrowSchema
	CArrowArray         = C.struct_ArrowArray
)

func newPackedWriter(path string, schema *arrow.Schema, bufferSize int) (*PackedWriter, error) {
	var cas cdata.CArrowSchema
	cdata.ExportArrowSchema(schema, &cas)
	cSchema := (*C.struct_ArrowSchema)(unsafe.Pointer(&cas))

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cBufferSize := C.int64_t(bufferSize)

	var cPackedWriter C.CPackedWriter
	status := C.NewPackedWriter(cPath, cSchema, cBufferSize, &cPackedWriter)
	if status != 0 {
		return nil, errors.New(fmt.Sprintf("failed to open file: %s, status: %d", path, status))
	}
	return &PackedWriter{cPackedWriter: cPackedWriter}, nil
}

func (pw *PackedWriter) writeRecordBatch(recordBatch arrow.Record) error {
	var caa cdata.CArrowArray
	var cas cdata.CArrowSchema

	cdata.ExportArrowRecordBatch(recordBatch, &caa, &cas)

	cArr := (*C.struct_ArrowArray)(unsafe.Pointer(&caa))
	cSchema := (*C.struct_ArrowSchema)(unsafe.Pointer(&cas))

	status := C.WriteRecordBatch(pw.cPackedWriter, cArr, cSchema)
	if status != 0 {
		return errors.New("PackedWriter: failed to write record batch")
	}

	return nil
}

func (pw *PackedWriter) close() (ColumnOffsetMapping, error) {
	var columnOffsetMapping ColumnOffsetMapping
	status := C.Close(pw.cPackedWriter, &columnOffsetMapping)
	if status != 0 {
		return columnOffsetMapping, errors.New("PackedWriter: failed to close file")
	}
	return columnOffsetMapping, nil
}

func Open(path string, schema *arrow.Schema, bufferSize int) (arrio.Reader, error) {
	// var cSchemaPtr uintptr
	// cSchema := cdata.SchemaFromPtr(cSchemaPtr)
	var cas cdata.CArrowSchema
	cdata.ExportArrowSchema(schema, &cas)
	casPtr := (*C.struct_ArrowSchema)(unsafe.Pointer(&cas))
	var cass cdata.CArrowArrayStream

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	status := C.Open(cPath, casPtr, C.int64_t(bufferSize), (*C.struct_ArrowArrayStream)(unsafe.Pointer(&cass)))
	if status != 0 {
		return nil, errors.New(fmt.Sprintf("failed to open file: %s, status: %d", path, status))
	}
	reader := cdata.ImportCArrayStream((*cdata.CArrowArrayStream)(unsafe.Pointer(&cass)), schema)
	return reader, nil
}
