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
#include "arrow/c/abi.h"
#include "arrow/c/helpers.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/cdata"
)

func newPackedReader(path string, schema *arrow.Schema, bufferSize int, opt *packedReaderOption) (*PackedReader, error) {
	var cas cdata.CArrowSchema
	cdata.ExportArrowSchema(schema, &cas)
	cSchema := (*C.struct_ArrowSchema)(unsafe.Pointer(&cas))

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cBufferSize := C.int64_t(bufferSize)

	var cPackedReader C.CPackedReader
	cNeedColumns := (*C.int)(C.malloc(C.size_t(len(opt.needColumns)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	for i, col := range opt.needColumns {
		(*[1<<31 - 1]C.int)(unsafe.Pointer(cNeedColumns))[i] = C.int(col)
	}
	cNumNeededColumns := C.int(len(opt.needColumns))
	status := C.NewPackedReader(cPath, cSchema, cBufferSize, cNeedColumns, cNumNeededColumns, &cPackedReader)
	if status != 0 {
		return nil, errors.New(fmt.Sprintf("failed to new packed reader: %s, status: %d", path, status))
	}
	return &PackedReader{cPackedReader: cPackedReader, schema: schema}, nil
}

func (pr *PackedReader) readNext() (arrow.Record, error) {
	var cArr C.CArrowArray
	var cSchema C.CArrowSchema
	status := C.ReadNext(pr.cPackedReader, &cArr, &cSchema)
	if status != 0 {
		return nil, fmt.Errorf("ReadNext failed with error code %d", status)
	}

	if cArr == nil {
		return nil, nil // end of stream, no more records to read
	}

	// Convert ArrowArray to Go RecordBatch using cdata
	goCArr := (*cdata.CArrowArray)(unsafe.Pointer(cArr))
	goCSchema := (*cdata.CArrowSchema)(unsafe.Pointer(cSchema))
	recordBatch, err := cdata.ImportCRecordBatch(goCArr, goCSchema)
	if err != nil {
		return nil, fmt.Errorf("failed to convert ArrowArray to Record: %w", err)
	}

	// Return the RecordBatch as an arrow.Record
	return recordBatch, nil
}

func (pr *PackedReader) close() error {
	status := C.CloseReader(pr.cPackedReader)
	if status != 0 {
		return errors.New("PackedReader: failed to close file")
	}
	return nil
}
