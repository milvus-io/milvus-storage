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
#include "milvus-storage/packed/column_offset_mapping_c.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type ColumnOffset struct {
	PathIndex int
	ColIndex  int
}

func GetColumnOffsetMappingKeys(cColumnOffsetMapping C.CColumnOffsetMapping) ([]string, error) {
	size := int(C.GetColumnOffsetMappingSize(cColumnOffsetMapping))
	if size == 0 {
		return nil, fmt.Errorf("ColumnOffsetMapping is empty")
	}
	keys := make([]unsafe.Pointer, size)

	C.GetColumnOffsetMappingKeys(cColumnOffsetMapping, unsafe.Pointer(&keys[0]))
	ret := make([]string, size)
	for i := 0; i < size; i++ {
		ret[i] = C.GoString((*C.char)(keys[i]))
	}
	return ret, nil
}

func GetColumnOffset(cColumnOffsetMapping C.CColumnOffsetMapping, key string) (ColumnOffset, error) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	var cPathIndex, cColIndex C.int

	status := C.GetColumnOffset(cColumnOffsetMapping, cKey, &cPathIndex, &cColIndex)
	if status != 0 {
		return ColumnOffset{-1, -1}, fmt.Errorf("GetColumnOffset from key %s failed", key)
	}
	return ColumnOffset{int(cPathIndex), int(cColIndex)}, nil
}
