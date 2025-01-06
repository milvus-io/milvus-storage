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

import (
	"testing"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/apache/arrow/go/v12/arrow/memory"
	"github.com/stretchr/testify/assert"
)

func TestPackedOneFile(t *testing.T) {
	batches := 100
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "a", Type: arrow.PrimitiveTypes.Int32},
		{Name: "b", Type: arrow.PrimitiveTypes.Int64},
		{Name: "c", Type: arrow.BinaryTypes.String},
	}, nil)

	b := array.NewRecordBuilder(memory.DefaultAllocator, schema)
	defer b.Release()
	for idx := range schema.Fields() {
		switch idx {
		case 0:
			b.Field(idx).(*array.Int32Builder).AppendValues(
				[]int32{int32(1), int32(2), int32(3)}, nil,
			)
		case 1:
			b.Field(idx).(*array.Int64Builder).AppendValues(
				[]int64{int64(4), int64(5), int64(6)}, nil,
			)
		case 2:
			b.Field(idx).(*array.StringBuilder).AppendValues(
				[]string{"a", "b", "c"}, nil,
			)
		}
	}
	rec := b.NewRecord()
	defer rec.Release()
	path := "/tmp/0"
	bufferSize := 10 * 1024 * 1024 // 10MB
	pw, err := newPackedWriter(path, schema, bufferSize)
	assert.NoError(t, err)
	for i := 0; i < batches; i++ {
		err = pw.writeRecordBatch(rec)
		assert.NoError(t, err)
	}
	mapping, err := pw.close()
	assert.NoError(t, err)
	assert.Equal(t, mapping["a"], ColumnOffset{0, 0})
	assert.Equal(t, mapping["b"], ColumnOffset{0, 1})
	assert.Equal(t, mapping["c"], ColumnOffset{0, 2})

	reader, err := Open(path, schema, 10*1024*1024 /* 10MB */)
	assert.NoError(t, err)
	rr, err := reader.Read()
	assert.NoError(t, err)
	defer rr.Release()
	assert.Equal(t, int64(3*batches), rr.NumRows())
}
