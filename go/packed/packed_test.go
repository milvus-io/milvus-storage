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
	"golang.org/x/exp/rand"
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
	path := "/tmp"
	bufferSize := 10 * 1024 * 1024 // 10MB
	pw, err := NewPackedWriter(path, schema, bufferSize)
	assert.NoError(t, err)
	for i := 0; i < batches; i++ {
		err = pw.WriteRecordBatch(rec)
		assert.NoError(t, err)
	}
	err = pw.Close()
	assert.NoError(t, err)

	reader, err := NewPackedReader(path, schema, bufferSize)
	assert.NoError(t, err)
	rr, err := reader.ReadNext()
	assert.NoError(t, err)
	defer rr.Release()
	assert.Equal(t, int64(3*batches), rr.NumRows())
}

func TestPackedMultiFiles(t *testing.T) {
	batches := 1000
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "a", Type: arrow.PrimitiveTypes.Int32},
		{Name: "b", Type: arrow.PrimitiveTypes.Int64},
		{Name: "c", Type: arrow.BinaryTypes.String},
	}, nil)

	b := array.NewRecordBuilder(memory.DefaultAllocator, schema)
	strLen := 1000
	arrLen := 30
	defer b.Release()
	for idx := range schema.Fields() {
		switch idx {
		case 0:
			values := make([]int32, arrLen)
			for i := 0; i < arrLen; i++ {
				values[i] = int32(i + 1)
			}
			b.Field(idx).(*array.Int32Builder).AppendValues(values, nil)
		case 1:
			values := make([]int64, arrLen)
			for i := 0; i < arrLen; i++ {
				values[i] = int64(i + 1)
			}
			b.Field(idx).(*array.Int64Builder).AppendValues(values, nil)
		case 2:
			values := make([]string, arrLen)
			for i := 0; i < arrLen; i++ {
				values[i] = randomString(strLen)
			}
			b.Field(idx).(*array.StringBuilder).AppendValues(values, nil)
		}
	}
	rec := b.NewRecord()
	defer rec.Release()
	path := "/tmp"
	bufferSize := 10 * 1024 * 1024 // 10MB
	pw, err := NewPackedWriter(path, schema, bufferSize)
	assert.NoError(t, err)
	for i := 0; i < batches; i++ {
		err = pw.WriteRecordBatch(rec)
		assert.NoError(t, err)
	}
	err = pw.Close()
	assert.NoError(t, err)

	reader, err := NewPackedReader(path, schema, bufferSize)
	assert.NoError(t, err)
	var rows int64 = 0
	var rr arrow.Record
	for {
		rr, err = reader.ReadNext()
		assert.NoError(t, err)
		if rr == nil {
			// end of file
			break
		}

		rows += rr.NumRows()
	}

	assert.Equal(t, int64(arrLen*batches), rows)
}

func randomString(length int) string {
	const charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	result := make([]byte, length)
	for i := range result {
		result[i] = charset[rand.Intn(len(charset))]
	}
	return string(result)
}
