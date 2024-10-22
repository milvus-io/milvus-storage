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
	"os"
	"testing"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/apache/arrow/go/v12/arrow/memory"
	"github.com/apache/arrow/go/v12/parquet"
	"github.com/apache/arrow/go/v12/parquet/pqarrow"
	"github.com/stretchr/testify/assert"
)

func TestRead(t *testing.T) {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "a", Type: arrow.PrimitiveTypes.Int64},
		{Name: "b", Type: arrow.BinaryTypes.String},
	}, nil)

	b := array.NewRecordBuilder(memory.DefaultAllocator, schema)
	defer b.Release()
	for idx := range schema.Fields() {
		switch idx {
		case 0:
			b.Field(idx).(*array.Int64Builder).AppendValues(
				[]int64{int64(1), int64(2), int64(3)}, nil,
			)
		case 1:
			b.Field(idx).(*array.StringBuilder).AppendValues(
				[]string{"a", "b", "c"}, nil,
			)
		}
	}
	rec := b.NewRecord()

	path := "/tmp/test.parquet"
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE, 0666)
	assert.NoError(t, err)
	writer, err := pqarrow.NewFileWriter(schema, file, parquet.NewWriterProperties(), pqarrow.DefaultWriterProps())
	assert.NoError(t, err)
	err = writer.Write(rec)
	assert.NoError(t, err)
	err = writer.Close()
	assert.NoError(t, err)

	reader, err := Open(path, schema, 10*1024*1024 /* 10MB */)
	assert.NoError(t, err)

	rr, err := reader.Read()
	assert.NoError(t, err)
	defer rr.Release()
	assert.Equal(t, rec.NumRows(), rr.NumRows())
}
