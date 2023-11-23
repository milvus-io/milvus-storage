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

package storage_test

import (
	"sync"
	"testing"

	"github.com/milvus-io/milvus-storage/go/storage/options"
	"github.com/milvus-io/milvus-storage/go/storage/schema"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/apache/arrow/go/v12/arrow/memory"
	"github.com/milvus-io/milvus-storage/go/filter"
	"github.com/milvus-io/milvus-storage/go/storage"
	"github.com/milvus-io/milvus-storage/go/storage/lock"
	"github.com/stretchr/testify/suite"
)

type SpaceTestSuite struct {
	suite.Suite
}

func createSchema() *schema.Schema {
	pkField := arrow.Field{
		Name:     "pk_field",
		Type:     arrow.DataType(&arrow.Int64Type{}),
		Nullable: false,
	}
	vsField := arrow.Field{
		Name:     "vs_field",
		Type:     arrow.DataType(&arrow.Int64Type{}),
		Nullable: false,
	}
	vecField := arrow.Field{
		Name:     "vec_field",
		Type:     arrow.DataType(&arrow.FixedSizeBinaryType{ByteWidth: 10}),
		Nullable: false,
	}
	columnField := arrow.Field{
		Name:     "column_field",
		Type:     arrow.DataType(&arrow.Int64Type{}),
		Nullable: false,
	}
	fields := []arrow.Field{pkField, vsField, vecField, columnField}

	as := arrow.NewSchema(fields, nil)
	schemaOptions := &schema.SchemaOptions{
		PrimaryColumn: "pk_field",
		VersionColumn: "vs_field",
		VectorColumn:  "vec_field",
	}

	sc := schema.NewSchema(as, schemaOptions)
	return sc
}

func recordReader() array.RecordReader {
	pkBuilder := array.NewInt64Builder(memory.DefaultAllocator)
	pkBuilder.AppendValues([]int64{1, 2, 3}, nil)
	pkArr := pkBuilder.NewArray()

	vsBuilder := array.NewInt64Builder(memory.DefaultAllocator)
	vsBuilder.AppendValues([]int64{1, 2, 3}, nil)
	vsArr := vsBuilder.NewArray()

	vecBuilder := array.NewFixedSizeBinaryBuilder(memory.DefaultAllocator, &arrow.FixedSizeBinaryType{ByteWidth: 10})
	vecBuilder.AppendValues([][]byte{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	}, nil)
	vecArr := vecBuilder.NewArray()

	columnBuilder := array.NewInt64Builder(memory.DefaultAllocator)
	columnBuilder.AppendValues([]int64{1, 2, 3}, nil)
	columnArr := columnBuilder.NewArray()

	arrs := []arrow.Array{pkArr, vsArr, vecArr, columnArr}

	rec := array.NewRecord(createSchema().Schema(), arrs, 3)
	recReader, err := array.NewRecordReader(createSchema().Schema(), []arrow.Record{rec})
	if err != nil {
		panic(err)
	}
	return recReader
}

func deleteRecordReader() array.RecordReader {
	pkField := arrow.Field{
		Name:     "pk_field",
		Type:     arrow.DataType(&arrow.Int64Type{}),
		Nullable: false,
	}
	vsField := arrow.Field{
		Name:     "vs_field",
		Type:     arrow.DataType(&arrow.Int64Type{}),
		Nullable: false,
	}

	deleteArrowSchema := arrow.NewSchema([]arrow.Field{pkField, vsField}, nil)

	deletePkBuilder := array.NewInt64Builder(memory.DefaultAllocator)
	deletePkBuilder.AppendValues([]int64{1}, nil)
	deletePkArr := deletePkBuilder.NewArray()

	deleteVsBuilder := array.NewInt64Builder(memory.DefaultAllocator)
	deleteVsBuilder.AppendValues([]int64{1}, nil)
	deleteVsArr := deleteVsBuilder.NewArray()

	deleteArray := []arrow.Array{deletePkArr, deleteVsArr}
	rec := array.NewRecord(deleteArrowSchema, deleteArray, 1)
	recReader, err := array.NewRecordReader(deleteArrowSchema, []arrow.Record{rec})
	if err != nil {
		panic(err)
	}
	return recReader
}

func (suite *SpaceTestSuite) TestSpaceReadWrite() {
	sc := createSchema()
	err := sc.Validate()
	suite.NoError(err)

	opts := options.NewSpaceOptionBuilder().SetSchema(sc).SetVersion(0).Build()

	space, err := storage.Open("file:///"+suite.T().TempDir(), opts)
	suite.NoError(err)

	writeOpt := &options.WriteOptions{MaxRecordPerFile: 1000}
	err = space.Write(recordReader(), writeOpt)
	suite.NoError(err)

	f := filter.NewConstantFilter(filter.Equal, "pk_field", int64(1))
	readOpt := options.NewReadOptions()
	readOpt.AddFilter(f)
	readOpt.AddColumn("pk_field")
	readReader, err := space.Read(readOpt)
	suite.NoError(err)
	var resVals []int64
	for readReader.Next() {
		rec := readReader.Record()
		cols := rec.Columns()
		values := cols[0].(*array.Int64).Int64Values()
		resVals = append(resVals, values...)
	}

	suite.ElementsMatch([]int64{1}, resVals)
}

func (suite *SpaceTestSuite) TestSpaceReadWriteConcurrency() {
	sc := createSchema()
	err := sc.Validate()
	suite.NoError(err)

	opts := options.Options{
		Version:     0,
		LockManager: lock.NewMemoryLockManager(),
		Schema:      sc,
	}

	space, err := storage.Open("file:///"+suite.T().TempDir(), opts)
	suite.NoError(err)

	writeOpt := &options.WriteOptions{MaxRecordPerFile: 1000}

	wg := sync.WaitGroup{}
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			err = space.Write(recordReader(), writeOpt)
			wg.Done()
		}()
	}

	wg.Wait()
}

func (suite *SpaceTestSuite) TestSpaceDelete() {
	sc := createSchema()
	err := sc.Validate()
	suite.NoError(err)

	opts := options.NewSpaceOptionBuilder().SetSchema(sc).SetVersion(0).Build()

	space, err := storage.Open("file:///"+suite.T().TempDir(), opts)
	suite.NoError(err)

	err = space.Delete(deleteRecordReader())
	suite.NoError(err)
}

func (suite *SpaceTestSuite) TestSpaceReadWithFilter() {
	sc := createSchema()
	err := sc.Validate()
	suite.NoError(err)

	opts := options.NewSpaceOptionBuilder().SetSchema(sc).SetVersion(0).Build()

	space, err := storage.Open("file:///"+suite.T().TempDir(), opts)
	suite.NoError(err)

	writeOpt := &options.WriteOptions{MaxRecordPerFile: 1000}
	err = space.Write(recordReader(), writeOpt)
	suite.NoError(err)

	f := filter.NewConstantFilter(filter.Equal, "pk_field", int64(1))
	readOpt := options.NewReadOptions()
	readOpt.AddFilter(f)
	readOpt.AddColumn("pk_field")
	readReader, err := space.Read(readOpt)
	suite.NoError(err)
	var resValues []int64
	for readReader.Next() {
		rec := readReader.Record()
		cols := rec.Columns()
		values := cols[0].(*array.Int64).Int64Values()
		resValues = append(resValues, values...)
	}
	suite.ElementsMatch([]int64{1}, resValues)

	f = filter.NewConstantFilter(filter.GreaterThan, "pk_field", int64(1))
	readOpt = options.NewReadOptions()
	readOpt.AddFilter(f)
	readOpt.AddColumn("pk_field")
	readReader, err = space.Read(readOpt)
	suite.NoError(err)
	resValues = []int64{}
	for readReader.Next() {
		rec := readReader.Record()
		cols := rec.Columns()
		values := cols[0].(*array.Int64).Int64Values()
		resValues = append(resValues, values...)
	}
	suite.ElementsMatch([]int64{2, 3}, resValues)

	f = filter.NewConstantFilter(filter.NotEqual, "pk_field", int64(1))
	readOpt = options.NewReadOptions()
	readOpt.AddFilter(f)
	readOpt.AddColumn("pk_field")
	readReader, err = space.Read(readOpt)
	suite.NoError(err)
	resValues = []int64{}
	for readReader.Next() {
		rec := readReader.Record()
		cols := rec.Columns()
		values := cols[0].(*array.Int64).Int64Values()
		resValues = append(resValues, values...)
	}
	suite.ElementsMatch([]int64{2, 3}, resValues)

	f = filter.NewConstantFilter(filter.LessThan, "pk_field", int64(1))
	readOpt = options.NewReadOptions()
	readOpt.AddFilter(f)
	readOpt.AddColumn("pk_field")
	readReader, err = space.Read(readOpt)
	suite.NoError(err)
	resValues = []int64{}
	for readReader.Next() {
		rec := readReader.Record()
		cols := rec.Columns()
		values := cols[0].(*array.Int64).Int64Values()
		resValues = append(resValues, values...)
	}
	suite.ElementsMatch([]int64{}, resValues)

	f = filter.NewConstantFilter(filter.LessThan, "pk_field", int64(1))
	readOpt = options.NewReadOptions()
	readOpt.AddFilter(f)
	readOpt.AddColumn("pk_field")
	readReader, err = space.Read(readOpt)
	suite.NoError(err)
	resValues = []int64{}
	for readReader.Next() {
		rec := readReader.Record()
		cols := rec.Columns()
		values := cols[0].(*array.Int64).Int64Values()
		resValues = append(resValues, values...)
	}
	suite.ElementsMatch([]int64{}, resValues)

	f = filter.NewConstantFilter(filter.LessThanOrEqual, "pk_field", int64(1))
	readOpt = options.NewReadOptions()
	readOpt.AddFilter(f)
	readOpt.AddColumn("pk_field")
	readReader, err = space.Read(readOpt)
	suite.NoError(err)
	resValues = []int64{}
	for readReader.Next() {
		rec := readReader.Record()
		cols := rec.Columns()
		values := cols[0].(*array.Int64).Int64Values()
		resValues = append(resValues, values...)
	}
	suite.ElementsMatch([]int64{1}, resValues)

	f = filter.NewConstantFilter(filter.GreaterThanOrEqual, "pk_field", int64(1))
	readOpt = options.NewReadOptions()
	readOpt.AddFilter(f)
	readOpt.AddColumn("pk_field")
	readReader, err = space.Read(readOpt)
	suite.NoError(err)
	resValues = []int64{}
	for readReader.Next() {
		rec := readReader.Record()
		cols := rec.Columns()
		values := cols[0].(*array.Int64).Int64Values()
		resValues = append(resValues, values...)
	}
	suite.ElementsMatch([]int64{1, 2, 3}, resValues)

	f = filter.NewConstantFilter(filter.GreaterThan, "pk_field", int64(2))
	readOpt = options.NewReadOptions()
	readOpt.AddFilter(f)
	readOpt.AddColumn("pk_field")
	readReader, err = space.Read(readOpt)
	suite.NoError(err)
	resValues = []int64{}
	for readReader.Next() {
		rec := readReader.Record()
		cols := rec.Columns()
		values := cols[0].(*array.Int64).Int64Values()
		resValues = append(resValues, values...)
	}
	suite.ElementsMatch([]int64{3}, resValues)
}

func TestSpaceTestSuite(t *testing.T) {
	suite.Run(t, new(SpaceTestSuite))
}
