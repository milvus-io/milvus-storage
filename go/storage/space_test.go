package storage_test

import (
	"github.com/milvus-io/milvus-storage-format/storage/options"
	"github.com/milvus-io/milvus-storage-format/storage/schema"
	"testing"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/apache/arrow/go/v12/arrow/memory"
	"github.com/milvus-io/milvus-storage-format/filter"
	"github.com/milvus-io/milvus-storage-format/storage"
	"github.com/stretchr/testify/suite"
)

type DefaultSpaceTestSuite struct {
	suite.Suite
}

type SpaceTestSuite struct {
	suite.Suite
}

func (suite *SpaceTestSuite) TestSpaceReadWrite() {
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
		Type:     arrow.DataType(&arrow.FixedSizeBinaryType{ByteWidth: 16}),
		Nullable: false,
	}
	fields := []arrow.Field{pkField, vsField, vecField}

	as := arrow.NewSchema(fields, nil)
	schemaOptions := &options.SchemaOptions{
		PrimaryColumn: "pk_field",
		VersionColumn: "vs_field",
		VectorColumn:  "vec_field",
	}

	sc := schema.NewSchema(as, schemaOptions)
	validate := sc.Validate()
	if !validate.IsOK() {
		panic(validate.Msg())
	}

	pkBuilder := array.NewInt64Builder(memory.DefaultAllocator)
	pkBuilder.AppendValues([]int64{1, 2, 3, 4, 5, 6, 7, 8, 9}, nil)
	pkArr := pkBuilder.NewArray()

	vsBuilder := array.NewInt64Builder(memory.DefaultAllocator)
	vsBuilder.AppendValues([]int64{1, 2, 3, 4, 5, 6, 7, 8, 9}, nil)
	vsArr := vsBuilder.NewArray()

	vecBuilder := array.NewFixedSizeBinaryBuilder(memory.DefaultAllocator, &arrow.FixedSizeBinaryType{ByteWidth: 16})
	vecBuilder.AppendValues([][]byte{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
	}, nil)
	vecArr := vecBuilder.NewArray()

	arrs := []arrow.Array{pkArr, vsArr, vecArr}

	rec := array.NewRecord(as, arrs, 9)
	recReader, err := array.NewRecordReader(as, []arrow.Record{rec, rec, rec})
	if err != nil {
		panic(err)
	}

	space := storage.NewDefaultSpace(sc, &options.Options{Uri: suite.T().TempDir()})
	writeOpt := &options.WriteOptions{MaxRecordPerFile: 1000}
	err = space.Write(recReader, writeOpt)
	if err != nil {
		panic(err)
	}
}

func (suite *DefaultSpaceTestSuite) TestSpaceReadWrite() {
	field := arrow.Field{
		Name:     "int32",
		Type:     arrow.PrimitiveTypes.Int32,
		Nullable: false,
	}

	schem := arrow.NewSchema([]arrow.Field{field}, nil)
	builder := array.NewInt32Builder(memory.DefaultAllocator)
	builder.AppendValues([]int32{1, 3, 4, 5, 6, 8, 2, 4, 6}, nil)
	arr := builder.NewArray()
	rec := array.NewRecord(schem, []arrow.Array{arr}, int64(arr.Len()))
	recReader, err := array.NewRecordReader(schem, []arrow.Record{rec})
	suite.NoError(err)

	space := storage.NewReferenceSpace(schem, &options.SpaceOptions{Fs: options.InMemory})
	writeOpt := &options.WriteOptions{MaxRecordPerFile: 10}
	space.Write(recReader, writeOpt)

	f := filter.NewConstantFilter(filter.GreaterThan, int32(3))
	res, err := space.Read(&options.ReadOptions{Filters: map[string]filter.Filter{"int32": f}})
	suite.NoError(err)

	var resVals []int32
	for res.Next() {
		rec := res.Record()
		cols := rec.Columns()
		values := cols[0].(*array.Int32).Int32Values()
		resVals = append(resVals, values...)
	}

	suite.ElementsMatch([]int32{4, 5, 6, 8, 4, 6}, resVals)
}

func TestDefaultSpaceTestSuite(t *testing.T) {
	suite.Run(t, new(DefaultSpaceTestSuite))
	suite.Run(t, new(SpaceTestSuite))
}
