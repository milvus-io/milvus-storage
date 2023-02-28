package storage_test

import (
	"testing"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/apache/arrow/go/v12/arrow/memory"
	"github.com/milvus-io/milvus-storage-format/filter"
	"github.com/milvus-io/milvus-storage-format/options"
	"github.com/milvus-io/milvus-storage-format/storage"
	"github.com/stretchr/testify/suite"
)

type DefaultSpaceTestSuite struct {
	suite.Suite
}

func (suite *DefaultSpaceTestSuite) TestSpaceReadWrite() {
	field := arrow.Field{
		Name:     "int32",
		Type:     arrow.PrimitiveTypes.Int32,
		Nullable: false,
	}

	schema := arrow.NewSchema([]arrow.Field{field}, nil)
	builder := array.NewInt32Builder(memory.DefaultAllocator)
	builder.AppendValues([]int32{1, 3, 4, 5, 6, 8, 2, 4, 6}, nil)
	arr := builder.NewArray()
	rec := array.NewRecord(schema, []arrow.Array{arr}, int64(arr.Len()))
	recReader, err := array.NewRecordReader(schema, []arrow.Record{rec})
	suite.NoError(err)

	space := storage.NewDefaultSpace(schema, &options.SpaceOptions{Fs: options.InMemory})
	writeOpt := &options.WriteOptions{MaxRowsPerFile: 10}
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
}
