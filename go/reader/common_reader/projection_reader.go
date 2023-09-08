package common_reader

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage/go/common/utils"
	"github.com/milvus-io/milvus-storage/go/storage/options/option"
)

type ProjectionReader struct {
	array.RecordReader
	reader  array.RecordReader
	options *option.ReadOptions
	schema  *arrow.Schema
}

func NewProjectionReader(reader array.RecordReader, options *option.ReadOptions, schema *arrow.Schema) array.RecordReader {
	projectionSchema := utils.ProjectSchema(schema, options.Columns)
	return &ProjectionReader{reader: reader, options: options, schema: projectionSchema}
}
