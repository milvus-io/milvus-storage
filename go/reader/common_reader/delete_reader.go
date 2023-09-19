package common_reader

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
	"github.com/milvus-io/milvus-storage/go/storage/options"
	"github.com/milvus-io/milvus-storage/go/storage/schema"
)

type DeleteReader struct {
	recordReader    array.RecordReader
	schemaOptions   *schema.SchemaOptions
	deleteFragments fragment.DeleteFragmentVector
	options         *options.ReadOptions
}

func (d DeleteReader) Retain() {
	//TODO implement me
	panic("implement me")
}

func (d DeleteReader) Release() {
	//TODO implement me
	panic("implement me")
}

func (d DeleteReader) Schema() *arrow.Schema {
	//TODO implement me
	panic("implement me")
}

func (d DeleteReader) Next() bool {
	//TODO implement me
	panic("implement me")
}

func (d DeleteReader) Record() arrow.Record {
	//TODO implement me
	panic("implement me")
}

func (d DeleteReader) Err() error {
	//TODO implement me
	panic("implement me")
}

func NewDeleteReader(recordReader array.RecordReader, schemaOptions *schema.SchemaOptions, deleteFragments fragment.DeleteFragmentVector, options *options.ReadOptions) *DeleteReader {
	return &DeleteReader{recordReader: recordReader, schemaOptions: schemaOptions, deleteFragments: deleteFragments, options: options}
}
