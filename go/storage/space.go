package storage

import (
	"errors"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage-format/storage/options"
)

var (
	ErrSchemaNotMatch error = errors.New("schema not match")
	ErrColumnNotExist error = errors.New("column not exist")
)

type Space interface {
	Write(reader array.RecordReader, options *options.WriteOptions) error
	Read(options *options.ReadOptions) (array.RecordReader, error)
}
