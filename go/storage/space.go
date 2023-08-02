package storage

import (
	"errors"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage-format/common/status"
	"github.com/milvus-io/milvus-storage-format/storage/options/option"
)

var (
	ErrSchemaNotMatch error = errors.New("schema not match")
	ErrColumnNotExist error = errors.New("column not exist")
)

type Space interface {
	Write(reader array.RecordReader, options *option.WriteOptions) status.Status
	Read(options *option.ReadOptions) (array.RecordReader, error)
}
