package storage

import (
	"errors"

	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage/storage/options/option"
	"google.golang.org/grpc/status"
)

var (
	ErrSchemaNotMatch error = errors.New("schema not match")
	ErrColumnNotExist error = errors.New("column not exist")
)

type Space interface {
	Write(reader array.RecordReader, options *option.WriteOptions) status.Status
	Read(options *option.ReadOptions) (array.RecordReader, error)
}
