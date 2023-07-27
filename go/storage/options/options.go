package options

import (
	"github.com/milvus-io/milvus-storage-format/filter"
	"github.com/milvus-io/milvus-storage-format/storage/schema"
)

type Options struct {
	Schema  *schema.Schema
	Version int64
}

func NewOptions(schema *schema.Schema, version int64) *Options {
	return &Options{
		Schema:  schema,
		Version: version,
	}
}

func Init() *Options {
	return &Options{}
}

type WriteOptions struct {
	MaxRecordPerFile int64
}

func NewWriteOption() *WriteOptions {
	return &WriteOptions{
		MaxRecordPerFile: 1024,
	}
}

type FsType int8

const (
	InMemory FsType = iota
	LocalFS
)

type SpaceOptions struct {
	Fs            FsType
	VectorColumns []string
}

type ReadOptions struct {
	Filters map[string]filter.Filter
	Columns []string
}
