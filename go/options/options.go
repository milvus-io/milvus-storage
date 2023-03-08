package options

import "github.com/milvus-io/milvus-storage-format/filter"

type FsType int8

const (
	InMemory FsType = iota
)

type SpaceOptions struct {
	Fs            FsType
	VectorColumns []string
}

type ReadOptions struct {
	Filters map[string]filter.Filter
	Columns []string
}

type WriteOptions struct {
	MaxRowsPerFile int64
}
