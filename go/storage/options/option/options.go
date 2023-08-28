package option

import (
	"github.com/milvus-io/milvus-storage/filter"
	"github.com/milvus-io/milvus-storage/storage/schema"
	"math"
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
	S3
)

type SpaceOptions struct {
	Fs            FsType
	VectorColumns []string
}

// TODO: Change to FilterSet type
type FilterSet []filter.Filter

var version int64 = math.MaxInt64

type ReadOptions struct {
	//Filters map[string]filter.Filter
	Filters   map[string]filter.Filter
	FiltersV2 FilterSet
	Columns   []string
	version   int64
}

func NewReadOptions() *ReadOptions {
	return &ReadOptions{
		Filters:   make(map[string]filter.Filter),
		FiltersV2: make(FilterSet, 0),
		Columns:   make([]string, 0),
		version:   math.MaxInt64,
	}
}

func (o *ReadOptions) AddFilter(filter filter.Filter) {
	o.Filters[filter.GetColumnName()] = filter
	o.FiltersV2 = append(o.FiltersV2, filter)
}

func (o *ReadOptions) AddColumn(column string) {
	o.Columns = append(o.Columns, column)
}

func (o *ReadOptions) SetColumns(columns []string) {
	o.Columns = columns
}

func (o *ReadOptions) SetVersion(version int64) {
	o.version = version
}

func (o *ReadOptions) GetVersion() int64 {
	return o.version
}

func (o *ReadOptions) OutputColumns() []string {
	return o.Columns
}
