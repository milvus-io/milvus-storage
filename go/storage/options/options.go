package options

import (
	"math"

	"github.com/milvus-io/milvus-storage/go/common/constant"
	"github.com/milvus-io/milvus-storage/go/filter"
	"github.com/milvus-io/milvus-storage/go/storage/lock"
	"github.com/milvus-io/milvus-storage/go/storage/schema"
)

type Options struct {
	Schema      *schema.Schema   // optional
	Version     int64            // optional
	LockManager lock.LockManager // optional, no lock manager as default
}

type SpaceOptionsBuilder struct {
	options Options
}

func (b *SpaceOptionsBuilder) SetSchema(schema *schema.Schema) *SpaceOptionsBuilder {
	b.options.Schema = schema
	return b
}

func (b *SpaceOptionsBuilder) SetVersion(version int64) *SpaceOptionsBuilder {
	b.options.Version = version
	return b
}

func (b *SpaceOptionsBuilder) SetLockManager(lockManager lock.LockManager) *SpaceOptionsBuilder {
	b.options.LockManager = lockManager
	return b
}

func (b *SpaceOptionsBuilder) Reset() {
	b.options = Options{LockManager: &lock.EmptyLockManager{}}
}

func (b *SpaceOptionsBuilder) Build() Options { return b.options }

func NewSpaceOptionBuilder() *SpaceOptionsBuilder {
	return &SpaceOptionsBuilder{
		options: Options{
			Version:     constant.LatestManifestVersion,
			LockManager: &lock.EmptyLockManager{},
		},
	}
}

func DefaultOptions() *Options {
	return &Options{}
}

type WriteOptions struct {
	MaxRecordPerFile int64
}

var DefaultWriteOptions = WriteOptions{
	MaxRecordPerFile: 1024,
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
	Filters         map[string]filter.Filter
	FiltersV2       FilterSet
	Columns         []string
	ManifestVersion int64
	version         int64
}

func NewReadOptions() *ReadOptions {
	return &ReadOptions{
		Filters:         make(map[string]filter.Filter),
		FiltersV2:       make(FilterSet, 0),
		Columns:         make([]string, 0),
		ManifestVersion: constant.LatestManifestVersion,
		version:         math.MaxInt64,
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
