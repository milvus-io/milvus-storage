package options

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/milvus-io/milvus-storage-format/common/status"
	"github.com/milvus-io/milvus-storage-format/filter"
	"github.com/milvus-io/milvus-storage-format/proto/manifest_proto"
	"github.com/milvus-io/milvus-storage-format/proto/schema_proto"
)

type Options struct {
	Uri string
}

func (o *Options) ToProtobuf() *manifest_proto.Options {
	options := &manifest_proto.Options{}
	options.Uri = o.Uri
	return options
}

func (o *Options) FromProtobuf(options *manifest_proto.Options) {
	o.Uri = options.Uri
}

type SchemaOptions struct {
	PrimaryColumn string
	VersionColumn string
	VectorColumn  string
}

func (o *SchemaOptions) ToProtobuf() *schema_proto.SchemaOptions {
	options := &schema_proto.SchemaOptions{}
	options.PrimaryColumn = o.PrimaryColumn
	options.VersionColumn = o.VersionColumn
	options.VectorColumn = o.VectorColumn
	return options
}

func (o *SchemaOptions) FromProtobuf(options *schema_proto.SchemaOptions) {
	o.PrimaryColumn = options.PrimaryColumn
	o.VersionColumn = options.VersionColumn
	o.VectorColumn = options.VectorColumn
}

func (o *SchemaOptions) Validate(schema *arrow.Schema) status.Status {
	if o.PrimaryColumn != "" {
		primaryField, b := schema.FieldsByName(o.PrimaryColumn)
		if !b {
			return status.InvalidArgument("primary column not found")
		} else if primaryField[0].Type.ID() != arrow.STRING && primaryField[0].Type.ID() != arrow.INT64 {
			return status.InvalidArgument("primary column is not int64 or string")
		}
	} else {
		return status.InvalidArgument("primary column is empty")
	}
	if o.VersionColumn != "" {
		versionField, b := schema.FieldsByName(o.VersionColumn)
		if !b {
			return status.InvalidArgument("version column not found")
		} else if versionField[0].Type.ID() != arrow.INT64 {
			return status.InvalidArgument("version column is not int64")
		}
	}
	if o.VectorColumn != "" {
		vectorField, b := schema.FieldsByName(o.VectorColumn)
		if !b {
			return status.InvalidArgument("vector column not found")
		} else if vectorField[0].Type.ID() != arrow.FIXED_SIZE_BINARY {
			return status.InvalidArgument("vector column is not fixed size binary")
		}
	} else {
		return status.InvalidArgument("vector column is empty")
	}
	return status.OK()
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
