package schema_option

import (
	"errors"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/milvus-io/milvus-storage/proto/schema_proto"
)

var (
	ErrPrimaryColumnNotFound = errors.New("primary column not found")
	ErrPrimaryColumnType     = errors.New("primary column is not int64 or string")
	ErrPrimaryColumnEmpty    = errors.New("primary column is empty")
	ErrVersionColumnNotFound = errors.New("version column not found")
	ErrVersionColumnType     = errors.New("version column is not int64")
	ErrVectorColumnNotFound  = errors.New("vector column not found")
	ErrVectorColumnType      = errors.New("vector column is not fixed size binary")
	ErrVectorColumnEmpty     = errors.New("vector column is empty")
)

type SchemaOptions struct {
	PrimaryColumn string
	VersionColumn string
	VectorColumn  string
}

func Init() *SchemaOptions {
	return &SchemaOptions{
		PrimaryColumn: "",
		VersionColumn: "",
		VectorColumn:  "",
	}
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

func (o *SchemaOptions) Validate(schema *arrow.Schema) error {
	if o.PrimaryColumn != "" {
		primaryField, ok := schema.FieldsByName(o.PrimaryColumn)
		if !ok {
			return ErrPrimaryColumnNotFound
		} else if primaryField[0].Type.ID() != arrow.STRING && primaryField[0].Type.ID() != arrow.INT64 {
			return ErrPrimaryColumnType
		}
	} else {
		return ErrPrimaryColumnEmpty
	}
	if o.VersionColumn != "" {
		versionField, ok := schema.FieldsByName(o.VersionColumn)
		if !ok {
			return ErrVersionColumnNotFound
		} else if versionField[0].Type.ID() != arrow.INT64 {
			return ErrVersionColumnType
		}
	}
	if o.VectorColumn != "" {
		vectorField, b := schema.FieldsByName(o.VectorColumn)
		if !b {
			return ErrVectorColumnNotFound
		} else if vectorField[0].Type.ID() != arrow.FIXED_SIZE_BINARY {
			return ErrVectorColumnType
		}
	} else {
		return ErrVectorColumnEmpty
	}
	return nil
}

func (o *SchemaOptions) HasVersionColumn() bool {
	return o.VersionColumn != ""
}
