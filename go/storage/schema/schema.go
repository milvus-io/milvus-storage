package schema

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/milvus-io/milvus-storage-format/common"
	"github.com/milvus-io/milvus-storage-format/common/result"
	"github.com/milvus-io/milvus-storage-format/common/status"
	"github.com/milvus-io/milvus-storage-format/common/utils"
	"github.com/milvus-io/milvus-storage-format/proto/schema_proto"
	"github.com/milvus-io/milvus-storage-format/storage/options"
)

// Schema is a wrapper of arrow schema
type Schema struct {
	schema       *arrow.Schema
	scalarSchema *arrow.Schema
	vectorSchema *arrow.Schema
	deleteSchema *arrow.Schema

	options *options.SchemaOptions
}

func (s *Schema) Schema() *arrow.Schema {
	return s.schema
}

func (s *Schema) Options() *options.SchemaOptions {
	return s.options
}

func NewSchema(schema *arrow.Schema, options *options.SchemaOptions) *Schema {
	return &Schema{
		schema:  schema,
		options: options,
	}
}

func (s *Schema) Validate() status.Status {
	validate := s.options.Validate(s.schema)
	if !validate.IsOK() {
		return status.InternalStateError(validate.Msg())
	}
	scalarSchema := s.BuildScalarSchema()
	if !scalarSchema.IsOK() {
		return status.InternalStateError(scalarSchema.Msg())
	}
	vectorSchema := s.BuildVectorSchema()
	if !vectorSchema.IsOK() {
		return status.InternalStateError(vectorSchema.Msg())
	}
	deleteSchema := s.BuildDeleteSchema()
	if !deleteSchema.IsOK() {
		return status.InternalStateError(deleteSchema.Msg())
	}
	return status.OK()
}

func (s *Schema) ScalarSchema() *arrow.Schema {
	return s.scalarSchema
}

func (s *Schema) VectorSchema() *arrow.Schema {
	return s.vectorSchema
}

func (s *Schema) DeleteSchema() *arrow.Schema {
	return s.deleteSchema
}

func (s *Schema) FromProtobuf(schema *schema_proto.Schema) status.Status {
	schemaType := utils.FromProtobufSchema(schema.ArrowSchema)
	if !schemaType.Ok() {
		return status.ArrowError("invalid schema")
	}
	s.schema = schemaType.Value()
	s.options.FromProtobuf(schema.GetSchemaOptions())
	s.BuildScalarSchema()
	s.BuildVectorSchema()
	s.BuildDeleteSchema()
	return status.OK()
}

func (s *Schema) ToProtobuf() *result.Result[*schema_proto.Schema] {
	schema := &schema_proto.Schema{}
	arrowSchema := utils.ToProtobufSchema(s.schema)
	if !arrowSchema.Ok() {
		return result.NewResultFromStatus[*schema_proto.Schema](*arrowSchema.Status())
	}
	schema.ArrowSchema = arrowSchema.Value()
	schema.SchemaOptions = s.options.ToProtobuf()
	return result.NewResult[*schema_proto.Schema](schema, status.OK())
}

func (s *Schema) BuildScalarSchema() status.Status {
	fields := make([]arrow.Field, 0, len(s.schema.Fields()))
	for _, field := range s.schema.Fields() {
		if field.Name == s.options.VectorColumn {
			continue
		}
		fields = append(fields, field)
	}
	offsetFiled := arrow.Field{Name: common.KOffsetFieldName, Type: arrow.DataType(&arrow.Int64Type{})}
	fields = append(fields, offsetFiled)
	s.scalarSchema = arrow.NewSchema(fields, nil)

	return status.OK()
}

func (s *Schema) BuildVectorSchema() status.Status {
	fields := make([]arrow.Field, 0, len(s.schema.Fields()))
	for _, field := range s.schema.Fields() {
		if field.Name == s.options.VectorColumn ||
			field.Name == s.options.PrimaryColumn ||
			field.Name == s.options.VersionColumn {
			fields = append(fields, field)
		}
	}
	s.vectorSchema = arrow.NewSchema(fields, nil)

	return status.OK()
}

func (s *Schema) BuildDeleteSchema() status.Status {
	pkColumn, b := s.schema.FieldsByName(s.options.PrimaryColumn)
	if !b {
		return status.InvalidArgument("primary column not found")
	}
	versionField, b := s.schema.FieldsByName(s.options.VersionColumn)
	if !b {
		return status.InvalidArgument("version column not found")
	}
	fields := make([]arrow.Field, 0, 2)
	fields = append(fields, pkColumn[0])
	fields = append(fields, versionField[0])
	s.deleteSchema = arrow.NewSchema(fields, nil)
	return status.OK()
}
