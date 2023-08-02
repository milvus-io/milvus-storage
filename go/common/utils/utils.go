package utils

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/endian"
	"github.com/google/uuid"
	"github.com/milvus-io/milvus-storage-format/common/constant"
	"github.com/milvus-io/milvus-storage-format/common/log"
	"github.com/milvus-io/milvus-storage-format/common/result"
	"github.com/milvus-io/milvus-storage-format/common/status"
	"github.com/milvus-io/milvus-storage-format/proto/schema_proto"
	"path/filepath"
	"strconv"
	"strings"
)

func ToProtobufType(dataType arrow.Type) *result.Result[schema_proto.LogicType] {
	typeId := int(dataType)
	if typeId < 0 || typeId >= int(schema_proto.LogicType_MAX_ID) {
		return result.NewResultFromStatus[schema_proto.LogicType](status.InvalidArgument("Invalid type id: " + strconv.Itoa(typeId)))
	}
	return result.NewResult[schema_proto.LogicType](schema_proto.LogicType(typeId), status.OK())
}

func ToProtobufMetadata(metadata *arrow.Metadata) *result.Result[*schema_proto.KeyValueMetadata] {
	keys := metadata.Keys()
	values := metadata.Values()
	return result.NewResult[*schema_proto.KeyValueMetadata](&schema_proto.KeyValueMetadata{Keys: keys, Values: values}, status.OK())
}

func ToProtobufDataType(dataType arrow.DataType) *result.Result[*schema_proto.DataType] {
	protoType := &schema_proto.DataType{}
	stat := SetTypeValues(protoType, dataType)
	if !stat.IsOK() {
		return result.NewResultFromStatus[*schema_proto.DataType](stat)
	}
	logicType := ToProtobufType(dataType.ID())
	if !logicType.Status().IsOK() {
		return result.NewResultFromStatus[*schema_proto.DataType](*logicType.Status())
	}
	protoType.LogicType = logicType.Value()

	if len(GetFields(dataType)) > 0 {
		for _, field := range GetFields(dataType) {
			protoField := &schema_proto.Field{}
			protoFieldType := ToProtobufField(&field)
			if !protoFieldType.Ok() {
				return result.NewResultFromStatus[*schema_proto.DataType](*protoFieldType.Status())
			}
			protoField = protoFieldType.Value()
			protoType.Children = append(protoType.Children, protoField)
		}
	}

	return result.NewResult[*schema_proto.DataType](protoType, status.OK())
}

// GetFields TODO CHECK MORE TYPES
func GetFields(dataType arrow.DataType) []arrow.Field {
	switch dataType.ID() {
	case arrow.LIST:
		listType, _ := dataType.(*arrow.ListType)
		return listType.Fields()
	case arrow.STRUCT:
		structType, _ := dataType.(*arrow.StructType)
		return structType.Fields()
	case arrow.MAP:
		mapType, _ := dataType.(*arrow.MapType)
		return mapType.Fields()
	default:
		return nil
	}
}

func ToProtobufField(field *arrow.Field) *result.Result[*schema_proto.Field] {
	protoField := &schema_proto.Field{}
	protoField.Name = field.Name
	protoField.Nullable = field.Nullable

	if field.Metadata.Len() != 0 {
		fieldMetadata := ToProtobufMetadata(&field.Metadata)
		if !fieldMetadata.Status().IsOK() {
			return result.NewResultFromStatus[*schema_proto.Field](*fieldMetadata.Status())
		}
		protoField.Metadata = fieldMetadata.Value()
	}

	dataType := ToProtobufDataType(field.Type)
	if !dataType.Status().IsOK() {
		return result.NewResultFromStatus[*schema_proto.Field](*dataType.Status())
	}
	protoField.DataType = dataType.Value()
	return result.NewResult[*schema_proto.Field](protoField, status.OK())
}

func SetTypeValues(protoType *schema_proto.DataType, dataType arrow.DataType) status.Status {
	switch dataType.ID() {
	case arrow.STRING:
		_, ok := dataType.(*arrow.StringType)
		if !ok {
			return status.InvalidArgument("invalid type")
		}
		stringType := schema_proto.LogicType_STRING
		protoType = &schema_proto.DataType{LogicType: stringType}
		break

	case arrow.INT64:
		_, ok := dataType.(*arrow.Int64Type)
		if !ok {
			return status.InvalidArgument("invalid type")
		}
		int64Type := schema_proto.LogicType_INT64
		protoType = &schema_proto.DataType{LogicType: int64Type}
		break

	case arrow.FIXED_SIZE_BINARY:
		realType, ok := dataType.(*arrow.FixedSizeBinaryType)
		if !ok {
			return status.InvalidArgument("invalid type")
		}
		fixedSizeBinaryType := &schema_proto.FixedSizeBinaryType{}
		fixedSizeBinaryType.ByteWidth = int32(realType.ByteWidth)
		protoType.TypeRelatedValues = &schema_proto.DataType_FixedSizeBinaryType{FixedSizeBinaryType: fixedSizeBinaryType}
		break
	case arrow.FIXED_SIZE_LIST:
		realType, ok := dataType.(*arrow.FixedSizeListType)
		if !ok {
			return status.InvalidArgument("invalid type")
		}
		fixedSizeListType := &schema_proto.FixedSizeListType{}
		fixedSizeListType.ListSize = int32(realType.Len())
		protoType.TypeRelatedValues = &schema_proto.DataType_FixedSizeListType{FixedSizeListType: fixedSizeListType}
		break
	case arrow.DICTIONARY:
		realType, ok := dataType.(*arrow.DictionaryType)
		if !ok {
			return status.InvalidArgument("invalid type")
		}
		dictionaryType := &schema_proto.DictionaryType{}
		indexType := ToProtobufDataType(realType.IndexType)
		if !indexType.Status().IsOK() {
			return *indexType.Status()
		}
		dictionaryType.IndexType = indexType.Value()
		valueType := ToProtobufDataType(realType.ValueType)
		if !valueType.Status().IsOK() {
			return *valueType.Status()
		}
		dictionaryType.ValueType = valueType.Value()
		dictionaryType.Ordered = realType.Ordered
		protoType.TypeRelatedValues = &schema_proto.DataType_DictionaryType{DictionaryType: dictionaryType}
		break

	case arrow.MAP:
		realType, ok := dataType.(*arrow.MapType)
		if !ok {
			return status.InvalidArgument("invalid type")
		}
		mapType := &schema_proto.MapType{}
		mapType.KeysSorted = realType.KeysSorted
		protoType.TypeRelatedValues = &schema_proto.DataType_MapType{MapType: mapType}
		break

	default:
		return status.InvalidArgument("Invalid type id: " + dataType.ID().String())
	}

	return status.OK()
}

func ToProtobufSchema(schema *arrow.Schema) *result.Result[*schema_proto.ArrowSchema] {
	protoSchema := &schema_proto.ArrowSchema{}
	for _, field := range schema.Fields() {
		protoField := ToProtobufField(&field)
		if !protoField.Status().IsOK() {
			return result.NewResultFromStatus[*schema_proto.ArrowSchema](*protoField.Status())
		}
		protoSchema.Fields = append(protoSchema.Fields, protoField.Value())
	}
	if schema.Endianness() == endian.LittleEndian {
		protoSchema.Endianness = schema_proto.Endianness_Little
	} else if schema.Endianness() == endian.BigEndian {
		protoSchema.Endianness = schema_proto.Endianness_Big
	}

	// TODO FIX ME: golang proto not support proto_schema->mutable_metadata()->add_keys(key);
	if schema.HasMetadata() && !schema.HasMetadata() {
		for _, key := range schema.Metadata().Keys() {
			protoKeyValue := protoSchema.GetMetadata()
			protoKeyValue.Keys = append(protoKeyValue.Keys, key)
		}
		for _, value := range schema.Metadata().Values() {
			protoKeyValue := protoSchema.GetMetadata()
			protoKeyValue.Values = append(protoKeyValue.Values, value)
		}
	}

	return result.NewResult[*schema_proto.ArrowSchema](protoSchema, status.OK())
}

func FromProtobufSchema(schema *schema_proto.ArrowSchema) *result.Result[*arrow.Schema] {
	fields := make([]arrow.Field, 0, len(schema.Fields))
	for _, field := range schema.Fields {
		tmp := FromProtobufField(field)
		if !tmp.Status().IsOK() {
			return result.NewResultFromStatus[*arrow.Schema](*tmp.Status())
		}
		fields = append(fields, *tmp.Value())
	}
	tmp := FromProtobufKeyValueMetadata(schema.Metadata)
	if !tmp.Status().IsOK() {
		return result.NewResultFromStatus[*arrow.Schema](*tmp.Status())
	}
	metadata := tmp.Value()
	newSchema := arrow.NewSchema(fields, metadata)
	return result.NewResult[*arrow.Schema](newSchema, status.OK())
}

func FromProtobufField(field *schema_proto.Field) *result.Result[*arrow.Field] {
	datatype := FromProtobufDataType(field.DataType)
	if !datatype.Status().IsOK() {
		return result.NewResultFromStatus[*arrow.Field](*datatype.Status())
	}
	dataType := datatype.Value()

	metadata := FromProtobufKeyValueMetadata(field.GetMetadata())
	if !metadata.Status().IsOK() {
		return result.NewResultFromStatus[*arrow.Field](*metadata.Status())
	}
	metaData := metadata.Value()

	return result.NewResult[*arrow.Field](&arrow.Field{Name: field.Name, Type: dataType, Nullable: field.Nullable, Metadata: *metaData}, status.OK())
}

func FromProtobufKeyValueMetadata(metadata *schema_proto.KeyValueMetadata) *result.Result[*arrow.Metadata] {
	keys := make([]string, 0)
	values := make([]string, 0)
	if metadata != nil {
		keys = metadata.Keys
		values = metadata.Values
	}
	newMetadata := arrow.NewMetadata(keys, values)
	return result.NewResult[*arrow.Metadata](&newMetadata, status.OK())
}
func FromProtobufDataType(dataType *schema_proto.DataType) *result.Result[arrow.DataType] {
	switch dataType.LogicType {
	case schema_proto.LogicType_NA:
		return result.NewResult[arrow.DataType](&arrow.NullType{}, status.OK())
	case schema_proto.LogicType_BOOL:
		return result.NewResult[arrow.DataType](&arrow.BooleanType{}, status.OK())
	case schema_proto.LogicType_UINT8:
		return result.NewResult[arrow.DataType](&arrow.Uint8Type{}, status.OK())
	case schema_proto.LogicType_INT8:
		return result.NewResult[arrow.DataType](&arrow.Int8Type{}, status.OK())
	case schema_proto.LogicType_UINT16:
		return result.NewResult[arrow.DataType](&arrow.Uint16Type{}, status.OK())
	case schema_proto.LogicType_INT16:
		return result.NewResult[arrow.DataType](&arrow.Int16Type{}, status.OK())
	case schema_proto.LogicType_UINT32:
		return result.NewResult[arrow.DataType](&arrow.Uint32Type{}, status.OK())
	case schema_proto.LogicType_INT32:
		return result.NewResult[arrow.DataType](&arrow.Int32Type{}, status.OK())
	case schema_proto.LogicType_UINT64:
		return result.NewResult[arrow.DataType](&arrow.Uint64Type{}, status.OK())
	case schema_proto.LogicType_INT64:
		return result.NewResult[arrow.DataType](&arrow.Int64Type{}, status.OK())
	case schema_proto.LogicType_HALF_FLOAT:
		return result.NewResult[arrow.DataType](&arrow.Float16Type{}, status.OK())
	case schema_proto.LogicType_FLOAT:
		return result.NewResult[arrow.DataType](&arrow.Float32Type{}, status.OK())
	case schema_proto.LogicType_DOUBLE:
		return result.NewResult[arrow.DataType](&arrow.Float64Type{}, status.OK())
	case schema_proto.LogicType_STRING:
		return result.NewResult[arrow.DataType](&arrow.StringType{}, status.OK())
	case schema_proto.LogicType_BINARY:
		return result.NewResult[arrow.DataType](&arrow.BinaryType{}, status.OK())

	case schema_proto.LogicType_LIST:
		fieldType := FromProtobufField(dataType.Children[0])
		if !fieldType.Status().IsOK() {
			return result.NewResultFromStatus[arrow.DataType](*fieldType.Status())
		}
		listType := arrow.ListOf(fieldType.Value().Type)
		return result.NewResult[arrow.DataType](listType, status.OK())

	case schema_proto.LogicType_STRUCT:
		fields := make([]arrow.Field, 0, len(dataType.Children))
		for _, child := range dataType.Children {
			field := FromProtobufField(child)
			if !field.Status().IsOK() {
				return result.NewResultFromStatus[arrow.DataType](*field.Status())
			}
			fields = append(fields, *field.Value())
		}
		structType := arrow.StructOf(fields...)
		return result.NewResult[arrow.DataType](structType, status.OK())

	case schema_proto.LogicType_DICTIONARY:
		keyType := FromProtobufField(dataType.Children[0])
		if !keyType.Status().IsOK() {
			return result.NewResultFromStatus[arrow.DataType](*keyType.Status())
		}
		valueType := FromProtobufField(dataType.Children[1])
		if !valueType.Status().IsOK() {
			return result.NewResultFromStatus[arrow.DataType](*valueType.Status())
		}
		dictType := &arrow.DictionaryType{
			IndexType: keyType.Value().Type,
			ValueType: valueType.Value().Type,
		}
		return result.NewResult[arrow.DataType](dictType, status.OK())

	case schema_proto.LogicType_MAP:
		fieldType := FromProtobufField(dataType.Children[0])
		if !fieldType.Status().IsOK() {
			return result.NewResultFromStatus[arrow.DataType](*fieldType.Status())
		}
		//TODO FIX ME
		return result.NewResult[arrow.DataType](arrow.MapOf(fieldType.Value().Type, fieldType.Value().Type), status.OK())

	case schema_proto.LogicType_FIXED_SIZE_BINARY:

		sizeBinaryType := arrow.FixedSizeBinaryType{ByteWidth: int(dataType.GetFixedSizeBinaryType().ByteWidth)}
		return result.NewResult[arrow.DataType](&sizeBinaryType, status.OK())

	case schema_proto.LogicType_FIXED_SIZE_LIST:
		fieldType := FromProtobufField(dataType.Children[0])
		if !fieldType.Status().IsOK() {
			return result.NewResultFromStatus[arrow.DataType](*fieldType.Status())
		}
		fixedSizeListType := arrow.FixedSizeListOf(int32(int(dataType.GetFixedSizeListType().ListSize)), fieldType.Value().Type)
		return result.NewResult[arrow.DataType](fixedSizeListType, status.OK())

	default:
		return result.NewResultFromStatus[arrow.DataType](status.InvalidArgument("invalid data type"))
	}
}

func GetNewParquetFilePath(path string) string {
	scalarFileId := uuid.New()
	path = filepath.Join(path, scalarFileId.String()+constant.KParquetDataFileSuffix)
	return path
}

func GetManifestFilePath(path string, version int64) string {
	path = filepath.Join(path, constant.KManifestDir, strconv.FormatInt(version, 10)+constant.KManifestFileSuffix)
	return path
}

func GetManifestTmpFilePath(path string, version int64) string {
	path = filepath.Join(path, constant.KManifestDir, strconv.FormatInt(version, 10)+constant.KManifestTempFileSuffix)
	return path
}

func GetManifestDir(path string) string {
	path = filepath.Join(path, constant.KManifestDir)
	return path
}

func ParseVersionFromFileName(path string) int64 {
	pos := strings.Index(path, constant.KManifestFileSuffix)
	if pos == -1 || !strings.HasSuffix(path, constant.KManifestFileSuffix) {
		log.Warn("manifest file suffix not match", log.String("path", path))
		return -1
	}
	version := path[0:pos]
	versionInt, err := strconv.ParseInt(version, 10, 64)
	if err != nil {
		log.Error("parse version from file name error", log.String("path", path), log.String("version", version))
		return -1
	}
	return versionInt
}

func ProjectSchema(sc *arrow.Schema, columns []string) *arrow.Schema {
	var fields []arrow.Field
	for _, field := range sc.Fields() {
		for _, column := range columns {
			if field.Name == column {
				fields = append(fields, field)
				break
			}
		}
	}

	return arrow.NewSchema(fields, nil)
}
