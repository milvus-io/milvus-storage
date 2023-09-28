package utils

import (
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/endian"
	"github.com/google/uuid"
	"github.com/milvus-io/milvus-storage/go/common/constant"
	"github.com/milvus-io/milvus-storage/go/common/log"
	"github.com/milvus-io/milvus-storage/go/proto/schema_proto"
)

var (
	ErrInvalidArgument = errors.New("invalid argument")
)

func ToProtobufType(dataType arrow.Type) (schema_proto.LogicType, error) {
	typeId := int(dataType)
	if typeId < 0 || typeId >= int(schema_proto.LogicType_MAX_ID) {
		return schema_proto.LogicType_NA, fmt.Errorf("parse data type %v: %w", dataType, ErrInvalidArgument)
	}
	return schema_proto.LogicType(typeId), nil
}

func ToProtobufMetadata(metadata *arrow.Metadata) (*schema_proto.KeyValueMetadata, error) {
	keys := metadata.Keys()
	values := metadata.Values()
	return &schema_proto.KeyValueMetadata{Keys: keys, Values: values}, nil
}

func ToProtobufDataType(dataType arrow.DataType) (*schema_proto.DataType, error) {
	protoType := &schema_proto.DataType{}
	err := SetTypeValues(protoType, dataType)
	if err != nil {
		return nil, err
	}
	logicType, err := ToProtobufType(dataType.ID())
	if err != nil {
		return nil, err
	}
	protoType.LogicType = logicType

	if len(GetFields(dataType)) > 0 {
		for _, field := range GetFields(dataType) {
			protoField := &schema_proto.Field{}
			protoFieldType, err := ToProtobufField(&field)
			if err != nil {
				return nil, err
			}
			protoField = protoFieldType
			protoType.Children = append(protoType.Children, protoField)
		}
	}

	return protoType, nil
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

func ToProtobufField(field *arrow.Field) (*schema_proto.Field, error) {
	protoField := &schema_proto.Field{}
	protoField.Name = field.Name
	protoField.Nullable = field.Nullable

	if field.Metadata.Len() != 0 {
		fieldMetadata, err := ToProtobufMetadata(&field.Metadata)
		if err != nil {
			return nil, fmt.Errorf("convert to protobuf field: %w", err)
		}
		protoField.Metadata = fieldMetadata
	}

	dataType, err := ToProtobufDataType(field.Type)
	if err != nil {
		return nil, fmt.Errorf("convert to protobuf field: %w", err)
	}
	protoField.DataType = dataType
	return protoField, nil
}

func SetTypeValues(protoType *schema_proto.DataType, dataType arrow.DataType) error {
	switch dataType.ID() {
	case arrow.FIXED_SIZE_BINARY:
		realType, ok := dataType.(*arrow.FixedSizeBinaryType)
		if !ok {
			return fmt.Errorf("convert to fixed size binary type: %w", ErrInvalidArgument)
		}
		fixedSizeBinaryType := &schema_proto.FixedSizeBinaryType{}
		fixedSizeBinaryType.ByteWidth = int32(realType.ByteWidth)
		protoType.TypeRelatedValues = &schema_proto.DataType_FixedSizeBinaryType{FixedSizeBinaryType: fixedSizeBinaryType}
		break
	case arrow.FIXED_SIZE_LIST:
		realType, ok := dataType.(*arrow.FixedSizeListType)
		if !ok {
			return fmt.Errorf("convert to fixed size list type: %w", ErrInvalidArgument)
		}
		fixedSizeListType := &schema_proto.FixedSizeListType{}
		fixedSizeListType.ListSize = int32(realType.Len())
		protoType.TypeRelatedValues = &schema_proto.DataType_FixedSizeListType{FixedSizeListType: fixedSizeListType}
		break
	case arrow.DICTIONARY:
		realType, ok := dataType.(*arrow.DictionaryType)
		if !ok {
			return fmt.Errorf("convert to dictionary type: %w", ErrInvalidArgument)
		}
		dictionaryType := &schema_proto.DictionaryType{}
		indexType, err := ToProtobufDataType(realType.IndexType)
		if err != nil {
			return err
		}
		dictionaryType.IndexType = indexType
		valueType, err := ToProtobufDataType(realType.ValueType)
		if err != nil {
			return err
		}
		dictionaryType.ValueType = valueType
		dictionaryType.Ordered = realType.Ordered
		protoType.TypeRelatedValues = &schema_proto.DataType_DictionaryType{DictionaryType: dictionaryType}
		break

	case arrow.MAP:
		realType, ok := dataType.(*arrow.MapType)
		if !ok {
			return fmt.Errorf("convert to map type: %w", ErrInvalidArgument)
		}
		mapType := &schema_proto.MapType{}
		mapType.KeysSorted = realType.KeysSorted
		protoType.TypeRelatedValues = &schema_proto.DataType_MapType{MapType: mapType}
		break

	default:
	}

	return nil
}

func ToProtobufSchema(schema *arrow.Schema) (*schema_proto.ArrowSchema, error) {
	protoSchema := &schema_proto.ArrowSchema{}
	for _, field := range schema.Fields() {
		protoField, err := ToProtobufField(&field)
		if err != nil {
			return nil, err
		}
		protoSchema.Fields = append(protoSchema.Fields, protoField)
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

	return protoSchema, nil
}

func FromProtobufSchema(schema *schema_proto.ArrowSchema) (*arrow.Schema, error) {
	fields := make([]arrow.Field, 0, len(schema.Fields))
	for _, field := range schema.Fields {
		tmp, err := FromProtobufField(field)
		if err != nil {
			return nil, err
		}
		fields = append(fields, *tmp)
	}
	tmp, err := FromProtobufKeyValueMetadata(schema.Metadata)
	if err != nil {
		return nil, err
	}
	newSchema := arrow.NewSchema(fields, tmp)
	return newSchema, nil
}

func FromProtobufField(field *schema_proto.Field) (*arrow.Field, error) {
	datatype, err := FromProtobufDataType(field.DataType)
	if err != nil {
		return nil, err
	}

	metadata, err := FromProtobufKeyValueMetadata(field.GetMetadata())
	if err != nil {
		return nil, err
	}

	return &arrow.Field{Name: field.Name, Type: datatype, Nullable: field.Nullable, Metadata: *metadata}, nil
}

func FromProtobufKeyValueMetadata(metadata *schema_proto.KeyValueMetadata) (*arrow.Metadata, error) {
	keys := make([]string, 0)
	values := make([]string, 0)
	if metadata != nil {
		keys = metadata.Keys
		values = metadata.Values
	}
	newMetadata := arrow.NewMetadata(keys, values)
	return &newMetadata, nil
}

func FromProtobufDataType(dataType *schema_proto.DataType) (arrow.DataType, error) {
	switch dataType.LogicType {
	case schema_proto.LogicType_NA:
		return &arrow.NullType{}, nil
	case schema_proto.LogicType_BOOL:
		return &arrow.BooleanType{}, nil
	case schema_proto.LogicType_UINT8:
		return &arrow.Uint8Type{}, nil
	case schema_proto.LogicType_INT8:
		return &arrow.Int8Type{}, nil
	case schema_proto.LogicType_UINT16:
		return &arrow.Uint16Type{}, nil
	case schema_proto.LogicType_INT16:
		return &arrow.Int16Type{}, nil
	case schema_proto.LogicType_UINT32:
		return &arrow.Uint32Type{}, nil
	case schema_proto.LogicType_INT32:
		return &arrow.Int32Type{}, nil
	case schema_proto.LogicType_UINT64:
		return &arrow.Uint64Type{}, nil
	case schema_proto.LogicType_INT64:
		return &arrow.Int64Type{}, nil
	case schema_proto.LogicType_HALF_FLOAT:
		return &arrow.Float16Type{}, nil
	case schema_proto.LogicType_FLOAT:
		return &arrow.Float32Type{}, nil
	case schema_proto.LogicType_DOUBLE:
		return &arrow.Float64Type{}, nil
	case schema_proto.LogicType_STRING:
		return &arrow.StringType{}, nil
	case schema_proto.LogicType_BINARY:
		return &arrow.BinaryType{}, nil

	case schema_proto.LogicType_LIST:
		fieldType, err := FromProtobufField(dataType.Children[0])
		if err != nil {
			return nil, err
		}
		listType := arrow.ListOf(fieldType.Type)
		return listType, nil

	case schema_proto.LogicType_STRUCT:
		fields := make([]arrow.Field, 0, len(dataType.Children))
		for _, child := range dataType.Children {
			field, err := FromProtobufField(child)
			if err != nil {
				return nil, err
			}
			fields = append(fields, *field)
		}
		structType := arrow.StructOf(fields...)
		return structType, nil

	case schema_proto.LogicType_DICTIONARY:
		keyType, err := FromProtobufField(dataType.Children[0])
		if err != nil {
			return nil, err
		}
		valueType, err := FromProtobufField(dataType.Children[1])
		if err != nil {
			return nil, err
		}
		dictType := &arrow.DictionaryType{
			IndexType: keyType.Type,
			ValueType: valueType.Type,
		}
		return dictType, nil

	case schema_proto.LogicType_MAP:
		fieldType, err := FromProtobufField(dataType.Children[0])
		if err != nil {
			return nil, err
		}
		//TODO FIX ME
		return arrow.MapOf(fieldType.Type, fieldType.Type), nil

	case schema_proto.LogicType_FIXED_SIZE_BINARY:

		sizeBinaryType := arrow.FixedSizeBinaryType{ByteWidth: int(dataType.GetFixedSizeBinaryType().ByteWidth)}
		return &sizeBinaryType, nil

	case schema_proto.LogicType_FIXED_SIZE_LIST:
		fieldType, err := FromProtobufField(dataType.Children[0])
		if err != nil {
			return nil, err
		}
		fixedSizeListType := arrow.FixedSizeListOf(int32(int(dataType.GetFixedSizeListType().ListSize)), fieldType.Type)
		return fixedSizeListType, nil

	default:
		return nil, fmt.Errorf("parse protobuf datatype: %w", ErrInvalidArgument)
	}
}

func GetNewParquetFilePath(path string) string {
	scalarFileId := uuid.New()
	path = filepath.Join(path, scalarFileId.String()+constant.ParquetDataFileSuffix)
	return path
}

func GetManifestFilePath(path string, version int64) string {
	path = filepath.Join(path, constant.ManifestDir, strconv.FormatInt(version, 10)+constant.ManifestFileSuffix)
	return path
}

func GetManifestTmpFilePath(path string, version int64) string {
	path = filepath.Join(path, constant.ManifestDir, strconv.FormatInt(version, 10)+constant.ManifestTempFileSuffix)
	return path
}

func GetBlobFilePath(path string) string {
	blobId := uuid.New()
	return filepath.Join(GetBlobDir(path), blobId.String())
}

func GetManifestDir(path string) string {
	path = filepath.Join(path, constant.ManifestDir)
	return path
}

func GetVectorDataDir(path string) string {
	return filepath.Join(path, constant.VectorDataDir)
}

func GetScalarDataDir(path string) string {
	return filepath.Join(path, constant.ScalarDataDir)
}

func GetBlobDir(path string) string {
	return filepath.Join(path, constant.BlobDir)
}

func GetDeleteDataDir(path string) string {
	return filepath.Join(path, constant.DeleteDataDir)
}

func ParseVersionFromFileName(path string) int64 {
	pos := strings.Index(path, constant.ManifestFileSuffix)
	if pos == -1 || !strings.HasSuffix(path, constant.ManifestFileSuffix) {
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
