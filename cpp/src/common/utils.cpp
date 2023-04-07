#include "utils.h"
#include <arrow/result.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>
#include <memory>
#include "common/exception.h"
#include "parquet/exception.h"
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include "constants.h"

schema_proto::LogicType
ToProtobufType(arrow::Type::type type) {
  auto type_id = static_cast<int>(type);
  if (type_id < 0 || type_id >= static_cast<int>(schema_proto::LogicType::MAX_ID)) {
    throw StorageException("invalid type");
  }
  return static_cast<schema_proto::LogicType>(type_id);
}

std::unique_ptr<schema_proto::KeyValueMetadata>
ToProtobufMetadata(const arrow::KeyValueMetadata* metadata) {
  auto proto_metadata = std::make_unique<schema_proto::KeyValueMetadata>();
  for (const auto& key : metadata->keys()) {
    proto_metadata->add_keys(key);
  }
  for (const auto& value : metadata->values()) {
    proto_metadata->add_values(value);
  }
  return proto_metadata;
}

std::unique_ptr<schema_proto::DataType>
ToProtobufDataType(const arrow::DataType* type);

std::unique_ptr<schema_proto::Field>
ToProtobufField(const arrow::Field* field) {
  auto proto_field = std::make_unique<schema_proto::Field>();
  proto_field->set_name(field->name());
  proto_field->set_nullable(field->nullable());
  proto_field->set_allocated_metadata(ToProtobufMetadata(field->metadata().get()).release());
  proto_field->set_allocated_data_type(ToProtobufDataType(field->type().get()).release());
  return proto_field;
}

void
SetTypeValues(schema_proto::DataType* proto_type, const arrow::DataType* type) {
  switch (type->id()) {
    case arrow::Type::FIXED_SIZE_BINARY: {
      auto real_type = dynamic_cast<const arrow::FixedSizeBinaryType*>(type);
      auto fixed_size_binary_type = new schema_proto::FixedSizeBinaryType();
      fixed_size_binary_type->set_byte_width(real_type->byte_width());
      proto_type->set_allocated_fixed_size_binary_type(fixed_size_binary_type);
      return;
    }
    case arrow::Type::FIXED_SIZE_LIST: {
      auto real_type = dynamic_cast<const arrow::FixedSizeListType*>(type);
      auto fixed_size_list_type = new schema_proto::FixedSizeListType();
      fixed_size_list_type->set_list_size(real_type->list_size());
      proto_type->set_allocated_fixed_size_list_type(fixed_size_list_type);
      return;
    }
    case arrow::Type::DICTIONARY: {
      auto real_type = dynamic_cast<const arrow::DictionaryType*>(type);
      auto dictionary_type = new schema_proto::DictionaryType();
      dictionary_type->set_allocated_index_type(ToProtobufDataType(real_type->index_type().get()).release());
      dictionary_type->set_allocated_index_type(ToProtobufDataType(real_type->value_type().get()).release());
      dictionary_type->set_ordered(real_type->ordered());
      proto_type->set_allocated_dictionary_type(dictionary_type);
      return;
    }
    case arrow::Type::MAP: {
      auto real_type = dynamic_cast<const arrow::MapType*>(type);
      auto map_type = new schema_proto::MapType();
      map_type->set_keys_sorted(real_type->keys_sorted());
      proto_type->set_allocated_map_type(map_type);
      return;
    }
    default:
      return;
  }
}

std::unique_ptr<schema_proto::DataType>
ToProtobufDataType(const arrow::DataType* type) {
  auto proto_type = std::make_unique<schema_proto::DataType>();
  SetTypeValues(proto_type.get(), type);
  proto_type->set_logic_type(ToProtobufType(type->id()));
  for (const auto& field : type->fields()) {
    proto_type->mutable_children()->AddAllocated(ToProtobufField(field.get()).release());
  }

  return proto_type;
}

std::unique_ptr<schema_proto::ArrowSchema>
ToProtobufSchema(const arrow::Schema* schema) {
  auto proto_schema = std::make_unique<schema_proto::ArrowSchema>();

  for (const auto& field : schema->fields()) {
    proto_schema->mutable_fields()->AddAllocated(ToProtobufField(field.get()).release());
  }

  proto_schema->set_endianness(schema->endianness() == arrow::Endianness::Little ? schema_proto::Endianness::Little
                                                                                 : schema_proto::Endianness::Big);

  for (const auto& key : schema->metadata()->keys()) {
    proto_schema->mutable_metadata()->add_keys(key);
  }
  for (const auto& value : schema->metadata()->values()) {
    proto_schema->mutable_metadata()->add_values(value);
  }
  return proto_schema;
}

arrow::Type::type
FromProtobufType(schema_proto::LogicType type) {
  auto type_id = static_cast<int>(type);
  if (type_id < 0 || type_id >= static_cast<int>(arrow::Type::MAX_ID)) {
    throw StorageException("invalid type");
  }
  return static_cast<arrow::Type::type>(type_id);
}

std::shared_ptr<arrow::KeyValueMetadata>
FromProtobufKeyValueMetadata(const schema_proto::KeyValueMetadata& metadata) {
  std::vector<std::string> keys(metadata.keys().begin(), metadata.keys().end());
  std::vector<std::string> values(metadata.values().begin(), metadata.values().end());
  return arrow::KeyValueMetadata::Make(keys, values);
}

std::shared_ptr<arrow::DataType>
FromProtobufDataType(const schema_proto::DataType& type);

std::shared_ptr<arrow::Field>
FromProtobufField(const schema_proto::Field& field) {
  auto data_type = FromProtobufDataType(field.data_type());
  auto metadata = FromProtobufKeyValueMetadata(field.metadata());
  return std::make_shared<arrow::Field>(field.name(), data_type, field.nullable(), metadata);
}

std::shared_ptr<arrow::DataType>
FromProtobufDataType(const schema_proto::DataType& type) {
  switch (type.logic_type()) {
    case schema_proto::NA:
      return std::make_shared<arrow::DataType>(arrow::NullType());
    case schema_proto::BOOL:
      return std::make_shared<arrow::DataType>(arrow::BooleanType());
    case schema_proto::UINT8:
      return std::make_shared<arrow::DataType>(arrow::UInt8Type());
    case schema_proto::INT8:
      return std::make_shared<arrow::DataType>(arrow::Int8Type());
    case schema_proto::UINT16:
      return std::make_shared<arrow::DataType>(arrow::UInt16Type());
    case schema_proto::INT16:
      return std::make_shared<arrow::DataType>(arrow::Int16Type());
    case schema_proto::UINT32:
      return std::make_shared<arrow::DataType>(arrow::UInt32Type());
    case schema_proto::INT32:
      return std::make_shared<arrow::DataType>(arrow::Int32Type());
    case schema_proto::UINT64:
      return std::make_shared<arrow::DataType>(arrow::UInt64Type());
    case schema_proto::INT64:
      return std::make_shared<arrow::DataType>(arrow::Int64Type());
    case schema_proto::HALF_FLOAT:
      return std::make_shared<arrow::DataType>(arrow::HalfFloatType());
    case schema_proto::FLOAT:
      return std::make_shared<arrow::DataType>(arrow::FloatType());
    case schema_proto::DOUBLE:
      return std::make_shared<arrow::DataType>(arrow::DoubleType());
    case schema_proto::STRING:
      return std::make_shared<arrow::DataType>(arrow::StringType());
    case schema_proto::BINARY:
      return std::make_shared<arrow::DataType>(arrow::BinaryType());
    case schema_proto::LIST: {
      auto field = FromProtobufField(type.children(0));
      return std::make_shared<arrow::DataType>(arrow::ListType(field));
    }
    case schema_proto::STRUCT: {
      std::vector<std::shared_ptr<arrow::Field>> fields;
      for (const auto& child : type.children()) {
        fields.push_back(FromProtobufField(child));
      }
      return std::make_shared<arrow::DataType>(arrow::StructType(fields));
    }
    case schema_proto::DICTIONARY: {
      auto index_type = FromProtobufDataType(type.dictionary_type().index_type());
      auto value_type = FromProtobufDataType(type.dictionary_type().value_type());
      return std::make_shared<arrow::DataType>(
          arrow::DictionaryType(index_type, value_type, type.dictionary_type().ordered()));
    }
    case schema_proto::MAP: {
      auto value_field = FromProtobufField(type.children(0));
      return std::make_shared<arrow::DataType>(arrow::MapType(value_field, type.map_type().keys_sorted()));
    }
    case schema_proto::FIXED_SIZE_BINARY:
      return std::make_shared<arrow::DataType>(arrow::FixedSizeBinaryType(type.fixed_size_binary_type().byte_width()));

    case schema_proto::FIXED_SIZE_LIST: {
      auto field = FromProtobufField(type.children(0));
      return std::make_shared<arrow::DataType>(
          arrow::FixedSizeListType(field, type.fixed_size_list_type().list_size()));
    }
    default:
      throw StorageException("invalid type");
  }
}

std::shared_ptr<arrow::Schema>
FromProtobufSchema(schema_proto::ArrowSchema* schema) {
  arrow::SchemaBuilder schema_builder;
  for (const auto& field : schema->fields()) {
    PARQUET_THROW_NOT_OK(schema_builder.AddField(FromProtobufField(field)));
  }
  PARQUET_ASSIGN_OR_THROW(auto res, schema_builder.Finish());
  return res;
}

std::string
GetNewParquetFilePath(std::string& path) {
  auto scalar_file_id = boost::uuids::random_generator()();
  return path + boost::uuids::to_string(scalar_file_id) + kParquetDataFileSuffix;
}

std::string
GetManifestFilePath(std::string& path) {
  return path + kManifestFileName;
}

std::string
GetManifestTmpFilePath(std::string& path) {
  return path + kManifestTempFileName;
}