#include "common/utils.h"
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>
#include <memory>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <string>
#include "common/constants.h"
#include "common/macro.h"
#include "arrow/filesystem/path_util.h"
#include "boost/algorithm/string/predicate.hpp"
#include <cstdlib>
namespace milvus_storage {

Result<schema_proto::LogicType> ToProtobufType(arrow::Type::type type) {
  auto type_id = static_cast<int>(type);
  if (type_id < 0 || type_id >= static_cast<int>(schema_proto::LogicType::MAX_ID)) {
    return Status::InvalidArgument("Invalid type id: " + std::to_string(type_id));
  }
  return static_cast<schema_proto::LogicType>(type_id);
}

std::unique_ptr<schema_proto::KeyValueMetadata> ToProtobufMetadata(const arrow::KeyValueMetadata* metadata) {
  auto proto_metadata = std::make_unique<schema_proto::KeyValueMetadata>();
  assert(metadata != nullptr);
  for (const auto& key : metadata->keys()) {
    proto_metadata->add_keys(key);
  }
  for (const auto& value : metadata->values()) {
    proto_metadata->add_values(value);
  }
  return proto_metadata;
}

Result<std::unique_ptr<schema_proto::DataType>> ToProtobufDataType(const arrow::DataType* type);

Result<std::unique_ptr<schema_proto::Field>> ToProtobufField(const arrow::Field* field) {
  auto proto_field = std::make_unique<schema_proto::Field>();
  proto_field->set_name(field->name());
  proto_field->set_nullable(field->nullable());
  if (field->metadata() != nullptr) {
    proto_field->set_allocated_metadata(ToProtobufMetadata(field->metadata().get()).release());
  }
  ASSIGN_OR_RETURN_NOT_OK(auto data_type, ToProtobufDataType(field->type().get()));

  proto_field->set_allocated_data_type(data_type.release());
  return proto_field;
}

Status SetTypeValues(schema_proto::DataType* proto_type, const arrow::DataType* type) {
  switch (type->id()) {
    case arrow::Type::INT64: {
      proto_type->set_logic_type(schema_proto::LogicType::INT64);
      break;
    }
    case arrow::Type::FIXED_SIZE_BINARY: {
      auto real_type = dynamic_cast<const arrow::FixedSizeBinaryType*>(type);
      auto fixed_size_binary_type = new schema_proto::FixedSizeBinaryType();
      fixed_size_binary_type->set_byte_width(real_type->byte_width());
      proto_type->set_allocated_fixed_size_binary_type(fixed_size_binary_type);
      break;
    }
    case arrow::Type::FIXED_SIZE_LIST: {
      auto real_type = dynamic_cast<const arrow::FixedSizeListType*>(type);
      auto fixed_size_list_type = new schema_proto::FixedSizeListType();
      fixed_size_list_type->set_list_size(real_type->list_size());
      proto_type->set_allocated_fixed_size_list_type(fixed_size_list_type);
      break;
    }
    case arrow::Type::DICTIONARY: {
      auto real_type = dynamic_cast<const arrow::DictionaryType*>(type);
      auto dictionary_type = new schema_proto::DictionaryType();
      ASSIGN_OR_RETURN_NOT_OK(auto index_type, ToProtobufDataType(real_type->index_type().get()));
      dictionary_type->set_allocated_index_type(index_type.release());
      ASSIGN_OR_RETURN_NOT_OK(auto value_type, ToProtobufDataType(real_type->value_type().get()));
      dictionary_type->set_allocated_index_type(value_type.release());
      dictionary_type->set_ordered(real_type->ordered());
      proto_type->set_allocated_dictionary_type(dictionary_type);
      break;
    }
    case arrow::Type::MAP: {
      auto real_type = dynamic_cast<const arrow::MapType*>(type);
      auto map_type = new schema_proto::MapType();
      map_type->set_keys_sorted(real_type->keys_sorted());
      proto_type->set_allocated_map_type(map_type);
      break;
    }
    default:
      return Status::InvalidArgument("Invalid type id: " + std::to_string(type->id()));
  }
  return Status::OK();
}
Result<std::unique_ptr<schema_proto::DataType>> ToProtobufDataType(const arrow::DataType* type) {
  auto proto_type = std::make_unique<schema_proto::DataType>();
  RETURN_NOT_OK(SetTypeValues(proto_type.get(), type));
  ASSIGN_OR_RETURN_NOT_OK(auto logic_type, ToProtobufType(type->id()));
  proto_type->set_logic_type(logic_type);
  for (const auto& field : type->fields()) {
    ASSIGN_OR_RETURN_NOT_OK(auto field_proto, ToProtobufField(field.get()));
    proto_type->mutable_children()->AddAllocated(field_proto.release());
  }

  return proto_type;
}

Result<std::unique_ptr<schema_proto::ArrowSchema>> ToProtobufSchema(const arrow::Schema* schema) {
  auto proto_schema = std::make_unique<schema_proto::ArrowSchema>();

  for (const auto& field : schema->fields()) {
    ASSIGN_OR_RETURN_NOT_OK(auto field_proto, ToProtobufField(field.get()));
    proto_schema->mutable_fields()->AddAllocated(field_proto.release());
  }

  proto_schema->set_endianness(schema->endianness() == arrow::Endianness::Little ? schema_proto::Endianness::Little
                                                                                 : schema_proto::Endianness::Big);

  if (schema->metadata() != nullptr) {
    for (const auto& key : schema->metadata()->keys()) {
      proto_schema->mutable_metadata()->add_keys(key);
    }
    for (const auto& value : schema->metadata()->values()) {
      proto_schema->mutable_metadata()->add_values(value);
    }
  }
  return proto_schema;
}

Result<arrow::Type::type> FromProtobufType(schema_proto::LogicType type) {
  auto type_id = static_cast<int>(type);
  if (type_id < 0 || type_id >= static_cast<int>(arrow::Type::MAX_ID)) {
    return Status::InvalidArgument("Invalid proto type id: " + std::to_string(type_id));
  }
  return static_cast<arrow::Type::type>(type_id);
}

std::shared_ptr<arrow::KeyValueMetadata> FromProtobufKeyValueMetadata(const schema_proto::KeyValueMetadata& metadata) {
  std::vector<std::string> keys(metadata.keys().begin(), metadata.keys().end());
  std::vector<std::string> values(metadata.values().begin(), metadata.values().end());
  return arrow::KeyValueMetadata::Make(keys, values);
}

Result<std::shared_ptr<arrow::DataType>> FromProtobufDataType(const schema_proto::DataType& type);

Result<std::shared_ptr<arrow::Field>> FromProtobufField(const schema_proto::Field& field) {
  ASSIGN_OR_RETURN_NOT_OK(auto data_type, FromProtobufDataType(field.data_type()));
  auto metadata = FromProtobufKeyValueMetadata(field.metadata());
  return std::make_shared<arrow::Field>(field.name(), data_type, field.nullable(), metadata);
}

Result<std::shared_ptr<arrow::DataType>> FromProtobufDataType(const schema_proto::DataType& type) {
  switch (type.logic_type()) {
    case schema_proto::NA:
      return std::shared_ptr<arrow::DataType>(new arrow::NullType());
    case schema_proto::BOOL:
      return std::shared_ptr<arrow::DataType>(new arrow::BooleanType());
    case schema_proto::UINT8:
      return std::shared_ptr<arrow::DataType>(new arrow::UInt8Type());
    case schema_proto::INT8:
      return std::shared_ptr<arrow::DataType>(new arrow::Int8Type());
    case schema_proto::UINT16:
      return std::shared_ptr<arrow::DataType>(new arrow::UInt16Type());
    case schema_proto::INT16:
      return std::shared_ptr<arrow::DataType>(new arrow::Int16Type());
    case schema_proto::UINT32:
      return std::shared_ptr<arrow::DataType>(new arrow::UInt32Type());
    case schema_proto::INT32:
      return std::shared_ptr<arrow::DataType>(new arrow::Int32Type());
    case schema_proto::UINT64:
      return std::shared_ptr<arrow::DataType>(new arrow::UInt64Type());
    case schema_proto::INT64:
      return std::shared_ptr<arrow::DataType>(new arrow::Int64Type());
    case schema_proto::HALF_FLOAT:
      return std::shared_ptr<arrow::DataType>(new arrow::HalfFloatType());
    case schema_proto::FLOAT:
      return std::shared_ptr<arrow::DataType>(new arrow::FloatType());
    case schema_proto::DOUBLE:
      return std::shared_ptr<arrow::DataType>(new arrow::DoubleType());
    case schema_proto::STRING:
      return std::shared_ptr<arrow::DataType>(new arrow::StringType());
    case schema_proto::BINARY:
      return std::shared_ptr<arrow::DataType>(new arrow::BinaryType());
    case schema_proto::LIST: {
      ASSIGN_OR_RETURN_NOT_OK(auto field, FromProtobufField(type.children(0)))
      return std::shared_ptr<arrow::DataType>(new arrow::ListType(field));
    }
    case schema_proto::STRUCT: {
      std::vector<std::shared_ptr<arrow::Field>> fields;
      for (const auto& child : type.children()) {
        ASSIGN_OR_RETURN_NOT_OK(auto field, FromProtobufField(child));
        fields.push_back(field);
      }
      return std::shared_ptr<arrow::DataType>(new arrow::StructType(fields));
    }
    case schema_proto::DICTIONARY: {
      ASSIGN_OR_RETURN_NOT_OK(auto index_type, FromProtobufDataType(type.dictionary_type().index_type()));
      ASSIGN_OR_RETURN_NOT_OK(auto value_type, FromProtobufDataType(type.dictionary_type().value_type()));
      return std::shared_ptr<arrow::DataType>(
          new arrow::DictionaryType(index_type, value_type, type.dictionary_type().ordered()));
    }
    case schema_proto::MAP: {
      ASSIGN_OR_RETURN_NOT_OK(auto field, FromProtobufField(type.children(0)));
      return std::shared_ptr<arrow::DataType>(new arrow::MapType(field, type.map_type().keys_sorted()));
    }
    case schema_proto::FIXED_SIZE_BINARY:
      return std::shared_ptr<arrow::DataType>(
          new arrow::FixedSizeBinaryType(type.fixed_size_binary_type().byte_width()));

    case schema_proto::FIXED_SIZE_LIST: {
      ASSIGN_OR_RETURN_NOT_OK(auto field, FromProtobufField(type.children(0)));
      return std::shared_ptr<arrow::DataType>(
          new arrow::FixedSizeListType(field, type.fixed_size_list_type().list_size()));
    }
    default:
      return Status::InvalidArgument("Invalid proto type: " + std::to_string(type.logic_type()));
  }
}

Result<std::shared_ptr<arrow::Schema>> FromProtobufSchema(const schema_proto::ArrowSchema& schema) {
  arrow::SchemaBuilder schema_builder;
  for (const auto& field : schema.fields()) {
    ASSIGN_OR_RETURN_NOT_OK(auto r, FromProtobufField(field));
    RETURN_ARROW_NOT_OK(schema_builder.AddField(r));
  }
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto res, schema_builder.Finish());
  return res;
}

std::string GetNewParquetFilePath(const std::string& path) {
  auto scalar_file_id = boost::uuids::random_generator()();
  return arrow::fs::internal::JoinAbstractPath(
      std::vector<std::string_view>{path, boost::uuids::to_string(scalar_file_id) + kParquetDataFileSuffix});
}

std::string GetScalarDataDir(const std::string& path) {
  return arrow::fs::internal::JoinAbstractPath(std::vector<std::string_view>{path, kScalarDataDir});
}

std::string GetVectorDataDir(const std::string& path) {
  return arrow::fs::internal::JoinAbstractPath(std::vector<std::string_view>{path, kVectorDataDir});
}

std::string GetDeleteDataDir(const std::string& path) {
  return arrow::fs::internal::JoinAbstractPath(std::vector<std::string_view>{path, kDeleteDataDir});
}

std::string GetManifestFilePath(const std::string& path, const int64_t version) {
  return arrow::fs::internal::JoinAbstractPath(
      std::vector<std::string_view>{path, kManifestsDir, std::to_string(version) + kManifestFileSuffix});
}

std::string GetManifestTmpFilePath(const std::string& path, const int64_t version) {
  return arrow::fs::internal::JoinAbstractPath(
      std::vector<std::string_view>{path, kManifestsDir, std::to_string(version) + kManifestTempFileSuffix});
}
std::string GetBolbDir(const std::string& path) {
  return arrow::fs::internal::JoinAbstractPath(std::vector<std::string_view>{path, kBlobDir});
}

std::string GetNewBlobFilePath(const std::string& path) {
  auto scalar_file_id = boost::uuids::random_generator()();
  return arrow::fs::internal::JoinAbstractPath(
      std::vector<std::string_view>{path, kBlobDir, boost::uuids::to_string(scalar_file_id)});
}

int64_t ParseVersionFromFileName(const std::string& path) {
  auto pos = path.find(kManifestFileSuffix);
  if (pos == std::string::npos || !boost::algorithm::ends_with(path, kManifestFileSuffix)) {
    return -1;
  }
  auto version = path.substr(0, pos);
  return std::atol(version.c_str());
}

Result<std::shared_ptr<arrow::Schema>> ProjectSchema(std::shared_ptr<arrow::Schema> schema,
                                                     std::vector<std::string> columns) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (auto const& field : schema->fields()) {
    if (std::find(columns.begin(), columns.end(), field->name()) != columns.end()) {
      fields.push_back(field);
    }
  }

  arrow::SchemaBuilder schema_builder;
  RETURN_ARROW_NOT_OK(schema_builder.AddFields(fields));
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto projection_schema, schema_builder.Finish());
  return projection_schema;
}

std::string GetManifestDir(const std::string& path) {
  return arrow::fs::internal::JoinAbstractPath(std::vector<std::string_view>{path, kManifestsDir});
}
}  // namespace milvus_storage
