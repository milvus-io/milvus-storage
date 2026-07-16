// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus-storage/format/parquet/parquet_format_reader.h"

#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fmt/format.h>

#include <arrow/array/util.h>
#include <arrow/buffer.h>
#include <arrow/extension_type.h>
#include <arrow/io/caching.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/schema.h>
#include <parquet/metadata.h>
#include <parquet/type_fwd.h>

#include "milvus-storage/format/parquet/key_retriever.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::parquet {

static std::vector<int> get_leaf_column_offsets_by_field(const arrow::Schema& file_schema);

static arrow::Result<std::vector<RowGroupInfo>> try_build_row_group_infos(
    const std::shared_ptr<::parquet::FileMetaData>& metadata) {
  std::vector<RowGroupInfo> row_group_infos;
  auto key_value_metadata = metadata->key_value_metadata();
  if (!key_value_metadata) {
    return row_group_infos;
  }

  auto row_group_meta_result = key_value_metadata->Get(ROW_GROUP_META_KEY);
  if (!row_group_meta_result.ok()) {
    return row_group_infos;
  }

  auto row_group_metadatas = RowGroupMetadataVector::Deserialize(row_group_meta_result.ValueOrDie());
  row_group_infos.reserve(row_group_metadatas.size());
  size_t offset = 0;
  for (size_t i = 0; i < row_group_metadatas.size(); ++i) {
    auto row_group_metadata = row_group_metadatas.Get(i);
    row_group_infos.emplace_back(RowGroupInfo{
        .start_offset = offset,
        .end_offset = offset + row_group_metadata.row_num(),
        .memory_size = row_group_metadata.memory_size(),
    });
    offset += row_group_metadata.row_num();
  }

  return row_group_infos;
}

static arrow::Result<std::vector<RowGroupInfo>> create_row_group_infos_from_metadata(
    const std::shared_ptr<::parquet::FileMetaData>& metadata,
    const arrow::Schema& file_schema,
    const std::string& path) {
  // Keep the existing row-group memory estimate as the total-size anchor: prefer
  // Milvus private metadata when available, otherwise fall back to the Parquet
  // footer's total byte size. Parquet leaf-column uncompressed sizes are used only
  // as relative weights. Leaves belonging to a nested top-level Arrow field are
  // aggregated so column_memory_sizes follows the complete file-schema order and
  // sums exactly to the row group's memory_size.
  assert(metadata);
  if (!metadata) {
    return arrow::Status::Invalid(fmt::format("Failed to get parquet file metadata for file: {}", path));
  }

  // try use the private kv metas to build row group infos
  ARROW_ASSIGN_OR_RAISE(auto row_group_infos, try_build_row_group_infos(metadata));
  if (row_group_infos.empty()) {
    // use the parquet file metadata to build row group infos
    row_group_infos.reserve(metadata->num_row_groups());
    size_t offset = 0;
    for (int i = 0; i < metadata->num_row_groups(); ++i) {
      auto row_group_meta = metadata->RowGroup(i);
      auto total_byte_size = row_group_meta->total_byte_size();
      if (total_byte_size < 0) {
        return arrow::Status::Invalid(
            fmt::format("Parquet row-group total byte size is negative. [path={}, row_group={}]", path, i));
      }
      row_group_infos.emplace_back(RowGroupInfo{
          .start_offset = offset,
          .end_offset = offset + static_cast<size_t>(row_group_meta->num_rows()),
          .memory_size = static_cast<uint64_t>(total_byte_size),
      });
      offset += row_group_meta->num_rows();
    }
  }

  // Map every top-level Arrow field to a half-open range of physical Parquet
  // leaf columns. For example, offsets [0, 1, 3] mean field 0 owns leaf [0, 1)
  // and field 1 owns leaves [1, 3).
  const auto leaf_column_offsets_by_field = get_leaf_column_offsets_by_field(file_schema);

  // The private row-group metadata and the Parquet footer must describe the
  // same row groups, while the Arrow schema must expand to every Parquet leaf.
  if (row_group_infos.size() != static_cast<size_t>(metadata->num_row_groups()) ||
      leaf_column_offsets_by_field.back() != metadata->num_columns()) {
    return arrow::Status::Invalid(
        fmt::format("Parquet row-group metadata does not match the file schema. [path={}]", path));
  }

  for (int row_group_index = 0; row_group_index < metadata->num_row_groups(); ++row_group_index) {
    auto row_group_meta = metadata->RowGroup(row_group_index);

    // Build one weight per top-level Arrow field. Nested fields contribute the
    // sum of total_uncompressed_size from all physical leaves below that field.
    std::vector<uint64_t> column_weights;
    column_weights.reserve(file_schema.num_fields());
    for (int field_index = 0; field_index < file_schema.num_fields(); ++field_index) {
      uint64_t field_weight = 0;
      for (int leaf_column_index = leaf_column_offsets_by_field[field_index];
           leaf_column_index < leaf_column_offsets_by_field[field_index + 1]; ++leaf_column_index) {
        auto uncompressed_size = row_group_meta->ColumnChunk(leaf_column_index)->total_uncompressed_size();
        if (uncompressed_size < 0) {
          return arrow::Status::Invalid(
              fmt::format("Parquet column uncompressed size is negative. [path={}, row_group={}, column={}]", path,
                          row_group_index, leaf_column_index));
        }
        auto column_weight = static_cast<uint64_t>(uncompressed_size);
        if (column_weight > std::numeric_limits<uint64_t>::max() - field_weight) {
          return arrow::Status::Invalid(
              fmt::format("Parquet top-level column size exceeds the uint64_t range. [path={}, row_group={}, field={}]",
                          path, row_group_index, field_index));
        }
        field_weight += column_weight;
      }
      column_weights.emplace_back(field_weight);
    }

    // The Parquet sizes above define only the relative column proportions.
    // Normalize them against the existing row-group memory estimate so the
    // resulting column sizes sum exactly to memory_size.
    ARROW_ASSIGN_OR_RAISE(row_group_infos[row_group_index].column_memory_sizes,
                          DistributeMemorySizes(row_group_infos[row_group_index].memory_size, column_weights));
  }
  return row_group_infos;
}

arrow::Result<std::vector<RowGroupInfo>> ParquetFormatReader::create_row_group_infos(
    const std::shared_ptr<::parquet::FileMetaData>& metadata) {
  assert(schema_);
  return create_row_group_infos_from_metadata(metadata, *schema_, path_);
}

ParquetFormatReader::ParquetFormatReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                         const std::string& path,
                                         const milvus_storage::api::Properties& properties,
                                         const std::vector<std::string>& needed_columns,
                                         const std::function<std::string(const std::string&)>& key_retriever,
                                         uint64_t file_size,
                                         uint64_t footer_size)
    : path_(path),
      fs_(fs),
      schema_(nullptr),
      properties_(properties),
      needed_columns_(needed_columns),
      key_retriever_(key_retriever),
      file_size_(file_size),
      footer_size_(footer_size),
      file_reader_(nullptr) {}

static ::parquet::ReaderProperties make_reader_properties(
    const std::function<std::string(const std::string&)>& key_retriever) {
  ::parquet::ReaderProperties reader_props = ::parquet::default_reader_properties();
  if (key_retriever) {
    reader_props.file_decryption_properties(::parquet::FileDecryptionProperties::Builder()
                                                .key_retriever(std::make_shared<KeyRetriever>(key_retriever))
                                                ->plaintext_files_allowed()
                                                ->build());
  }
  return reader_props;
}

// Parquet file trailer: [4B footer_length (LE)] [4B magic "PAR1"].
// Pre-read the parquet footer in a single IO instead of Arrow's default 2-step approach.
// Returns nullptr on any failure, letting the caller fall back to Arrow's normal path.
static std::shared_ptr<arrow::Buffer> try_read_footer_buffer(const std::shared_ptr<arrow::io::RandomAccessFile>& file,
                                                             uint64_t file_size,
                                                             uint64_t footer_size) {
#define PARQUET_MAGIC "PAR1"
#define PARQUET_MAGIC_SIZE 4
#define PARQUET_FOOTER_TRAILER_SIZE 8  // footer_length(4B) + magic(4B)

  const auto offset = static_cast<int64_t>(file_size - footer_size);
  const auto size = static_cast<int64_t>(footer_size);
  std::shared_ptr<arrow::Buffer> suffix;
#ifdef WITH_CRT
  if (auto* async_file = dynamic_cast<milvus_storage::NonBlockingReadAtFile*>(file.get())) {
    auto maybe_buf = arrow::AllocateResizableBuffer(size, file->io_context().pool());
    if (!maybe_buf.ok()) {
      return nullptr;
    }
    auto buf = std::move(maybe_buf).ValueOrDie();
    auto read_result = async_file->ReadAtAsyncInto(offset, size, buf->mutable_data()).result();
    if (!read_result.ok()) {
      return nullptr;
    }
    const auto bytes_read = *read_result;
    if (bytes_read < 0 || bytes_read > size || !buf->Resize(bytes_read).ok()) {
      return nullptr;
    }
    suffix = std::shared_ptr<arrow::Buffer>(std::move(buf));
  } else
#endif
  {
    auto suffix_result = file->ReadAt(offset, size);
    if (!suffix_result.ok()) {
      return nullptr;
    }
    suffix = *suffix_result;
  }
  if (!suffix || static_cast<uint64_t>(suffix->size()) < PARQUET_FOOTER_TRAILER_SIZE) {
    return nullptr;
  }
  const uint8_t* data = suffix->data();

  // Parse footer_length from the 4 bytes before the magic.
  uint32_t footer_length = 0;
  std::memcpy(&footer_length, data + suffix->size() - PARQUET_FOOTER_TRAILER_SIZE, sizeof(footer_length));

  // Validate magic bytes and ensure the suffix covers the entire Thrift metadata.
  if (std::memcmp(data + suffix->size() - PARQUET_MAGIC_SIZE, PARQUET_MAGIC, PARQUET_MAGIC_SIZE) != 0 ||
      footer_length + PARQUET_FOOTER_TRAILER_SIZE > static_cast<uint64_t>(suffix->size())) {
    return nullptr;
  }

  return suffix;

#undef PARQUET_FOOTER_TRAILER_SIZE
#undef PARQUET_MAGIC_SIZE
#undef PARQUET_MAGIC
}

static std::shared_ptr<::parquet::FileMetaData> try_parse_footer_metadata(
    const std::shared_ptr<arrow::Buffer>& suffix, const ::parquet::ReaderProperties& reader_props) {
  if (!suffix) {
    return nullptr;
  }

#define PARQUET_FOOTER_TRAILER_SIZE 8  // footer_length(4B) + magic(4B)
  if (static_cast<uint64_t>(suffix->size()) < PARQUET_FOOTER_TRAILER_SIZE) {
    return nullptr;
  }
  const uint8_t* data = suffix->data();
  uint32_t footer_length = 0;
  std::memcpy(&footer_length, data + suffix->size() - PARQUET_FOOTER_TRAILER_SIZE, sizeof(footer_length));
  if (footer_length + PARQUET_FOOTER_TRAILER_SIZE > static_cast<uint64_t>(suffix->size())) {
    return nullptr;
  }

  // Deserialize the Thrift FileMetaData from the suffix buffer.
  const uint8_t* thrift_data = data + suffix->size() - PARQUET_FOOTER_TRAILER_SIZE - footer_length;
  try {
    return ::parquet::FileMetaData::Make(thrift_data, &footer_length, reader_props);
  } catch (...) {
    return nullptr;
  }

#undef PARQUET_FOOTER_TRAILER_SIZE
}

static arrow::Result<std::unique_ptr<::parquet::arrow::FileReader>> create_parquet_file_reader(
    const std::shared_ptr<arrow::fs::FileSystem>& fs,
    const std::string& file_path,
    const milvus_storage::api::Properties& properties,
    const std::function<std::string(const std::string&)>& key_retriever,
    std::shared_ptr<::parquet::FileMetaData> metadata = nullptr,
    uint64_t file_size = 0,
    uint64_t footer_size = 0) {
  std::unique_ptr<::parquet::arrow::FileReader> result;

  ::parquet::arrow::FileReaderBuilder builder;
  auto reader_props = make_reader_properties(key_retriever);
  ::parquet::ArrowReaderProperties arrow_reader_props = ::parquet::default_arrow_reader_properties();
  if (key_retriever) {
    // Encrypted Parquet needs the parquet reader to initialize decryptors from
    // its own footer read path. Passing caller-supplied FileMetaData can leave
    // page decryptors incomplete.
    metadata = nullptr;
  }

  arrow_reader_props.set_batch_size(INT64_MAX);
  arrow_reader_props.set_pre_buffer(true);
  auto cache_options = arrow_reader_props.cache_options();
  auto hole_size_limit =
      milvus_storage::api::GetValueNoError<int64_t>(properties, PROPERTY_READER_PARQUET_PREBUFFER_HOLE_SIZE_LIMIT);
  auto range_size_limit =
      milvus_storage::api::GetValueNoError<int64_t>(properties, PROPERTY_READER_PARQUET_PREBUFFER_RANGE_SIZE_LIMIT);
  if (hole_size_limit > 0) {
    cache_options.hole_size_limit = hole_size_limit;
  }
  if (range_size_limit > 0) {
    cache_options.range_size_limit = range_size_limit;
  }
  if (cache_options.range_size_limit <= cache_options.hole_size_limit) {
    return arrow::Status::Invalid(fmt::format(
        "{} must be greater than {} for Arrow read-range coalescing. [range_size_limit={}, hole_size_limit={}]",
        PROPERTY_READER_PARQUET_PREBUFFER_RANGE_SIZE_LIMIT, PROPERTY_READER_PARQUET_PREBUFFER_HOLE_SIZE_LIMIT,
        cache_options.range_size_limit, cache_options.hole_size_limit));
  }
  arrow_reader_props.set_cache_options(cache_options);

  std::shared_ptr<arrow::io::RandomAccessFile> parquet_file;
  if (file_size > 0) {
    // Use pre-known file size to skip the S3 HEAD request that OpenInputFile(path) would trigger.
    arrow::fs::FileInfo file_info(file_path, arrow::fs::FileType::File);
    file_info.set_size(static_cast<int64_t>(file_size));
    ARROW_ASSIGN_OR_RAISE(parquet_file, fs->OpenInputFile(file_info));
  } else {
    ARROW_ASSIGN_OR_RAISE(parquet_file, fs->OpenInputFile(file_path));
  }

  if (!key_retriever && footer_size > 0 && !metadata && file_size > 0 && footer_size <= file_size) {
    auto footer_buffer = try_read_footer_buffer(parquet_file, file_size, footer_size);
    metadata = try_parse_footer_metadata(footer_buffer, reader_props);
  }

  ARROW_RETURN_NOT_OK(builder.Open(std::move(parquet_file), reader_props, metadata));
  ARROW_RETURN_NOT_OK(
      builder.memory_pool(arrow::default_memory_pool())->properties(arrow_reader_props)->Build(&result));
  return std::move(result);
}

std::string ParquetFormatReader::MetaTrait::cache_key(const api::ColumnGroupFile& file) {
  const auto file_size = file.Get<uint64_t>(api::kPropertyFileSize);
  const auto footer_size = file.Get<uint64_t>(api::kPropertyFooterSize);

  std::string key = fmt::format("parquet:path={};file_size={};footer_size={}", file.path, file_size, footer_size);
  auto metadata_it = file.properties.find(api::kPropertyMetadata);
  if (metadata_it != file.properties.end()) {
    key += fmt::format(";metadata_size={};metadata={}", metadata_it->second.size(), metadata_it->second);
  }
  return key;
}

arrow::Result<ParquetFormatReader::MetaTrait::MetadataPtr> ParquetFormatReader::MetaTrait::load_metadata(
    const api::ColumnGroupFile& file,
    const api::Properties& properties,
    const milvus_storage::KeyRetriever& key_retriever) {
  ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties, file.path));
  ARROW_ASSIGN_OR_RAISE(auto uri, StorageUri::Parse(file.path));

  const auto file_size = file.Get<uint64_t>(api::kPropertyFileSize);
  const auto footer_size = file.Get<uint64_t>(api::kPropertyFooterSize);

  ARROW_ASSIGN_OR_RAISE(auto file_reader, create_parquet_file_reader(fs, uri.key, properties, key_retriever,
                                                                     nullptr /* metadata */, file_size, footer_size));
  if (!file_reader->parquet_reader()) {
    return arrow::Status::Invalid(fmt::format("Failed to open parquet reader metadata. [path={}]", uri.key));
  }

  auto parquet_metadata = file_reader->parquet_reader()->metadata();
  if (!parquet_metadata) {
    return arrow::Status::Invalid(fmt::format("Failed to get parquet file metadata. [path={}]", uri.key));
  }

  std::shared_ptr<arrow::Schema> file_schema;
  ARROW_RETURN_NOT_OK(file_reader->GetSchema(&file_schema));
  if (!file_schema) {
    return arrow::Status::Invalid(fmt::format("Failed to get parquet file schema. [path={}]", uri.key));
  }

  ARROW_ASSIGN_OR_RAISE(auto row_group_infos,
                        create_row_group_infos_from_metadata(parquet_metadata, *file_schema, uri.key));

  auto metadata = std::make_shared<Metadata>();
  metadata->cache_key = cache_key(file);
  metadata->path = uri.key;
  metadata->file_schema = std::move(file_schema);
  metadata->row_group_infos = std::move(row_group_infos);
  metadata->cache_size = parquet_metadata->size();
  metadata->payload.fs = std::move(fs);
  if (!key_retriever) {
    metadata->payload.parquet_metadata = std::move(parquet_metadata);
  }
  metadata->payload.properties = properties;
  metadata->payload.key_retriever = key_retriever;

  MetadataPtr metadata_ptr = std::move(metadata);
  return metadata_ptr;
}

arrow::Result<std::shared_ptr<ParquetFormatReader>> ParquetFormatReader::MetaTrait::create_from_metadata(
    MetadataPtr metadata,
    const api::ColumnGroupFile& file,
    const std::shared_ptr<arrow::Schema>& /*read_schema*/,
    const std::vector<std::string>& needed_columns,
    const std::string& /*predicate*/) {
  if (!metadata) {
    return arrow::Status::Invalid("Cannot open parquet reader from null metadata");
  }
  if (!metadata->payload.fs) {
    return arrow::Status::Invalid(
        fmt::format("Cannot open parquet reader from metadata without filesystem. [path={}]", metadata->path));
  }
  if (!metadata->payload.key_retriever && !metadata->payload.parquet_metadata) {
    return arrow::Status::Invalid(
        fmt::format("Cannot open parquet reader from metadata without parquet footer. [path={}]", metadata->path));
  }

  std::shared_ptr<::parquet::FileMetaData> parquet_metadata;
  if (!metadata->payload.key_retriever) {
    parquet_metadata = metadata->payload.parquet_metadata;
  }

  const auto file_size = file.Get<uint64_t>(api::kPropertyFileSize);
  const auto footer_size = file.Get<uint64_t>(api::kPropertyFooterSize);

  ARROW_ASSIGN_OR_RAISE(
      auto file_reader,
      create_parquet_file_reader(metadata->payload.fs, metadata->path, metadata->payload.properties,
                                 metadata->payload.key_retriever, std::move(parquet_metadata), file_size, footer_size));

  auto reader =
      std::make_shared<ParquetFormatReader>(metadata->payload.fs, metadata->path, metadata->payload.properties,
                                            needed_columns, metadata->payload.key_retriever, file_size, footer_size);
  reader->schema_ = metadata->file_schema;
  reader->row_group_infos_ = metadata->row_group_infos;
  reader->file_reader_ = std::shared_ptr<::parquet::arrow::FileReader>(std::move(file_reader));
  ARROW_RETURN_NOT_OK(reader->set_needed_columns(needed_columns));
  return reader;
}

arrow::Status ParquetFormatReader::open() {
  assert(file_reader_ == nullptr);

  // create file reader
  ARROW_ASSIGN_OR_RAISE(auto file_reader, create_parquet_file_reader(fs_, path_, properties_, key_retriever_,
                                                                     nullptr /* metadata */, file_size_, footer_size_));
  file_reader_ = std::shared_ptr<::parquet::arrow::FileReader>(std::move(file_reader));

  // get the schema and create needed column indices
  std::shared_ptr<arrow::Schema> file_schema;
  ARROW_RETURN_NOT_OK(file_reader_->GetSchema(&file_schema));
  schema_ = file_schema;

  // create row group infos
  assert(file_reader_->parquet_reader() && "arrow logical fault");
  ARROW_ASSIGN_OR_RAISE(row_group_infos_, create_row_group_infos(file_reader_->parquet_reader()->metadata()));

  return set_needed_columns(needed_columns_);
}

arrow::Result<std::vector<RowGroupInfo>> ParquetFormatReader::get_row_group_infos() { return row_group_infos_; }

// Parquet uses a single schema_ (always derived from the file footer) rather than
// separate read_schema_/file_schema_ like Lance/Vortex. Projection needs both
// Arrow field indices and Parquet leaf column indices.
std::shared_ptr<arrow::Schema> ParquetFormatReader::get_schema() const { return schema_; }

static int get_leaf_column_count(const std::shared_ptr<arrow::DataType>& type) {
  if (type->id() == arrow::Type::EXTENSION) {
    const auto& extension_type = static_cast<const arrow::ExtensionType&>(*type);
    return get_leaf_column_count(extension_type.storage_type());
  }

  if (type->num_fields() == 0) {
    return 1;
  }

  int count = 0;
  for (int i = 0; i < type->num_fields(); ++i) {
    count += get_leaf_column_count(type->field(i)->type());
  }
  return count;
}

static std::vector<int> get_leaf_column_offsets_by_field(const arrow::Schema& file_schema) {
  std::vector<int> leaf_column_offsets_by_field;
  leaf_column_offsets_by_field.reserve(file_schema.num_fields() + 1);
  leaf_column_offsets_by_field.emplace_back(0);
  for (int field_index = 0; field_index < file_schema.num_fields(); ++field_index) {
    leaf_column_offsets_by_field.emplace_back(leaf_column_offsets_by_field.back() +
                                              get_leaf_column_count(file_schema.field(field_index)->type()));
  }
  return leaf_column_offsets_by_field;
}

static arrow::Result<std::vector<int>> get_leaf_column_indices(const arrow::Schema& file_schema,
                                                               const std::vector<std::string>& needed_columns,
                                                               const std::string& path) {
  // Build the start offset of each top-level Arrow field in the Parquet leaf-column list.
  // ReadRowGroup expects Parquet leaf-column indices, but needed_columns contains
  // top-level Arrow field names. Nested fields therefore need to expand to all
  // leaf columns under that top-level field, while preserving needed_columns order.
  // ex.
  //   schema fields: [id, user struct<age: int32, name: string>, score]
  //   Parquet leaf columns: [id, user.age, user.name, score]
  //   leaf_column_offsets_by_field: [0, 1, 3, 4]
  //   needed columns: [score, user]
  //   leaf column indices: [3, 1, 2]
  const auto leaf_column_offsets_by_field = get_leaf_column_offsets_by_field(file_schema);

  const int leaf_column_count = leaf_column_offsets_by_field.back();
  if (needed_columns.empty()) {
    // No projection: read every Parquet leaf column.
    std::vector<int> leaf_column_indices(leaf_column_count);
    std::iota(leaf_column_indices.begin(), leaf_column_indices.end(), 0);
    return leaf_column_indices;
  }

  // Resolve requested top-level fields in projection order and pre-compute output size.
  // Continue the example above:
  //   needed columns: [score, user]
  //   top-level field indices: [2, 1]
  //   projected leaf column count: 1 + 2 = 3
  std::vector<int> top_level_field_indices;
  top_level_field_indices.reserve(needed_columns.size());
  int projected_leaf_column_count = 0;
  for (const auto& column_name : needed_columns) {
    const int top_level_field_index = file_schema.GetFieldIndex(column_name);
    if (top_level_field_index < 0) {
      return arrow::Status::Invalid(fmt::format("Column '{}' not found in schema. [path={}]", column_name, path));
    }
    top_level_field_indices.emplace_back(top_level_field_index);
    projected_leaf_column_count +=
        leaf_column_offsets_by_field[top_level_field_index + 1] - leaf_column_offsets_by_field[top_level_field_index];
  }

  // Expand each top-level field to the leaf columns that Parquet ReadRowGroup expects.
  // The final order follows needed_columns, not file schema order.
  // ex.
  //   score -> leaf range [3, 4) -> [3]
  //   user  -> leaf range [1, 3) -> [1, 2]
  //   leaf column indices: [3, 1, 2]
  std::vector<int> leaf_column_indices;
  leaf_column_indices.reserve(projected_leaf_column_count);
  for (const int top_level_field_index : top_level_field_indices) {
    for (int column_index = leaf_column_offsets_by_field[top_level_field_index];
         column_index < leaf_column_offsets_by_field[top_level_field_index + 1]; ++column_index) {
      leaf_column_indices.emplace_back(column_index);
    }
  }

  return leaf_column_indices;
}

arrow::Status ParquetFormatReader::set_needed_columns(const std::vector<std::string>& needed_columns) {
  if (!schema_) {
    return arrow::Status::Invalid(fmt::format("Parquet file schema is not initialized. [path={}]", path_));
  }

  ARROW_ASSIGN_OR_RAISE(auto leaf_column_indices, get_leaf_column_indices(*schema_, needed_columns, path_));

  needed_columns_ = needed_columns;
  projected_leaf_column_indices_ = std::move(leaf_column_indices);

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ParquetFormatReader::get_chunk(const int& row_group_index) {
  std::shared_ptr<arrow::Table> table;
  assert(file_reader_);

  if (row_group_index >= row_group_infos_.size()) {
    return arrow::Status::Invalid(
        fmt::format("Row group index out of range [path={}, row_group_index={}, row_group_infos={}]", path_,
                    row_group_index, row_group_infos_.size()));
  }

  ARROW_RETURN_NOT_OK(file_reader_->ReadRowGroup(row_group_index, projected_leaf_column_indices_, &table));

  if (!table) {
    return arrow::Status::Invalid(
        fmt::format("Failed to read row group. Rowgroup Info: {}", row_group_infos_[row_group_index].ToString()));
  }

  return milvus_storage::ConvertTableToRecordBatch(table, false);
}

arrow::Result<std::shared_ptr<arrow::Table>> ParquetFormatReader::get_chunks_internal(
    const std::vector<int>& rg_indices_in_file) {
  std::shared_ptr<arrow::Table> table;
  assert(file_reader_);

  ARROW_RETURN_NOT_OK(file_reader_->ReadRowGroups(rg_indices_in_file, projected_leaf_column_indices_, &table));

  if (!table) {
    return arrow::Status::Invalid(fmt::format("Failed to read row groups. [path={}]", path_));
  }
  return table;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ParquetFormatReader::get_chunks(
    const std::vector<int>& rg_indices_in_file) {
  std::shared_ptr<arrow::Table> table;
  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  std::unique_ptr<arrow::RecordBatchReader> rb_reader;
  assert(file_reader_);

  ARROW_ASSIGN_OR_RAISE(table, get_chunks_internal(rg_indices_in_file));
  rb_reader = std::make_unique<arrow::TableBatchReader>(*table);

  std::shared_ptr<arrow::RecordBatch> rb;
  while (true) {
    ARROW_RETURN_NOT_OK(rb_reader->ReadNext(&rb));
    if (!rb) {
      break;
    }
    result.emplace_back(rb);
  }
  ARROW_RETURN_NOT_OK(rb_reader->Close());

  return result;
}

arrow::Result<std::vector<int>> ParquetFormatReader::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  std::vector<int> chunk_indices;
  chunk_indices.reserve(row_indices.size());

  for (const auto& row_index : row_indices) {
    auto it = std::upper_bound(row_group_infos_.begin(), row_group_infos_.end(), row_index,
                               [](int64_t val, const RowGroupInfo& info) { return val < info.start_offset; });

    bool found = false;
    if (it != row_group_infos_.begin()) {
      auto prev = std::prev(it);
      if (row_index < prev->end_offset) {
        chunk_indices.emplace_back(std::distance(row_group_infos_.begin(), prev));
        found = true;
      }
    }

    if (!found) {
      return arrow::Status::Invalid(
          fmt::format("Row index out of range: {}. [path={}, valid_range=[0, {}]]", row_index, path_,
                      row_group_infos_.empty() ? "0" : std::to_string(row_group_infos_.back().end_offset)));
    }
  }
  assert(chunk_indices.size() == row_indices.size());

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::Table>> ParquetFormatReader::take(const std::vector<int64_t>& row_indices) {
  ARROW_ASSIGN_OR_RAISE(auto chunk_indices, get_chunk_indices(row_indices));
  assert(chunk_indices.size() == row_indices.size());

  // Deduplicate chunk indices
  auto unique_chunk_indices = chunk_indices;
  unique_chunk_indices.erase(std::unique(unique_chunk_indices.begin(), unique_chunk_indices.end()),
                             unique_chunk_indices.end());

  // The input row_indices must be sorted and unique
  ARROW_ASSIGN_OR_RAISE(auto table, get_chunks_internal(unique_chunk_indices));

  // Build a map of chunk_id -> offset in the result table
  std::unordered_map<int, int64_t> chunk_base_offsets;
  int64_t current_accumulated_rows = 0;
  for (int chunk_id : unique_chunk_indices) {
    chunk_base_offsets[chunk_id] = current_accumulated_rows;
    // Accumulate the number of rows for each chunk (end - start)
    const auto& rg_info = row_group_infos_[chunk_id];
    current_accumulated_rows += (rg_info.end_offset - rg_info.start_offset);
  }

  // Calculate take indices for each target row
  std::vector<int64_t> table_take_indices(row_indices.size());
  for (size_t i = 0; i < row_indices.size(); ++i) {
    int chunk_id = chunk_indices[i];
    // Formula: base_offset_in_table + (global_row_index - chunk_start_offset)
    table_take_indices[i] = chunk_base_offsets[chunk_id] + (row_indices[i] - row_group_infos_[chunk_id].start_offset);
  }

  ARROW_ASSIGN_OR_RAISE(table, CopySelectedRows(table, table_take_indices));
  return table;
}

// RangeRecordBatchReader: Uses ReadRowGroups for batch I/O (benefits from pre_buffer)
// instead of GetRecordBatchReader which does lazy per-row-group I/O
class RangeRecordBatchReader : public arrow::RecordBatchReader {
  public:
  RangeRecordBatchReader(std::shared_ptr<::parquet::arrow::FileReader> file_reader,
                         std::shared_ptr<arrow::Schema> schema,
                         std::vector<int> rg_indices,
                         std::vector<int> leaf_column_indices,
                         uint64_t first_rg_slice_offset,
                         uint64_t total_rows)
      : file_reader_(std::move(file_reader)),
        schema_(std::move(schema)),
        rg_indices_(std::move(rg_indices)),
        leaf_column_indices_(std::move(leaf_column_indices)),
        first_rg_slice_offset_(first_rg_slice_offset),
        total_rows_(total_rows) {}

  ~RangeRecordBatchReader() override = default;

  arrow::Status ReadNext(std::shared_ptr<::arrow::RecordBatch>* out) override {
    // Lazy load: read all row groups on first ReadNext call
    if (!loaded_) {
      ARROW_RETURN_NOT_OK(LoadData());
    }

    if (current_batch_index_ >= batches_.size()) {
      *out = nullptr;
      return arrow::Status::OK();
    }

    *out = batches_[current_batch_index_++];
    return arrow::Status::OK();
  }

  [[nodiscard]] std::shared_ptr<::arrow::Schema> schema() const override { return schema_; }

  private:
  arrow::Status LoadData() {
    // Read all row groups at once using ReadRowGroups (benefits from pre_buffer)
    std::shared_ptr<arrow::Table> table;
    ARROW_RETURN_NOT_OK(file_reader_->ReadRowGroups(rg_indices_, leaf_column_indices_, &table));

    if (!table) {
      return arrow::Status::Invalid("Failed to read row groups");
    }

    // Apply slicing if needed
    if (first_rg_slice_offset_ > 0 || static_cast<uint64_t>(table->num_rows()) > total_rows_ + first_rg_slice_offset_) {
      table = table->Slice(first_rg_slice_offset_, total_rows_);
    }

    // Convert table to record batches
    ARROW_ASSIGN_OR_RAISE(batches_, ConvertTableToRecordBatchs(table));

    loaded_ = true;
    return arrow::Status::OK();
  }

  std::shared_ptr<::parquet::arrow::FileReader> file_reader_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<int> rg_indices_;
  std::vector<int> leaf_column_indices_;
  uint64_t first_rg_slice_offset_;
  uint64_t total_rows_;

  bool loaded_{false};
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  size_t current_batch_index_{0};
};

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> ParquetFormatReader::read_with_range(
    const uint64_t& start_offset, const uint64_t& end_offset) {
  if (row_group_infos_.empty()) {
    return arrow::Status::Invalid(fmt::format("Empty row group infos. [path={}]", path_));
  }

  if (start_offset >= end_offset || start_offset < row_group_infos_.front().start_offset ||
      end_offset > row_group_infos_.back().end_offset) {
    return arrow::Status::Invalid(
        fmt::format("Invalid range: start_offset={}, end_offset={}. [path={}, valid_range=[{}, {}]]", start_offset,
                    end_offset, path_, row_group_infos_.front().start_offset, row_group_infos_.back().end_offset));
  }

  std::vector<int> rg_indices;
  uint64_t first_rg_start_offset = 0;
  bool first_rg_found = false;

  for (size_t i = 0; i < row_group_infos_.size(); ++i) {
    const auto& rg_info = row_group_infos_[i];
    uint64_t rg_start = rg_info.start_offset;
    uint64_t rg_end = rg_info.end_offset;

    if (rg_end > start_offset && rg_start < end_offset) {
      rg_indices.emplace_back(i);
      if (!first_rg_found) {
        first_rg_start_offset = rg_start;
        first_rg_found = true;
      }
    }
  }

  assert(end_offset <= row_group_infos_[rg_indices.back()].end_offset);
  uint64_t first_rg_slice_offset = start_offset - first_rg_start_offset;
  uint64_t total_rows = end_offset - start_offset;

  auto projected_schema = schema_;
  if (!needed_columns_.empty()) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(needed_columns_.size());
    for (const auto& column_name : needed_columns_) {
      const int top_level_field_index = schema_->GetFieldIndex(column_name);
      if (top_level_field_index < 0) {
        return arrow::Status::Invalid(fmt::format("Column '{}' not found in schema. [path={}]", column_name, path_));
      }
      fields.emplace_back(schema_->field(top_level_field_index));
    }
    projected_schema = arrow::schema(std::move(fields));
  }

  // Use RangeRecordBatchReader which internally uses ReadRowGroups for batch I/O
  return std::make_shared<RangeRecordBatchReader>(file_reader_, projected_schema, std::move(rg_indices),
                                                  projected_leaf_column_indices_, first_rg_slice_offset, total_rows);
}

arrow::Result<std::shared_ptr<FormatReader>> ParquetFormatReader::clone_reader() {
  assert(file_reader_);

  ARROW_ASSIGN_OR_RAISE(auto parquet_reader, create_parquet_file_reader(fs_, path_, properties_, key_retriever_,
                                                                        file_reader_->parquet_reader()->metadata(),
                                                                        file_size_, footer_size_));
  return std::shared_ptr<ParquetFormatReader>(new ParquetFormatReader(*this, std::move(parquet_reader)));
}

ParquetFormatReader::ParquetFormatReader(const ParquetFormatReader& other,
                                         std::unique_ptr<::parquet::arrow::FileReader> cloned_file_reader)
    : path_(other.path_),
      fs_(other.fs_),
      schema_(other.schema_),
      properties_(other.properties_),
      needed_columns_(other.needed_columns_),
      key_retriever_(other.key_retriever_),
      file_size_(other.file_size_),
      footer_size_(other.footer_size_),
      projected_leaf_column_indices_(other.projected_leaf_column_indices_),
      row_group_infos_(other.row_group_infos_),
      file_reader_(std::move(cloned_file_reader)) {}

}  // namespace milvus_storage::parquet
