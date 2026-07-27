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

#include "milvus-storage/format/paimon/paimon_format_reader.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>

#include <arrow/array/builder_primitive.h>
#include <arrow/array/builder_binary.h>
#include <arrow/c/bridge.h>
#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <arrow/util/byte_size.h>
#include <fmt/format.h>
#include <folly/json/json.h>

#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/paimon/paimon_common.h"

namespace milvus_storage::paimon {
namespace {

constexpr size_t kMaxPaimonMetadataBytes = 12 * 1024 * 1024;

// Conservative decoded Arrow widths for formats that do not expose per-group
// memory metadata. This remains independent of compressed file bytes: Segment
// sizing and load estimation keep the same logical-row/Take semantics as
// Iceberg, while storage cache cells no longer see the old one-byte placeholder.
constexpr uint64_t kVariableWidthColumnBytes = 32;
constexpr uint64_t kUnknownColumnBytes = 16;

struct ParsedMetadata {
  uint32_t version = 0;
  std::string read_path;
  std::string data_format;
  uint64_t record_count = 0;
  folly::dynamic deletion_file;
};

arrow::Result<ParsedMetadata> ParseMetadata(const std::string& json) {
  folly::dynamic value;
  try {
    value = folly::parseJson(json);
  } catch (const std::exception& error) {
    return arrow::Status::Invalid("Cannot parse Paimon metadata: ", error.what());
  }
  if (!value.isObject()) {
    return arrow::Status::Invalid("Paimon metadata must be a JSON object");
  }
  try {
    auto version = value.getDefault("version", 0).asInt();
    if (version != 1) {
      return arrow::Status::Invalid("Unsupported Paimon metadata version: ", version);
    }
    auto read_path = value.getDefault("read_path", "").asString();
    if (read_path != "direct-file" && read_path != "data-split") {
      return arrow::Status::Invalid("Invalid Paimon metadata read_path: ", read_path);
    }
    auto record_count = value.getDefault("record_count", 0).asInt();
    if (record_count < 0) {
      return arrow::Status::Invalid("Paimon metadata has negative record_count");
    }
    return ParsedMetadata{
        .version = static_cast<uint32_t>(version),
        .read_path = std::move(read_path),
        .data_format = value.getDefault("data_format", "").asString(),
        .record_count = static_cast<uint64_t>(record_count),
        .deletion_file = value.getDefault("deletion_file", nullptr),
    };
  } catch (const std::exception& error) {
    return arrow::Status::Invalid("Paimon metadata has invalid field types: ", error.what());
  }
}

uint64_t SaturatingMultiply(uint64_t left, uint64_t right) {
  if (left == 0 || right == 0) {
    return 0;
  }
  if (left > std::numeric_limits<uint64_t>::max() / right) {
    return std::numeric_limits<uint64_t>::max();
  }
  return left * right;
}

size_t EstimateMemorySize(uint64_t rows, uint64_t row_width) {
  auto bytes = SaturatingMultiply(rows, std::max<uint64_t>(row_width, 1));
  return static_cast<size_t>(std::min<uint64_t>(bytes, std::numeric_limits<size_t>::max()));
}

uint64_t EstimateFieldByteWidth(const std::shared_ptr<arrow::DataType>& type) {
  if (!type) {
    return kUnknownColumnBytes;
  }
  switch (type->id()) {
    case arrow::Type::FIXED_SIZE_LIST: {
      auto list = std::static_pointer_cast<arrow::FixedSizeListType>(type);
      return SaturatingMultiply(static_cast<uint64_t>(std::max<int32_t>(1, list->list_size())),
                                EstimateFieldByteWidth(list->value_type()));
    }
    case arrow::Type::STRUCT: {
      uint64_t width = 0;
      for (const auto& field : type->fields()) {
        auto field_width = EstimateFieldByteWidth(field->type());
        width = field_width > std::numeric_limits<uint64_t>::max() - width
                    ? std::numeric_limits<uint64_t>::max()
                    : width + field_width;
      }
      return std::max<uint64_t>(width, 1);
    }
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
    case arrow::Type::STRING_VIEW:
    case arrow::Type::BINARY:
    case arrow::Type::LARGE_BINARY:
    case arrow::Type::BINARY_VIEW:
    case arrow::Type::LIST:
    case arrow::Type::LARGE_LIST:
    case arrow::Type::MAP:
      return kVariableWidthColumnBytes;
    default: {
      auto width = type->byte_width();
      return width > 0 ? static_cast<uint64_t>(width) : kUnknownColumnBytes;
    }
  }
}

// Estimated decoded bytes per row from the file schema.
uint64_t EstimateRowByteWidth(const std::shared_ptr<arrow::Schema>& schema) {
  if (!schema || schema->num_fields() == 0) {
    return static_cast<uint64_t>(kUnknownColumnBytes);
  }
  uint64_t width = 0;
  for (const auto& field : schema->fields()) {
    auto field_width = EstimateFieldByteWidth(field->type());
    width = field_width > std::numeric_limits<uint64_t>::max() - width
                ? std::numeric_limits<uint64_t>::max()
                : width + field_width;
  }
  return std::max<uint64_t>(width, 1);
}

std::vector<uint64_t> EstimateColumnByteWidths(const std::shared_ptr<arrow::Schema>& schema) {
  std::vector<uint64_t> widths;
  if (!schema) {
    return widths;
  }
  widths.reserve(schema->num_fields());
  for (const auto& field : schema->fields()) {
    widths.push_back(EstimateFieldByteWidth(field->type()));
  }
  return widths;
}

arrow::Result<std::vector<RowGroupInfo>> MakeLogicalRowGroups(uint64_t row_count,
                                                              uint64_t chunk_rows,
                                                              uint64_t row_byte_width,
                                                              const std::vector<uint64_t>& column_byte_widths) {
  std::vector<RowGroupInfo> groups;
  for (uint64_t start = 0; start < row_count;) {
    auto end = start + std::min(chunk_rows, row_count - start);
    auto memory_size = EstimateMemorySize(end - start, row_byte_width);
    std::vector<uint64_t> column_memory_sizes;
    if (!column_byte_widths.empty()) {
      // Distribute instead of multiplying each column out, so the per-column
      // parts always sum back to memory_size -- the invariant the column
      // group reader rescales against when it maps sizes onto field names.
      ARROW_ASSIGN_OR_RAISE(column_memory_sizes, DistributeMemorySizes(memory_size, column_byte_widths));
    }
    groups.push_back(RowGroupInfo{.start_offset = static_cast<size_t>(start),
                                  .end_offset = static_cast<size_t>(end),
                                  .memory_size = memory_size,
                                  .column_memory_sizes = std::move(column_memory_sizes)});
    start = end;
  }
  return groups;
}

arrow::Result<std::vector<RowGroupInfo>> MakeDirectLogicalRowGroups(const std::vector<RowGroupInfo>& physical,
                                                                    const std::vector<int64_t>& deletions) {
  if (deletions.empty()) {
    return physical;
  }
  std::vector<RowGroupInfo> result;
  result.reserve(physical.size());
  uint64_t logical_start = 0;
  for (const auto& group : physical) {
    if (group.end_offset < group.start_offset) {
      return arrow::Status::Invalid("Invalid physical Paimon row group: ", group.ToString());
    }
    auto first = std::lower_bound(deletions.begin(), deletions.end(), static_cast<int64_t>(group.start_offset));
    auto last = std::lower_bound(deletions.begin(), deletions.end(), static_cast<int64_t>(group.end_offset));
    auto deleted = static_cast<uint64_t>(std::distance(first, last));
    auto physical_rows = static_cast<uint64_t>(group.end_offset - group.start_offset);
    if (deleted > physical_rows) {
      return arrow::Status::Invalid("Paimon deletion count exceeds physical row group size");
    }
    auto logical_rows = physical_rows - deleted;
    uint64_t logical_memory_size = 0;
    if (physical_rows != 0) {
      logical_memory_size =
          static_cast<uint64_t>(static_cast<unsigned __int128>(group.memory_size) * logical_rows / physical_rows);
    }
    std::vector<uint64_t> logical_column_memory_sizes;
    if (group.memory_size_available) {
      ARROW_ASSIGN_OR_RAISE(logical_column_memory_sizes,
                            DistributeMemorySizes(logical_memory_size, group.column_memory_sizes));
    }
    result.push_back(RowGroupInfo{.start_offset = static_cast<size_t>(logical_start),
                                  .end_offset = static_cast<size_t>(logical_start + logical_rows),
                                  .memory_size = logical_memory_size,
                                  .column_memory_sizes = std::move(logical_column_memory_sizes),
                                  .memory_size_available = group.memory_size_available});
    logical_start += logical_rows;
  }
  return result;
}

arrow::Result<std::shared_ptr<arrow::Schema>> ProjectSchema(const std::shared_ptr<arrow::Schema>& file_schema,
                                                            const std::shared_ptr<arrow::Schema>& read_schema,
                                                            const std::vector<std::string>& columns) {
  if (read_schema) {
    return read_schema;
  }
  if (!file_schema) {
    return arrow::Status::Invalid("Paimon file schema is unavailable");
  }
  if (columns.empty()) {
    return file_schema;
  }
  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(columns.size());
  for (const auto& column : columns) {
    auto field = file_schema->GetFieldByName(column);
    if (!field) {
      return arrow::Status::Invalid("Paimon column not found: ", column);
    }
    fields.push_back(std::move(field));
  }
  return arrow::schema(std::move(fields));
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> FilterBatch(const std::shared_ptr<arrow::RecordBatch>& batch,
                                                               uint64_t physical_start,
                                                               const std::vector<int64_t>& deletions) {
  if (deletions.empty() || batch->num_rows() == 0) {
    return batch;
  }
  std::vector<int64_t> keep;
  keep.reserve(batch->num_rows());
  auto deletion = std::lower_bound(deletions.begin(), deletions.end(), static_cast<int64_t>(physical_start));
  for (int64_t row = 0; row < batch->num_rows(); ++row) {
    auto physical = static_cast<int64_t>(physical_start) + row;
    while (deletion != deletions.end() && *deletion < physical) {
      ++deletion;
    }
    if (deletion == deletions.end() || *deletion != physical) {
      keep.push_back(row);
    }
  }
  if (keep.size() == static_cast<size_t>(batch->num_rows())) {
    return batch;
  }
  arrow::Int64Builder builder;
  ARROW_RETURN_NOT_OK(builder.AppendValues(keep));
  ARROW_ASSIGN_OR_RAISE(auto indices, builder.Finish());
  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(batch->num_columns());
  for (const auto& column : batch->columns()) {
    if (column->type_id() == arrow::Type::STRING_VIEW) {
      auto values = std::static_pointer_cast<arrow::StringViewArray>(column);
      arrow::StringViewBuilder output;
      ARROW_RETURN_NOT_OK(output.Reserve(static_cast<int64_t>(keep.size())));
      for (auto row : keep) {
        if (values->IsNull(row)) {
          ARROW_RETURN_NOT_OK(output.AppendNull());
        } else {
          ARROW_RETURN_NOT_OK(output.Append(values->GetView(row)));
        }
      }
      ARROW_ASSIGN_OR_RAISE(auto array, output.Finish());
      columns.push_back(std::move(array));
      continue;
    }
    if (column->type_id() == arrow::Type::BINARY_VIEW) {
      auto values = std::static_pointer_cast<arrow::BinaryViewArray>(column);
      arrow::BinaryViewBuilder output;
      ARROW_RETURN_NOT_OK(output.Reserve(static_cast<int64_t>(keep.size())));
      for (auto row : keep) {
        if (values->IsNull(row)) {
          ARROW_RETURN_NOT_OK(output.AppendNull());
        } else {
          ARROW_RETURN_NOT_OK(output.Append(values->GetView(row)));
        }
      }
      ARROW_ASSIGN_OR_RAISE(auto array, output.Finish());
      columns.push_back(std::move(array));
      continue;
    }
    ARROW_ASSIGN_OR_RAISE(auto selected, arrow::compute::Take(column, indices));
    columns.push_back(selected.make_array());
  }
  return arrow::RecordBatch::Make(batch->schema(), static_cast<int64_t>(keep.size()), std::move(columns));
}

class DirectDeletionReader final : public arrow::RecordBatchReader {
  public:
  DirectDeletionReader(std::shared_ptr<arrow::RecordBatchReader> source,
                       uint64_t physical_start,
                       std::shared_ptr<const std::vector<int64_t>> deletions)
      : source_(std::move(source)), physical_position_(physical_start), deletions_(std::move(deletions)) {}

  std::shared_ptr<arrow::Schema> schema() const override { return source_->schema(); }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
    while (true) {
      ARROW_RETURN_NOT_OK(source_->ReadNext(batch));
      if (!*batch) {
        return arrow::Status::OK();
      }
      auto physical_start = physical_position_;
      physical_position_ += static_cast<uint64_t>((*batch)->num_rows());
      ARROW_ASSIGN_OR_RAISE(*batch, FilterBatch(*batch, physical_start, *deletions_));
      // A physical batch may be completely deleted between two live logical
      // ranges. Do not expose that intermediate empty batch: callers consume
      // this reader as one contiguous logical range.
      if ((*batch)->num_rows() != 0) {
        return arrow::Status::OK();
      }
    }
  }

  private:
  std::shared_ptr<arrow::RecordBatchReader> source_;
  uint64_t physical_position_;
  std::shared_ptr<const std::vector<int64_t>> deletions_;
};

class RangeRecordBatchReader final : public arrow::RecordBatchReader {
  public:
  RangeRecordBatchReader(std::shared_ptr<arrow::RecordBatchReader> source,
                         uint64_t skip,
                         uint64_t rows,
                         bool validate_eof = false)
      : source_(std::move(source)), skip_(skip), remaining_(rows), validate_eof_(validate_eof) {}

  std::shared_ptr<arrow::Schema> schema() const override { return source_->schema(); }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* output) override {
    *output = nullptr;
    if (remaining_ == 0) {
      if (!validate_eof_ || eof_validated_) {
        return arrow::Status::OK();
      }
      if (has_buffered_rows_after_range_) {
        return arrow::Status::Invalid("Paimon data-split produced more rows than its declared row count");
      }
      while (true) {
        std::shared_ptr<arrow::RecordBatch> extra;
        ARROW_RETURN_NOT_OK(source_->ReadNext(&extra));
        if (!extra) {
          eof_validated_ = true;
          return arrow::Status::OK();
        }
        if (extra->num_rows() != 0) {
          return arrow::Status::Invalid("Paimon data-split produced more rows than its declared row count");
        }
      }
    }
    while (remaining_ > 0) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ARROW_RETURN_NOT_OK(source_->ReadNext(&batch));
      if (!batch) {
        return arrow::Status::Invalid("Paimon data-split ended before the requested range was complete: skip=", skip_,
                                      ", remaining=", remaining_);
      }
      auto available = static_cast<uint64_t>(batch->num_rows());
      if (skip_ >= available) {
        skip_ -= available;
        continue;
      }
      auto start = skip_;
      skip_ = 0;
      auto count = std::min(remaining_, available - start);
      remaining_ -= count;
      if (validate_eof_ && count < available - start) {
        has_buffered_rows_after_range_ = true;
      }
      *output = batch->Slice(static_cast<int64_t>(start), static_cast<int64_t>(count));
      return arrow::Status::OK();
    }
    return arrow::Status::OK();
  }

  private:
  std::shared_ptr<arrow::RecordBatchReader> source_;
  uint64_t skip_;
  uint64_t remaining_;
  bool validate_eof_;
  bool eof_validated_ = false;
  bool has_buffered_rows_after_range_ = false;
};

arrow::Result<std::shared_ptr<arrow::RecordBatch>> CombineBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, const std::shared_ptr<arrow::Schema>& schema) {
  if (batches.empty()) {
    return arrow::RecordBatch::MakeEmpty(schema);
  }
  if (batches.size() == 1) {
    return batches.front();
  }
  ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches(schema, batches));
  return table->CombineChunksToBatch();
}

}  // namespace

class DataSplitStreamCursor {
  public:
  DataSplitStreamCursor(std::shared_ptr<arrow::RecordBatchReader> source, uint64_t declared_rows)
      : source_(std::move(source)), declared_rows_(declared_rows) {}

  uint64_t position() const { return position_; }

  arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ReadRange(uint64_t start, uint64_t end) {
    if (end < start) {
      return arrow::Status::Invalid("Invalid Paimon range: end is before start");
    }
    if (start < position_) {
      return arrow::Status::Invalid("Paimon stream cursor cannot seek backwards");
    }
    ARROW_RETURN_NOT_OK(Consume(start - position_, nullptr));
    std::vector<std::shared_ptr<arrow::RecordBatch>> output;
    ARROW_RETURN_NOT_OK(Consume(end - start, &output));
    if (end == declared_rows_) {
      ARROW_RETURN_NOT_OK(ValidateExhausted());
    }
    return output;
  }

  private:
  arrow::Status ValidateExhausted() {
    if (pending_ && pending_offset_ < static_cast<uint64_t>(pending_->num_rows())) {
      return arrow::Status::Invalid("Paimon data-split produced more rows than its declared row count");
    }
    pending_.reset();
    pending_offset_ = 0;
    while (true) {
      std::shared_ptr<arrow::RecordBatch> extra;
      ARROW_RETURN_NOT_OK(source_->ReadNext(&extra));
      if (!extra) {
        return arrow::Status::OK();
      }
      if (extra->num_rows() != 0) {
        return arrow::Status::Invalid("Paimon data-split produced more rows than its declared row count");
      }
    }
  }

  arrow::Status Consume(uint64_t rows, std::vector<std::shared_ptr<arrow::RecordBatch>>* output) {
    while (rows > 0) {
      if (!pending_ || pending_offset_ == static_cast<uint64_t>(pending_->num_rows())) {
        pending_.reset();
        pending_offset_ = 0;
        ARROW_RETURN_NOT_OK(source_->ReadNext(&pending_));
        if (!pending_) {
          return arrow::Status::Invalid("Paimon data-split ended before its declared row count");
        }
      }
      auto available = static_cast<uint64_t>(pending_->num_rows()) - pending_offset_;
      auto count = std::min(rows, available);
      if (output != nullptr) {
        output->push_back(pending_->Slice(static_cast<int64_t>(pending_offset_), static_cast<int64_t>(count)));
      }
      pending_offset_ += count;
      position_ += count;
      rows -= count;
    }
    return arrow::Status::OK();
  }

  std::shared_ptr<arrow::RecordBatchReader> source_;
  std::shared_ptr<arrow::RecordBatch> pending_;
  uint64_t pending_offset_ = 0;
  uint64_t position_ = 0;
  uint64_t declared_rows_;
};

std::string PaimonFormatReader::MetaTrait::cache_key(const api::ColumnGroupFile& file) {
  auto metadata = file.Get<std::string>(api::kPropertyMetadata);
  return fmt::format("paimon|path:{}|metadata:{}", file.path, metadata);
}

arrow::Result<PaimonFormatReader::MetaTrait::MetadataPtr> PaimonFormatReader::MetaTrait::load_metadata(
    const api::ColumnGroupFile& file, const api::Properties& properties, const KeyRetriever& key_retriever) {
  const auto metadata_json = file.Get<std::string>(api::kPropertyMetadata);
  if (metadata_json.empty()) {
    return arrow::Status::Invalid("Paimon column group is missing metadata");
  }
  if (metadata_json.size() > kMaxPaimonMetadataBytes) {
    return arrow::Status::Invalid("Paimon metadata is too large: ", metadata_json.size(), " bytes exceeds ",
                                  kMaxPaimonMetadataBytes);
  }
  ARROW_ASSIGN_OR_RAISE(auto parsed, ParseMetadata(metadata_json));
  ARROW_ASSIGN_OR_RAISE(auto fs_config, FilesystemCache::resolve_config(properties, file.path));
  StorageOptions storage_options;
  try {
    storage_options = ToStorageOptions(fs_config);
  } catch (const std::exception& error) {
    return arrow::Status::Invalid("Cannot create Paimon storage options: ", error.what());
  }
  ARROW_ASSIGN_OR_RAISE(auto logical_chunk_rows,
                        api::GetValue<uint64_t>(properties, PROPERTY_READER_LOGICAL_CHUNK_ROWS));

  auto metadata = std::make_shared<Metadata>();
  metadata->cache_key = cache_key(file);
  metadata->path = file.path;
  metadata->payload.read_path = parsed.read_path;
  metadata->payload.data_format = parsed.data_format;
  metadata->payload.metadata_json = metadata_json;
  metadata->payload.properties = properties;
  metadata->payload.storage_options = storage_options;
  metadata->payload.record_count = parsed.record_count;
  metadata->payload.logical_chunk_rows = logical_chunk_rows;

  if (parsed.read_path == "direct-file") {
    size_t direct_cache_size = 0;
    if (parsed.data_format == "parquet") {
      ARROW_ASSIGN_OR_RAISE(metadata->payload.parquet_metadata,
                            parquet::ParquetFormatReader::MetaTrait::load_metadata(file, properties, key_retriever));
      metadata->file_schema = metadata->payload.parquet_metadata->file_schema;
      metadata->payload.direct_physical_row_groups = metadata->payload.parquet_metadata->row_group_infos;
      direct_cache_size = metadata->payload.parquet_metadata->cache_size;
    } else if (parsed.data_format == "vortex") {
      ARROW_ASSIGN_OR_RAISE(metadata->payload.vortex_metadata,
                            vortex::VortexFormatReader::MetaTrait::load_metadata(file, properties, key_retriever));
      metadata->file_schema = metadata->payload.vortex_metadata->file_schema;
      metadata->payload.direct_physical_row_groups = metadata->payload.vortex_metadata->row_group_infos;
      direct_cache_size = metadata->payload.vortex_metadata->cache_size;
    } else {
      return arrow::Status::NotImplemented("Paimon direct-file does not support format: ", parsed.data_format);
    }
    auto& physical_groups = metadata->payload.direct_physical_row_groups;
    uint64_t physical_rows = physical_groups.empty() ? 0 : physical_groups.back().end_offset;
    metadata->payload.physical_row_count = physical_rows;

    // Some Vortex files produced by Paimon do not carry decoded-size
    // statistics. Keep the shared Vortex reader, but avoid publishing a zero
    // load estimate for non-empty files by falling back to schema widths.
    const auto column_widths = EstimateColumnByteWidths(metadata->file_schema);
    const auto row_width = EstimateRowByteWidth(metadata->file_schema);
    for (auto& group : physical_groups) {
      const auto rows = static_cast<uint64_t>(group.end_offset - group.start_offset);
      if (rows == 0 || group.memory_size != 0) {
        continue;
      }
      group.memory_size = EstimateMemorySize(rows, row_width);
      ARROW_ASSIGN_OR_RAISE(group.column_memory_sizes,
                            DistributeMemorySizes(group.memory_size, column_widths));
      group.memory_size_available = true;
    }

    auto deletions = std::make_shared<std::vector<int64_t>>();
    if (!parsed.deletion_file.isNull()) {
      if (!parsed.deletion_file.isObject()) {
        return arrow::Status::Invalid("Paimon deletion_file must be an object");
      }
      std::string path;
      int64_t offset = -1;
      int64_t length = -1;
      int64_t cardinality = -1;
      try {
        path = parsed.deletion_file.getDefault("path", "").asString();
        offset = parsed.deletion_file.getDefault("offset", -1).asInt();
        length = parsed.deletion_file.getDefault("length", -1).asInt();
        cardinality = parsed.deletion_file.getDefault("cardinality", -1).asInt();
      } catch (const std::exception& error) {
        return arrow::Status::Invalid("Paimon deletion_file has invalid field types: ", error.what());
      }
      if (path.empty() || offset < 0 || length < 0) {
        return arrow::Status::Invalid("Paimon deletion_file has invalid path or range");
      }
      try {
        auto positions = ReadDeletionVector(path, static_cast<uint64_t>(offset), static_cast<uint64_t>(length),
                                            cardinality, storage_options);
        deletions->reserve(positions.size());
        for (auto position : positions) {
          if (position > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return arrow::Status::Invalid("Paimon deletion position exceeds int64 range");
          }
          if (position >= physical_rows) {
            return arrow::Status::Invalid("Paimon deletion position exceeds physical row count");
          }
          deletions->push_back(static_cast<int64_t>(position));
        }
      } catch (const std::exception& error) {
        // bitmap64 deletion vectors surface as NotImplemented instead of a
        // retryable IOError; transient read failures stay IOError.
        return ClassifyPaimonError("Cannot read Paimon deletion vector", error);
      }
    }
    std::sort(deletions->begin(), deletions->end());
    if (std::adjacent_find(deletions->begin(), deletions->end()) != deletions->end()) {
      return arrow::Status::Invalid("Paimon deletion vector contains duplicate positions");
    }
    metadata->payload.sorted_deletions = deletions;
    ARROW_ASSIGN_OR_RAISE(metadata->row_group_infos, MakeDirectLogicalRowGroups(physical_groups, *deletions));
    auto logical_rows = metadata->row_group_infos.empty() ? 0 : metadata->row_group_infos.back().end_offset;
    if (logical_rows != parsed.record_count) {
      return arrow::Status::Invalid("Paimon direct-file row count mismatch: descriptor=", parsed.record_count,
                                    ", reader=", logical_rows);
    }
    metadata->cache_size = direct_cache_size + deletions->size() * sizeof(int64_t) + metadata_json.size() +
                           physical_groups.size() * sizeof(RowGroupInfo);
  } else {
    try {
      auto reader = BlockingPaimonDataSplitReader::Open(metadata_json, ToStandardUri(file.path), storage_options);
      ArrowSchema c_schema{};
      reader->ExportSchema(&c_schema);
      ARROW_ASSIGN_OR_RAISE(metadata->file_schema, arrow::ImportSchema(&c_schema));
      // Keep the handle: every reader created from this cached metadata
      // shares it, so schema and snapshot resolution happen once per
      // descriptor rather than once per reader instance.
      metadata->payload.split_reader_handle = std::move(reader);
    } catch (const std::exception& error) {
      // An expired pinned snapshot or corrupt descriptor is terminal
      // (Invalid); transient storage failures keep IOError semantics.
      return ClassifyPaimonError("Cannot load Paimon data-split metadata", error);
    }
    // Cache-cell sizing wants the decoded in-memory width, and a schema
    // estimate can only guess variable-length columns. Sample the first
    // merge-read batch through the shared handle and measure the real Arrow
    // width; the schema estimate stays as the fallback. Sampling failures are
    // non-fatal on purpose: a broken split fails on the actual read with full
    // error classification, not on a sizing hint.
    auto row_width = EstimateRowByteWidth(metadata->file_schema);
    auto column_widths = EstimateColumnByteWidths(metadata->file_schema);
    try {
      auto stream = metadata->payload.split_reader_handle->OpenStream({});
      auto sample_reader = arrow::ImportRecordBatchReader(&stream);
      if (sample_reader.ok()) {
        std::shared_ptr<arrow::RecordBatch> sample;
        if ((*sample_reader)->ReadNext(&sample).ok() && sample && sample->num_rows() > 0) {
          const auto sample_rows = static_cast<uint64_t>(sample->num_rows());
          row_width = std::max<uint64_t>(1, static_cast<uint64_t>(arrow::util::TotalBufferSize(*sample)) / sample_rows);
          // The sampled stream is unprojected, so its columns line up with
          // file_schema position for position -- the alignment the column
          // group reader requires before it maps sizes onto field names.
          // Anything else keeps the schema estimate rather than risk a
          // mislabelled column.
          if (sample->num_columns() == metadata->file_schema->num_fields()) {
            std::vector<uint64_t> sampled_widths;
            sampled_widths.reserve(static_cast<size_t>(sample->num_columns()));
            for (const auto& column : sample->columns()) {
              sampled_widths.push_back(
                  std::max<uint64_t>(1, static_cast<uint64_t>(arrow::util::TotalBufferSize(*column)) / sample_rows));
            }
            column_widths = std::move(sampled_widths);
          }
        }
      } else if (stream.release != nullptr) {
        stream.release(&stream);
      }
    } catch (const std::exception& error) {
      LOG_STORAGE_WARNING_ << "Paimon data-split row-width sampling failed, using the schema estimate: "
                           << error.what();
    }
    ARROW_ASSIGN_OR_RAISE(metadata->row_group_infos,
                          MakeLogicalRowGroups(parsed.record_count, logical_chunk_rows, row_width, column_widths));
    // The retained bridge handle pins table schema state on the Rust side;
    // charge a nominal footprint so the cache can still bound total entries.
    constexpr size_t kSplitReaderHandleBytes = 16 * 1024;
    metadata->cache_size = sizeof(Metadata) + metadata_json.size() +
                           metadata->row_group_infos.size() * sizeof(RowGroupInfo) + kSplitReaderHandleBytes;
  }

  MetadataPtr result = metadata;
  return result;
}

arrow::Result<std::shared_ptr<PaimonFormatReader>> PaimonFormatReader::MetaTrait::create_from_metadata(
    MetadataPtr metadata,
    const api::ColumnGroupFile& file,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& needed_columns,
    const std::string& predicate) {
  if (!metadata) {
    return arrow::Status::Invalid("Cannot create Paimon reader from null metadata");
  }
  ARROW_ASSIGN_OR_RAISE(auto output_schema, ProjectSchema(metadata->file_schema, read_schema, needed_columns));
  std::shared_ptr<FormatReader> direct_file_reader;
  std::shared_ptr<BlockingPaimonDataSplitReader> split_reader;
  std::vector<std::string> split_columns;
  if (metadata->payload.read_path == "direct-file") {
    if (metadata->payload.data_format == "parquet") {
      std::shared_ptr<parquet::ParquetFormatReader> parquet_reader;
      ARROW_ASSIGN_OR_RAISE(parquet_reader,
                            parquet::ParquetFormatReader::MetaTrait::create_from_metadata(
                                metadata->payload.parquet_metadata, file, read_schema, needed_columns, predicate));
      direct_file_reader = std::static_pointer_cast<FormatReader>(std::move(parquet_reader));
    } else if (metadata->payload.data_format == "vortex") {
      std::shared_ptr<vortex::VortexFormatReader> vortex_reader;
      ARROW_ASSIGN_OR_RAISE(vortex_reader,
                            vortex::VortexFormatReader::MetaTrait::create_from_metadata(
                                metadata->payload.vortex_metadata, file, read_schema, needed_columns, predicate));
      direct_file_reader = std::static_pointer_cast<FormatReader>(std::move(vortex_reader));
    } else {
      return arrow::Status::NotImplemented("Paimon direct-file does not support format: ",
                                           metadata->payload.data_format);
    }
  } else {
    split_columns = needed_columns;
    if (split_columns.empty() && read_schema) {
      split_columns = read_schema->field_names();
    }
    // Reuse the projection-agnostic handle owned by the cached metadata:
    // no per-reader schema/snapshot resolution happens here anymore.
    split_reader = metadata->payload.split_reader_handle;
    if (!split_reader) {
      return arrow::Status::Invalid("Paimon data-split reader handle is unavailable in cached metadata");
    }
  }
  return std::shared_ptr<PaimonFormatReader>(new PaimonFormatReader(
      std::move(metadata), file, read_schema, needed_columns, predicate, std::move(direct_file_reader),
      std::move(split_reader), std::move(split_columns), std::move(output_schema)));
}

PaimonFormatReader::PaimonFormatReader(MetaTrait::MetadataPtr metadata,
                                       api::ColumnGroupFile file,
                                       std::shared_ptr<arrow::Schema> read_schema,
                                       std::vector<std::string> needed_columns,
                                       std::string predicate,
                                       std::shared_ptr<FormatReader> direct_file_reader,
                                       std::shared_ptr<BlockingPaimonDataSplitReader> split_reader,
                                       std::vector<std::string> split_columns,
                                       std::shared_ptr<arrow::Schema> output_schema)
    : metadata_(std::move(metadata)),
      file_(std::move(file)),
      read_schema_(std::move(read_schema)),
      needed_columns_(std::move(needed_columns)),
      predicate_(std::move(predicate)),
      direct_file_reader_(std::move(direct_file_reader)),
      split_reader_(std::move(split_reader)),
      split_columns_(std::move(split_columns)),
      output_schema_(std::move(output_schema)),
      row_group_infos_(metadata_->row_group_infos) {}

bool PaimonFormatReader::is_data_split() const { return metadata_->payload.read_path == "data-split"; }

arrow::Status PaimonFormatReader::open() {
  if (is_data_split()) {
    ARROW_ASSIGN_OR_RAISE(data_split_cursor_, make_data_split_cursor());
  } else if (!direct_file_reader_) {
    return arrow::Status::Invalid("Paimon direct-file reader is unavailable");
  }
  return arrow::Status::OK();
}

arrow::Result<std::unique_ptr<DataSplitStreamCursor>> PaimonFormatReader::make_data_split_cursor() const {
  try {
    auto stream = split_reader_->OpenStream(split_columns_);
    ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ImportRecordBatchReader(&stream));
    return std::make_unique<DataSplitStreamCursor>(std::move(reader), metadata_->payload.record_count);
  } catch (const std::exception& error) {
    return arrow::Status::IOError("Cannot open Paimon data-split stream: ", error.what());
  }
}

arrow::Result<std::vector<RowGroupInfo>> PaimonFormatReader::get_row_group_infos() { return row_group_infos_; }

arrow::Result<std::shared_ptr<arrow::RecordBatch>> PaimonFormatReader::filter_direct_batch(
    const std::shared_ptr<arrow::RecordBatch>& batch, uint64_t physical_start) const {
  return FilterBatch(batch, physical_start, *metadata_->payload.sorted_deletions);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> PaimonFormatReader::get_chunk(const int& row_group_index) {
  if (row_group_index < 0 || static_cast<size_t>(row_group_index) >= row_group_infos_.size()) {
    return arrow::Status::Invalid("Paimon row group index out of range: ", row_group_index);
  }
  const auto& group = row_group_infos_[row_group_index];
  if (is_data_split()) {
    if (!data_split_cursor_ || data_split_cursor_->position() > group.start_offset) {
      if (data_split_cursor_) {
        // Backward seek: the merge-read stream is forward-only, so this
        // restarts it and replays the prefix. GetPaimonDataSplitStreamOpenCount
        // exposes the restart count.
        LOG_STORAGE_DEBUG_ << "Restarting Paimon data-split stream for backward seek: position="
                           << data_split_cursor_->position() << ", requested_start=" << group.start_offset;
      }
      ARROW_ASSIGN_OR_RAISE(data_split_cursor_, make_data_split_cursor());
    }
    ARROW_ASSIGN_OR_RAISE(auto batches, data_split_cursor_->ReadRange(group.start_offset, group.end_offset));
    return CombineBatches(batches, output_schema_);
  }
  if (group.start_offset == group.end_offset) {
    return arrow::RecordBatch::MakeEmpty(output_schema_);
  }
  const auto& physical = metadata_->payload.direct_physical_row_groups[row_group_index];
  ARROW_ASSIGN_OR_RAISE(auto batch, direct_file_reader_->get_chunk(row_group_index));
  return filter_direct_batch(batch, physical.start_offset);
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> PaimonFormatReader::get_chunks(
    const std::vector<int>& indices) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> output;
  output.reserve(indices.size());
  if (indices.empty()) {
    return output;
  }
  if (is_data_split()) {
    ARROW_ASSIGN_OR_RAISE(auto cursor, make_data_split_cursor());
    int previous = -1;
    for (auto index : indices) {
      if (index < previous || index < 0 || static_cast<size_t>(index) >= row_group_infos_.size()) {
        return arrow::Status::Invalid("Paimon get_chunks requires sorted valid row group indices");
      }
      const auto& group = row_group_infos_[index];
      ARROW_ASSIGN_OR_RAISE(auto batches, cursor->ReadRange(group.start_offset, group.end_offset));
      ARROW_ASSIGN_OR_RAISE(auto batch, CombineBatches(batches, output_schema_));
      output.push_back(std::move(batch));
      previous = index;
    }
    return output;
  }
  for (auto index : indices) {
    if (index < 0 || static_cast<size_t>(index) >= metadata_->payload.direct_physical_row_groups.size()) {
      return arrow::Status::Invalid("Paimon direct-file row group index out of range");
    }
  }
  ARROW_ASSIGN_OR_RAISE(auto batches, direct_file_reader_->get_chunks(indices));
  if (batches.size() != indices.size()) {
    return arrow::Status::Invalid("Direct-file reader returned an unexpected Paimon chunk count");
  }
  for (size_t i = 0; i < batches.size(); ++i) {
    const auto& physical = metadata_->payload.direct_physical_row_groups[indices[i]];
    ARROW_ASSIGN_OR_RAISE(auto filtered, filter_direct_batch(batches[i], physical.start_offset));
    output.push_back(std::move(filtered));
  }
  return output;
}

int64_t PaimonFormatReader::logical_to_physical(int64_t logical_offset) const {
  const auto& deletions = *metadata_->payload.sorted_deletions;
  int64_t physical = logical_offset;
  const auto physical_rows = static_cast<int64_t>(metadata_->payload.physical_row_count);
  for (size_t iteration = 0; iteration <= deletions.size(); ++iteration) {
    auto deleted =
        static_cast<int64_t>(std::upper_bound(deletions.begin(), deletions.end(), physical) - deletions.begin());
    auto next = logical_offset + deleted;
    if (next == physical) {
      return std::min(physical, physical_rows);
    }
    physical = std::min(next, physical_rows);
  }
  return physical_rows;
}

arrow::Result<std::shared_ptr<arrow::Table>> PaimonFormatReader::take(const std::vector<int64_t>& indices) {
  if (indices.empty()) {
    return arrow::Table::MakeEmpty(output_schema_);
  }
  if (!is_data_split()) {
    std::vector<int64_t> physical;
    physical.reserve(indices.size());
    for (auto index : indices) {
      physical.push_back(logical_to_physical(index));
    }
    for (auto index : physical) {
      if (index < 0 || static_cast<uint64_t>(index) >= metadata_->payload.physical_row_count) {
        return arrow::Status::Invalid("Paimon direct-file take index is out of range");
      }
    }
    return direct_file_reader_->take(physical);
  }
  ARROW_ASSIGN_OR_RAISE(auto cursor, make_data_split_cursor());
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  int64_t previous = -1;
  size_t begin = 0;
  while (begin < indices.size()) {
    if (indices[begin] <= previous || indices[begin] < 0 ||
        static_cast<uint64_t>(indices[begin]) >= metadata_->payload.record_count) {
      return arrow::Status::Invalid("Paimon take indices must be sorted, unique, and in range");
    }
    size_t end = begin + 1;
    while (end < indices.size() && indices[end] == indices[end - 1] + 1) {
      ++end;
    }
    ARROW_ASSIGN_OR_RAISE(auto range_batches, cursor->ReadRange(static_cast<uint64_t>(indices[begin]),
                                                                static_cast<uint64_t>(indices[end - 1]) + 1));
    batches.insert(batches.end(), range_batches.begin(), range_batches.end());
    previous = indices[end - 1];
    begin = end;
  }
  return arrow::Table::FromRecordBatches(output_schema_, batches);
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> PaimonFormatReader::read_with_range(const uint64_t& start,
                                                                                             const uint64_t& end) {
  if (end < start || end > metadata_->payload.record_count) {
    return arrow::Status::Invalid("Invalid Paimon logical range");
  }
  if (start == end) {
    ARROW_ASSIGN_OR_RAISE(auto empty, arrow::RecordBatch::MakeEmpty(output_schema_));
    return arrow::RecordBatchReader::Make({empty});
  }
  if (is_data_split()) {
    try {
      auto stream = split_reader_->OpenStream(split_columns_);
      ARROW_ASSIGN_OR_RAISE(auto source, arrow::ImportRecordBatchReader(&stream));
      return std::make_shared<RangeRecordBatchReader>(std::move(source), start, end - start,
                                                      start == 0 && end == metadata_->payload.record_count);
    } catch (const std::exception& error) {
      return arrow::Status::IOError("Cannot open Paimon range stream: ", error.what());
    }
  }
  auto physical_start = static_cast<uint64_t>(logical_to_physical(static_cast<int64_t>(start)));
  auto physical_end = static_cast<uint64_t>(logical_to_physical(static_cast<int64_t>(end - 1)) + 1);
  ARROW_ASSIGN_OR_RAISE(auto source, direct_file_reader_->read_with_range(physical_start, physical_end));
  return std::make_shared<DirectDeletionReader>(std::move(source), physical_start, metadata_->payload.sorted_deletions);
}

arrow::Result<std::shared_ptr<FormatReader>> PaimonFormatReader::clone_reader() {
  ARROW_ASSIGN_OR_RAISE(auto cloned,
                        MetaTrait::create_from_metadata(metadata_, file_, read_schema_, needed_columns_, predicate_));
  ARROW_RETURN_NOT_OK(cloned->open());
  return std::static_pointer_cast<FormatReader>(cloned);
}

std::shared_ptr<arrow::Schema> PaimonFormatReader::get_schema() const { return metadata_->file_schema; }

}  // namespace milvus_storage::paimon
