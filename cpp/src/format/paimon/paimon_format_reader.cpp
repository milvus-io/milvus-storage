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
#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/paimon/paimon_common.h"
#include "paimon_bridge.h"

namespace milvus_storage::paimon {
namespace {

struct ParsedMetadata {
  std::string data_format;
  uint64_t record_count = 0;
  nlohmann::json deletion_file;
};

arrow::Result<ParsedMetadata> ParseMetadata(const std::string& json) {
  int64_t version;
  std::string read_path;
  int64_t record_count;
  std::string data_format;
  nlohmann::json deletion_file;
  try {
    auto value = nlohmann::json::parse(json);
    if (!value.is_object()) {
      return arrow::Status::Invalid("Paimon metadata must be a JSON object");
    }
    version = value.value("version", int64_t{0});
    read_path = value.value("read_path", std::string{});
    record_count = value.value("record_count", int64_t{0});
    data_format = value.value("data_format", std::string{});
    deletion_file = value.value("deletion_file", nlohmann::json(nullptr));
  } catch (const nlohmann::json::exception& error) {
    return arrow::Status::Invalid("Paimon metadata has invalid JSON or field types: ", error.what());
  }
  LOG_STORAGE_DEBUG_ << "Paimon metadata version=" << version;
  if (read_path != "direct-file") {
    return read_path == "data-split" ? arrow::Status::NotImplemented("Paimon data-split reads are not supported")
                                     : arrow::Status::Invalid("Paimon metadata read_path must be direct-file");
  }
  if (record_count < 0) {
    return arrow::Status::Invalid("Paimon metadata has negative record_count");
  }
  return ParsedMetadata{
      .data_format = std::move(data_format),
      .record_count = static_cast<uint64_t>(record_count),
      .deletion_file = std::move(deletion_file),
  };
}

arrow::Result<std::vector<RowGroupInfo>> MakeDirectLogicalRowGroups(const std::vector<RowGroupInfo>& physical,
                                                                    const std::vector<uint64_t>& deletions) {
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
    auto first = std::lower_bound(deletions.begin(), deletions.end(), group.start_offset);
    auto last = std::lower_bound(deletions.begin(), deletions.end(), group.end_offset);
    auto deleted = static_cast<uint64_t>(std::distance(first, last));
    auto physical_rows = static_cast<uint64_t>(group.end_offset - group.start_offset);
    if (deleted > physical_rows) {
      return arrow::Status::Invalid("Paimon deletion count exceeds physical row group size");
    }
    auto logical_rows = physical_rows - deleted;
    uint64_t logical_memory_size = 0;
    if (group.memory_size_available && physical_rows != 0) {
      logical_memory_size =
          static_cast<uint64_t>(static_cast<unsigned __int128>(group.memory_size) * logical_rows / physical_rows);
    }
    result.push_back(RowGroupInfo{.start_offset = static_cast<size_t>(logical_start),
                                  .end_offset = static_cast<size_t>(logical_start + logical_rows),
                                  .memory_size = logical_memory_size,
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
                                                               const std::vector<uint64_t>& deletions) {
  if (deletions.empty() || batch->num_rows() == 0) {
    return batch;
  }
  std::vector<int64_t> keep;
  keep.reserve(batch->num_rows());
  auto deletion = std::lower_bound(deletions.begin(), deletions.end(), physical_start);
  for (int64_t row = 0; row < batch->num_rows(); ++row) {
    auto physical = physical_start + static_cast<uint64_t>(row);
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
                       std::shared_ptr<const std::vector<uint64_t>> deletions)
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
  std::shared_ptr<const std::vector<uint64_t>> deletions_;
};

arrow::Status ValidatePredicatePushdown(const PaimonFormatReader::MetaTrait::Payload& payload,
                                        const std::string& predicate) {
  if (predicate.empty() || payload.data_format != "vortex" || !payload.sorted_deletions ||
      payload.sorted_deletions->empty()) {
    return arrow::Status::OK();
  }
  return arrow::Status::NotImplemented("Paimon Vortex predicate pushdown is not supported with deletion vectors");
}

}  // namespace

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
  ARROW_ASSIGN_OR_RAISE(auto parsed, ParseMetadata(metadata_json));
  ARROW_ASSIGN_OR_RAISE(auto fs_config, FilesystemCache::resolve_config(properties, file.path));
  ARROW_ASSIGN_OR_RAISE(auto storage_options, ToStorageOptions(fs_config));

  auto metadata = std::make_shared<Metadata>();
  metadata->cache_key = cache_key(file);
  metadata->path = file.path;
  metadata->payload.data_format = parsed.data_format;
  metadata->payload.record_count = parsed.record_count;

  size_t direct_cache_size = 0;
  if (parsed.data_format == "parquet") {
    ARROW_ASSIGN_OR_RAISE(auto parquet_metadata,
                          parquet::ParquetFormatReader::MetaTrait::load_metadata(file, properties, key_retriever));
    metadata->file_schema = parquet_metadata->file_schema;
    metadata->payload.direct_physical_row_groups = parquet_metadata->row_group_infos;
    direct_cache_size = parquet_metadata->cache_size;
    metadata->payload.direct_file_metadata = std::move(parquet_metadata);
  } else if (parsed.data_format == "vortex") {
    ARROW_ASSIGN_OR_RAISE(auto vortex_metadata,
                          vortex::VortexFormatReader::MetaTrait::load_metadata(file, properties, key_retriever));
    metadata->file_schema = vortex_metadata->file_schema;
    metadata->payload.direct_physical_row_groups = vortex_metadata->row_group_infos;
    direct_cache_size = vortex_metadata->cache_size;
    metadata->payload.direct_file_metadata = std::move(vortex_metadata);
  } else {
    return arrow::Status::NotImplemented("Paimon direct-file does not support format: ", parsed.data_format);
  }
  auto& physical_groups = metadata->payload.direct_physical_row_groups;
  uint64_t physical_rows = physical_groups.empty() ? 0 : physical_groups.back().end_offset;
  metadata->payload.physical_row_count = physical_rows;

  auto deletions = std::make_shared<std::vector<uint64_t>>();
  if (!parsed.deletion_file.is_null()) {
    if (!parsed.deletion_file.is_object()) {
      return arrow::Status::Invalid("Paimon deletion_file must be an object");
    }
    std::string path;
    int64_t offset = -1;
    int64_t length = -1;
    int64_t cardinality = -1;
    try {
      path = parsed.deletion_file.value("path", std::string{});
      offset = parsed.deletion_file.value("offset", int64_t{-1});
      length = parsed.deletion_file.value("length", int64_t{-1});
      cardinality = parsed.deletion_file.value("cardinality", int64_t{-1});
    } catch (const nlohmann::json::exception& error) {
      return arrow::Status::Invalid("Paimon deletion_file has invalid field types: ", error.what());
    }
    if (path.empty() || offset < 0 || length < 0) {
      return arrow::Status::Invalid("Paimon deletion_file has invalid path or range");
    }
    ARROW_ASSIGN_OR_RAISE(auto positions,
                          ReadDeletionVector(path, static_cast<uint64_t>(offset), static_cast<uint64_t>(length),
                                             cardinality, storage_options));
    deletions->reserve(positions.size());
    for (auto position : positions) {
      if (position >= physical_rows) {
        return arrow::Status::Invalid("Paimon deletion position exceeds physical row count");
      }
      deletions->push_back(position);
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
  metadata->cache_size = direct_cache_size + deletions->size() * sizeof(uint64_t) + metadata_json.size() +
                         physical_groups.size() * sizeof(RowGroupInfo);

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
  ARROW_RETURN_NOT_OK(ValidatePredicatePushdown(metadata->payload, predicate));
  ARROW_ASSIGN_OR_RAISE(auto output_schema, ProjectSchema(metadata->file_schema, read_schema, needed_columns));
  std::shared_ptr<FormatReader> direct_file_reader;
  if (metadata->payload.data_format == "parquet") {
    auto cached =
        std::get_if<parquet::ParquetFormatReader::MetaTrait::MetadataPtr>(&metadata->payload.direct_file_metadata);
    if (cached == nullptr) {
      return arrow::Status::Invalid("Paimon cached metadata does not match data format parquet");
    }
    std::shared_ptr<parquet::ParquetFormatReader> parquet_reader;
    ARROW_ASSIGN_OR_RAISE(parquet_reader, parquet::ParquetFormatReader::MetaTrait::create_from_metadata(
                                              *cached, file, read_schema, needed_columns, predicate));
    direct_file_reader = std::static_pointer_cast<FormatReader>(std::move(parquet_reader));
  } else if (metadata->payload.data_format == "vortex") {
    auto cached =
        std::get_if<vortex::VortexFormatReader::MetaTrait::MetadataPtr>(&metadata->payload.direct_file_metadata);
    if (cached == nullptr) {
      return arrow::Status::Invalid("Paimon cached metadata does not match data format vortex");
    }
    std::shared_ptr<vortex::VortexFormatReader> vortex_reader;
    ARROW_ASSIGN_OR_RAISE(vortex_reader, vortex::VortexFormatReader::MetaTrait::create_from_metadata(
                                             *cached, file, read_schema, needed_columns, predicate));
    direct_file_reader = std::static_pointer_cast<FormatReader>(std::move(vortex_reader));
  } else {
    return arrow::Status::NotImplemented("Paimon direct-file does not support format: ", metadata->payload.data_format);
  }
  return std::shared_ptr<PaimonFormatReader>(
      new PaimonFormatReader(std::move(metadata), file, read_schema, needed_columns, predicate,
                             std::move(direct_file_reader), std::move(output_schema)));
}

PaimonFormatReader::PaimonFormatReader(MetaTrait::MetadataPtr metadata,
                                       api::ColumnGroupFile file,
                                       std::shared_ptr<arrow::Schema> read_schema,
                                       std::vector<std::string> needed_columns,
                                       std::string predicate,
                                       std::shared_ptr<FormatReader> direct_file_reader,
                                       std::shared_ptr<arrow::Schema> output_schema)
    : metadata_(std::move(metadata)),
      file_(std::move(file)),
      read_schema_(std::move(read_schema)),
      needed_columns_(std::move(needed_columns)),
      predicate_(std::move(predicate)),
      direct_file_reader_(std::move(direct_file_reader)),
      output_schema_(std::move(output_schema)) {}

arrow::Status PaimonFormatReader::open() {
  if (!direct_file_reader_) {
    return arrow::Status::Invalid("Paimon direct-file reader is unavailable");
  }
  return arrow::Status::OK();
}

arrow::Result<std::vector<RowGroupInfo>> PaimonFormatReader::get_row_group_infos() {
  return metadata_->row_group_infos;
}

arrow::Result<std::vector<uint64_t>> PaimonFormatReader::get_rg_column_memsz(int64_t row_group_index) const {
  if (row_group_index < 0 || static_cast<size_t>(row_group_index) >= metadata_->row_group_infos.size()) {
    return arrow::Status::Invalid("Paimon row group index out of range: ", row_group_index);
  }
  const auto& logical_group = metadata_->row_group_infos[row_group_index];
  if (!logical_group.memory_size_available) {
    return arrow::Status::NotImplemented("Paimon column memory size statistics are not available");
  }
  ARROW_ASSIGN_OR_RAISE(auto physical_sizes, direct_file_reader_->get_rg_column_memsz(row_group_index));
  return DistributeMemorySizes(logical_group.memory_size, physical_sizes);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> PaimonFormatReader::filter_direct_batch(
    const std::shared_ptr<arrow::RecordBatch>& batch, uint64_t physical_start) const {
  return FilterBatch(batch, physical_start, *metadata_->payload.sorted_deletions);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> PaimonFormatReader::get_chunk(const int& row_group_index) {
  if (row_group_index < 0 || static_cast<size_t>(row_group_index) >= metadata_->row_group_infos.size()) {
    return arrow::Status::Invalid("Paimon row group index out of range: ", row_group_index);
  }
  const auto& group = metadata_->row_group_infos[row_group_index];
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

uint64_t PaimonFormatReader::logical_to_physical(uint64_t logical_offset) const {
  const auto& deletions = *metadata_->payload.sorted_deletions;
  uint64_t physical = logical_offset;
  const auto physical_rows = metadata_->payload.physical_row_count;
  for (size_t iteration = 0; iteration <= deletions.size(); ++iteration) {
    auto deleted =
        static_cast<uint64_t>(std::upper_bound(deletions.begin(), deletions.end(), physical) - deletions.begin());
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
  std::vector<int64_t> physical;
  physical.reserve(indices.size());
  for (auto index : indices) {
    if (index < 0 || static_cast<uint64_t>(index) >= metadata_->payload.record_count) {
      return arrow::Status::Invalid("Paimon direct-file take index is out of range");
    }
    auto physical_index = logical_to_physical(static_cast<uint64_t>(index));
    if (physical_index >= metadata_->payload.physical_row_count ||
        physical_index > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return arrow::Status::Invalid("Paimon direct-file physical index is out of range");
    }
    physical.push_back(static_cast<int64_t>(physical_index));
  }
  return direct_file_reader_->take(physical);
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
  auto physical_start = logical_to_physical(start);
  auto physical_end = logical_to_physical(end - 1) + 1;
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

arrow::Status PaimonFormatReader::set_predicate(const std::string& predicate) {
  if (!direct_file_reader_) {
    return arrow::Status::Invalid("Paimon direct-file reader is unavailable");
  }
  ARROW_RETURN_NOT_OK(ValidatePredicatePushdown(metadata_->payload, predicate));
  ARROW_RETURN_NOT_OK(direct_file_reader_->set_predicate(predicate));
  predicate_ = predicate;
  return arrow::Status::OK();
}

}  // namespace milvus_storage::paimon
