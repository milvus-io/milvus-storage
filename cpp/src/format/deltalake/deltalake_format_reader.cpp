// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#include "milvus-storage/format/deltalake/deltalake_format_reader.h"

#include <algorithm>
#include <atomic>
#include <future>
#include <limits>
#include <mutex>
#include <span>
#include <unordered_set>
#include <unordered_map>
#include <arrow/compute/api.h>
#include <arrow/builder.h>
#include <arrow/table.h>
#include <folly/Indestructible.h>
#include <nlohmann/json.hpp>
#include <parquet/metadata.h>

#include "format/deltalake/deltalake_common.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"

namespace milvus_storage::deltalake {
namespace {

struct Descriptor {
  std::string table;
  int64_t version;
};

std::string DescriptorCacheKey(const Descriptor& descriptor) {
  return nlohmann::json::array({descriptor.table, descriptor.version}).dump();
}

arrow::Result<Descriptor> ParseDescriptor(const api::ColumnGroupFile& file) {
  try {
    auto value = nlohmann::json::parse(file.Get<std::string>(api::kPropertyMetadata));
    if (!value.is_object() || !value.contains("version") || !value.at("version").is_number_integer() ||
        value.at("version") != 1 || !value.contains("table_uri") || !value.at("table_uri").is_string() ||
        !value.contains("snapshot_version") || !value.at("snapshot_version").is_number_integer()) {
      return arrow::Status::Invalid("Invalid Delta snapshot descriptor");
    }
    const auto& version = value.at("snapshot_version");
    if ((version.is_number_unsigned() && version.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max())) ||
        (!version.is_number_unsigned() && version.get<int64_t>() < 0)) {
      return arrow::Status::Invalid("Delta descriptor requires a nonnegative signed 64-bit version");
    }
    auto table = value.at("table_uri").get<std::string>();
    if (table.empty() || table.back() != '/' || file.start_index != 0 || file.end_index < 0) {
      return arrow::Status::Invalid("Invalid Delta table location or live row range");
    }
    const auto size = file.properties.find(api::kPropertyFileSize);
    if (size == file.properties.end() || !api::convert::convertFunc<uint64_t>(size->second).first) {
      return arrow::Status::Invalid("Delta descriptor requires a valid file size");
    }
    return Descriptor{std::move(table), version.get<int64_t>()};
  } catch (const std::exception& error) {
    return arrow::Status::Invalid("Cannot parse Delta snapshot descriptor: ", error.what());
  }
}

// Failures are never cached. A pending result retains its value only until
// callers leave the flight; the process registry itself owns weak references.
template <typename T, bool Weak = false>
class LoadCache {
  public:
  using Ptr = std::shared_ptr<T>;
  using Result = arrow::Result<Ptr>;
  template <typename Loader>
  Result Get(const std::string& key, Loader&& load) {
    std::shared_future<Result> pending;
    std::shared_ptr<std::promise<Result>> leader;
    {
      std::lock_guard lock(mutex_);
      if constexpr (Weak) {
        auto cached = weak_.get(key);
        if (cached) {
          if (auto value = cached->lock())
            return value;
          weak_.remove(key);
        }
      } else {
        auto found = strong_.find(key);
        if (found != strong_.end())
          return found->second;
      }
      auto flight = flights_.find(key);
      if (flight != flights_.end())
        pending = flight->second;
      else {
        leader = std::make_shared<std::promise<Result>>();
        pending = leader->get_future().share();
        flights_.emplace(key, pending);
      }
    }
    if (!leader)
      return pending.get();
    Result result = [&]() -> Result {
      try {
        return load();
      } catch (const std::bad_alloc& error) {
        return arrow::Status::OutOfMemory(error.what());
      } catch (const std::exception& error) {
        return arrow::Status::IOError("Delta metadata load: ", error.what());
      } catch (...) {
        return arrow::Status::UnknownError("Unknown Delta metadata load failure");
      }
    }();
    {
      std::lock_guard lock(mutex_);
      if (result.ok()) {
        try {
          if constexpr (Weak)
            weak_.put(key, std::weak_ptr<T>(*result));
          else
            strong_.emplace(key, *result);
        } catch (...) { /* Cache publication is best effort; always release waiters. */
        }
      }
      flights_.erase(key);
    }
    leader->set_value(result);
    return result;
  }

  private:
  std::mutex mutex_;
  LRUCache<std::string, std::weak_ptr<T>> weak_{512};
  std::unordered_map<std::string, Ptr> strong_;
  std::unordered_map<std::string, std::shared_future<Result>> flights_;
};

arrow::Result<std::vector<std::string>> RequestedColumns(const std::shared_ptr<arrow::Schema>& snapshot,
                                                         const std::shared_ptr<arrow::Schema>& requested,
                                                         const std::vector<std::string>& needed) {
  auto names = requested ? requested->field_names() : needed.empty() ? snapshot->field_names() : needed;
  std::unordered_set<std::string> seen;
  for (const auto& name : names) {
    auto field = snapshot->GetFieldByName(name);
    if (!field || !seen.insert(name).second)
      return arrow::Status::Invalid("Unknown or duplicate Delta field: ", name);
    if (requested) {
      const auto caller = requested->GetFieldByName(name);
      if (!caller || !field->type()->Equals(caller->type()))
        return arrow::Status::Invalid("Caller type conflicts with pinned Delta schema: ", name);
    }
  }
  return names;
}

}  // namespace

uint64_t LogicalToPhysical(uint64_t logical, std::span<const uint64_t> deleted) {
  uint64_t low = logical, high = logical + deleted.size();
  while (low < high) {
    const auto middle = low + (high - low) / 2;
    const auto removed = std::upper_bound(deleted.begin(), deleted.end(), middle) - deleted.begin();
    if (middle + 1 - removed > logical)
      high = middle;
    else
      low = middle + 1;
  }
  return low;
}

struct DeltaFileMetadata {
  parquet::ParquetFormatReader::MetaTrait::MetadataPtr parquet;
  api::ColumnGroupFile physical_file;
  std::shared_ptr<BlockingDeltaFile> kernel;
  std::vector<RowGroupInfo> groups;
  std::shared_ptr<std::atomic<uint64_t>> memory_counter;
  uint64_t charge = 0;
  ~DeltaFileMetadata() {
    if (memory_counter)
      memory_counter->fetch_sub(charge, std::memory_order_relaxed);
  }
};

class DeltaFileCache : public LoadCache<DeltaFileMetadata> {};

class DeltaTableContext {
  public:
  ResolvedLocation location;
  int64_t version;
  std::shared_ptr<BlockingDeltaSnapshot> snapshot;
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<DeltaFileCache> files = std::make_shared<DeltaFileCache>();
  std::shared_ptr<std::atomic<uint64_t>> file_bytes = std::make_shared<std::atomic<uint64_t>>(0);
};

static arrow::Result<std::shared_ptr<DeltaTableContext>> GetTable(
    const Descriptor& descriptor, const std::shared_ptr<const api::Properties>& properties) {
  ARROW_ASSIGN_OR_RAISE(auto location, ResolveLocation(descriptor.table, properties));
  const auto key = nlohmann::json::array({location.url, descriptor.version, location.filesystem_key}).dump();
  static folly::Indestructible<LoadCache<DeltaTableContext, true>> registry;
  return registry->Get(key, [&]() -> arrow::Result<std::shared_ptr<DeltaTableContext>> {
    auto table = std::make_shared<DeltaTableContext>();
    table->location = location;
    table->version = descriptor.version;
    ARROW_ASSIGN_OR_RAISE(table->snapshot, BlockingDeltaSnapshot::LoadForReader(location.url, descriptor.version,
                                                                                location.filesystem, location.binding));
    ARROW_ASSIGN_OR_RAISE(table->schema, table->snapshot->Schema());
    return table;
  });
}

static arrow::Result<std::shared_ptr<DeltaFileMetadata>> GetFile(
    const DeltaLakeFormatReader::MetaTrait::Payload& payload, const api::ColumnGroupFile& file) {
  ARROW_ASSIGN_OR_RAISE(auto resolved, ResolveDataFile(file.path, payload.table->location, payload.properties));
  auto result = payload.files->Get(resolved.url, [&]() -> arrow::Result<std::shared_ptr<DeltaFileMetadata>> {
    auto metadata = std::make_shared<DeltaFileMetadata>();
    metadata->physical_file = file;
    metadata->physical_file.path = resolved.path;
    ARROW_ASSIGN_OR_RAISE(metadata->parquet, parquet::ParquetFormatReader::MetaTrait::load_metadata_with_filesystem(
                                                 file, payload.table->location.filesystem, resolved.path,
                                                 payload.properties, payload.key_retriever));
    const auto footer = metadata->parquet->payload.parquet_metadata;
    if (footer->num_rows() < 0)
      return arrow::Status::Invalid("Negative Delta Parquet row count");
    // Kernel plans against the schema, not row-group payload metadata. Transfer
    // only the schema/key-values and the C++-decoded count. Serializing an actual
    // plaintext encrypted footer would require Arrow 17's signing encryptor;
    // separately encrypted column metadata must also stay with the C++ reader.
    auto schema_metadata =
        ::parquet::FileMetaDataBuilder::Make(footer->schema(), ::parquet::default_writer_properties());
    const auto serialized = schema_metadata->Finish(footer->key_value_metadata())->SerializeToString();
    ARROW_ASSIGN_OR_RAISE(metadata->kernel, payload.table->snapshot->LoadFile(
                                                resolved.url, serialized, footer->num_rows(), file.end_index,
                                                file.Get<uint64_t>(api::kPropertyFileSize)));
    // A changed schema can introduce data-dependent allocation (nested casts,
    // constants, missing arrays). Preserve the existing unavailable-estimate API.
    // Partition fields may also be materialized in Parquet. Kernel still
    // reconstructs them from Add metadata, so schema equality alone is not enough.
    const bool memory_available = !payload.table->snapshot->HasPartitionColumns() &&
                                  metadata->parquet->file_schema->Equals(*payload.table->schema, false);
    uint64_t offset = 0;
    const auto deleted = metadata->kernel->Deletions();
    uint64_t physical_offset = 0;
    if (metadata->parquet->row_group_infos.size() != static_cast<size_t>(footer->num_row_groups()))
      return arrow::Status::Invalid("Delta row group metadata count differs from footer");
    for (size_t group_index = 0; group_index < metadata->parquet->row_group_infos.size(); ++group_index) {
      const auto& physical = metadata->parquet->row_group_infos[group_index];
      const auto footer_rows = footer->RowGroup(group_index)->num_rows();
      if (footer_rows < 0 || physical.start_offset != physical_offset || physical.end_offset < physical.start_offset ||
          physical.end_offset - physical.start_offset != static_cast<uint64_t>(footer_rows))
        return arrow::Status::Invalid("Delta physical row group offsets differ from footer");
      physical_offset = physical.end_offset;
      const auto first = std::lower_bound(deleted.begin(), deleted.end(), physical.start_offset);
      const auto last = std::lower_bound(first, deleted.end(), physical.end_offset);
      const uint64_t rows = physical.end_offset - physical.start_offset;
      const auto live = rows - static_cast<uint64_t>(last - first);
      const bool available = memory_available && physical.memory_size_available;
      const auto size = available && rows
                            ? static_cast<uint64_t>(static_cast<unsigned __int128>(physical.memory_size) * live / rows)
                            : 0;
      metadata->groups.push_back(RowGroupInfo{offset, offset + live, size, available});
      offset += live;
    }
    if (physical_offset != metadata->kernel->PhysicalRows() || offset != metadata->kernel->LiveRows())
      return arrow::Status::Invalid("Delta row-group counts disagree with footer");
    metadata->charge = metadata->parquet->cache_size + serialized.size() + deleted.size_bytes() +
                       metadata->groups.size() * sizeof(RowGroupInfo);
    metadata->memory_counter = payload.table->file_bytes;
    metadata->memory_counter->fetch_add(metadata->charge, std::memory_order_relaxed);
    return metadata;
  });
  ARROW_ASSIGN_OR_RAISE(auto metadata, std::move(result));
  if (metadata->kernel->LiveRows() != static_cast<uint64_t>(file.end_index) ||
      file.Get<uint64_t>(api::kPropertyFileSize) != metadata->physical_file.Get<uint64_t>(api::kPropertyFileSize)) {
    return arrow::Status::Invalid("Delta file descriptor differs from cached metadata");
  }
  return metadata;
}

std::string DeltaLakeFormatReader::MetaTrait::cache_key(const api::ColumnGroupFile& file) {
  auto parsed = ParseDescriptor(file);
  if (!parsed.ok())
    return "invalid-delta:" + file.path + file.Get<std::string>(api::kPropertyMetadata);
  return DescriptorCacheKey(*parsed);
}

arrow::Result<DeltaLakeFormatReader::MetaTrait::MetadataPtr> DeltaLakeFormatReader::MetaTrait::load_metadata(
    const api::ColumnGroupFile& file,
    const std::shared_ptr<const api::Properties>& properties,
    const KeyRetriever& key_retriever) {
  ARROW_ASSIGN_OR_RAISE(auto descriptor, ParseDescriptor(file));
  ARROW_ASSIGN_OR_RAISE(auto table, GetTable(descriptor, properties));
  auto metadata = std::make_shared<Metadata>();
  metadata->cache_key = DescriptorCacheKey(descriptor);
  metadata->path = descriptor.table;
  metadata->file_schema = table->schema;
  // Decryption callbacks remain scoped to the top-level Reader. Never share
  // decrypted per-file metadata across unrelated callback identities.
  metadata->payload =
      Payload{table, key_retriever ? std::make_shared<DeltaFileCache>() : table->files, properties, key_retriever};
  ARROW_ASSIGN_OR_RAISE(auto first_file, GetFile(metadata->payload, file));
  metadata->cache_size = sizeof(Metadata) + table->snapshot->MemorySize() + table->file_bytes->load();
  return MetadataPtr(std::move(metadata));
}

arrow::Result<std::shared_ptr<DeltaLakeFormatReader>> DeltaLakeFormatReader::MetaTrait::create_from_metadata(
    MetadataPtr metadata,
    const api::ColumnGroupFile& file,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& needed_columns,
    const std::string&) {
  ARROW_ASSIGN_OR_RAISE(auto descriptor, ParseDescriptor(file));
  if (!metadata || !metadata->payload.table || metadata->path != descriptor.table ||
      metadata->payload.table->version != descriptor.version) {
    return arrow::Status::Invalid("Delta metadata does not match descriptor");
  }
  ARROW_ASSIGN_OR_RAISE(auto names, RequestedColumns(metadata->file_schema, read_schema, needed_columns));
  ARROW_ASSIGN_OR_RAISE(auto source, GetFile(metadata->payload, file));
  ARROW_ASSIGN_OR_RAISE(auto plan, source->kernel->Plan(*metadata->payload.table->snapshot, names));
  auto reader = std::shared_ptr<DeltaLakeFormatReader>(new DeltaLakeFormatReader(metadata, source, std::move(plan)));
  ARROW_RETURN_NOT_OK(reader->open());
  return reader;
}

DeltaLakeFormatReader::DeltaLakeFormatReader(MetaTrait::MetadataPtr metadata,
                                             std::shared_ptr<DeltaFileMetadata> file,
                                             std::shared_ptr<BlockingDeltaReadPlan> plan)
    : metadata_(std::move(metadata)), file_(std::move(file)), plan_(std::move(plan)) {}

arrow::Status DeltaLakeFormatReader::open() {
  if (!output_schema_) {
    ARROW_ASSIGN_OR_RAISE(auto schema, plan_->Schema());
    std::shared_ptr<parquet::ParquetFormatReader> reader;
    const auto leaves = plan_->Leaves();
    if (!leaves.empty()) {
      // Footer metadata is shared; each Reader keeps its own I/O options and
      // decryption callback when creating a new stateful Parquet reader.
      auto source = std::make_shared<parquet::ParquetFormatReader::MetaTrait::Metadata>(*file_->parquet);
      source->payload.properties = metadata_->payload.properties;
      source->payload.key_retriever = metadata_->payload.key_retriever;
      ARROW_ASSIGN_OR_RAISE(reader, parquet::ParquetFormatReader::MetaTrait::create_with_leaf_projection(
                                        source, file_->physical_file, leaves));
    }
    parquet_ = std::move(reader);
    output_schema_ = std::move(schema);
  }
  return arrow::Status::OK();
}

std::shared_ptr<arrow::Schema> DeltaLakeFormatReader::get_schema() const { return metadata_->file_schema; }
arrow::Result<std::vector<RowGroupInfo>> DeltaLakeFormatReader::get_row_group_infos() {
  if (!parquet_) {
    auto groups = file_->groups;
    for (auto& group : groups) group.read_size = 0;
    return groups;
  }
  ARROW_ASSIGN_OR_RAISE(auto groups, parquet_->get_row_group_infos());
  if (groups.size() != file_->groups.size())
    return arrow::Status::Invalid("Delta projected row group count differs from file metadata");
  for (size_t index = 0; index < groups.size(); ++index) {
    const auto read_size = groups[index].read_size;
    groups[index] = file_->groups[index];
    // Surviving rows still require the group's full projected physical bytes.
    // Output-memory estimates and live offsets belong to the logical group.
    groups[index].read_size = groups[index].start_offset == groups[index].end_offset ? 0 : read_size;
  }
  return groups;
}
arrow::Result<std::vector<uint64_t>> DeltaLakeFormatReader::get_rg_column_memsz(int64_t index) const {
  if (index < 0 || static_cast<size_t>(index) >= file_->groups.size())
    return arrow::Status::Invalid("Delta row group out of bounds");
  if (!file_->groups[index].memory_size_available || !parquet_)
    return arrow::Status::NotImplemented("Delta transformed column memory sizes unavailable");
  ARROW_ASSIGN_OR_RAISE(auto sizes, parquet_->get_rg_column_memsz(index));
  return DistributeMemorySizes(file_->groups[index].memory_size, sizes);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> DeltaLakeFormatReader::transform(
    const std::shared_ptr<arrow::RecordBatch>& batch, uint64_t physical_start) const {
  ARROW_ASSIGN_OR_RAISE(auto logical, plan_->Apply(batch));
  auto deleted = file_->kernel->Deletions();
  auto position = std::lower_bound(deleted.begin(), deleted.end(), physical_start);
  if (position == deleted.end() || *position >= physical_start + batch->num_rows())
    return logical;
  arrow::Int64Builder indices;
  for (int64_t row = 0; row < batch->num_rows(); ++row) {
    if (position != deleted.end() && *position == physical_start + row)
      ++position;
    else
      ARROW_RETURN_NOT_OK(indices.Append(row));
  }
  ARROW_ASSIGN_OR_RAISE(auto keep, indices.Finish());
  if (logical->num_columns() == 0)
    return arrow::RecordBatch::Make(output_schema_, keep->length(), std::vector<std::shared_ptr<arrow::Array>>{});
  ARROW_ASSIGN_OR_RAISE(auto selected, arrow::compute::Take(logical, keep));
  return selected.record_batch();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> DeltaLakeFormatReader::materialize_rows(int64_t rows) const {
  // Without physical leaves, every output field is constant/null within a file.
  // File and DV validation already ran, so materialize only rows being returned.
  return plan_->Apply(arrow::RecordBatch::Make(arrow::schema({}), rows, std::vector<std::shared_ptr<arrow::Array>>{}));
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> DeltaLakeFormatReader::get_chunk(const int& index) {
  if (index < 0 || static_cast<size_t>(index) >= file_->groups.size())
    return arrow::Status::Invalid("Delta row group out of bounds");
  const auto& group = file_->groups[index];
  if (group.start_offset == group.end_offset)
    return arrow::RecordBatch::MakeEmpty(output_schema_);
  if (!parquet_)
    return materialize_rows(group.end_offset - group.start_offset);
  ARROW_ASSIGN_OR_RAISE(auto batch, parquet_->get_chunk(index));
  const auto& physical = file_->parquet->row_group_infos[index];
  return transform(batch, physical.start_offset);
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> DeltaLakeFormatReader::get_chunks(
    const std::vector<int>& indices) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  for (int index : indices) {
    ARROW_ASSIGN_OR_RAISE(auto batch, get_chunk(index));
    batches.push_back(std::move(batch));
  }
  return batches;
}

arrow::Result<std::shared_ptr<arrow::Table>> DeltaLakeFormatReader::take(const std::vector<int64_t>& indices) {
  if (indices.empty())
    return arrow::Table::MakeEmpty(output_schema_);
  std::vector<int64_t> physical;
  if (parquet_)
    physical.reserve(indices.size());
  int64_t previous = -1;
  for (auto index : indices) {
    if (index <= previous || index < 0 || static_cast<uint64_t>(index) >= file_->kernel->LiveRows())
      return arrow::Status::Invalid("Delta take indices must be sorted, unique, and in bounds");
    previous = index;
    if (parquet_)
      physical.push_back(LogicalToPhysical(index, file_->kernel->Deletions()));
  }
  std::vector<std::shared_ptr<arrow::RecordBatch>> output;
  if (!parquet_) {
    ARROW_ASSIGN_OR_RAISE(auto batch, materialize_rows(indices.size()));
    output.push_back(std::move(batch));
  } else {
    ARROW_ASSIGN_OR_RAISE(auto table, parquet_->take(physical));
    arrow::TableBatchReader batches(*table);
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ARROW_RETURN_NOT_OK(batches.ReadNext(&batch));
      if (!batch)
        break;
      ARROW_ASSIGN_OR_RAISE(auto transformed, plan_->Apply(batch));
      output.push_back(std::move(transformed));
    }
  }
  return arrow::Table::FromRecordBatches(output_schema_, output);
}

class DeltaLakeFormatReader::RangeReader final : public arrow::RecordBatchReader {
  public:
  RangeReader(std::shared_ptr<DeltaLakeFormatReader> reader,
              std::shared_ptr<arrow::Schema> schema,
              std::span<const RowGroupInfo> groups,
              uint64_t start,
              uint64_t end)
      : reader_(std::move(reader)), schema_(std::move(schema)), groups_(groups), start_(start), end_(end) {
    // Empty/deleted groups may repeat offsets. Start at the first live end
    // beyond the requested offset rather than scanning every preceding group.
    const auto first =
        std::upper_bound(groups_.begin(), groups_.end(), start_,
                         [](uint64_t offset, const RowGroupInfo& group) { return offset < group.end_offset; });
    index_ = first - groups_.begin();
  }
  std::shared_ptr<arrow::Schema> schema() const override { return schema_; }
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* output) override {
    *output = nullptr;
    while (index_ < groups_.size() && start_ < end_) {
      const auto index = index_;
      const auto& group = groups_[index];
      if (group.end_offset <= start_ || group.start_offset >= end_) {
        ++index_;
        continue;
      }
      const auto start = std::max(start_, uint64_t(group.start_offset));
      const auto end = std::min(end_, uint64_t(group.end_offset));
      if (reader_->parquet_) {
        ARROW_ASSIGN_OR_RAISE(auto batch, reader_->get_chunk(static_cast<int>(index)));
        *output = batch->Slice(start - group.start_offset, end - start);
      } else {
        ARROW_ASSIGN_OR_RAISE(*output, reader_->materialize_rows(end - start));
      }
      ++index_;
      start_ = end;
      return arrow::Status::OK();
    }
    return arrow::Status::OK();
  }

  private:
  std::shared_ptr<DeltaLakeFormatReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  // reader_ retains the immutable file metadata underlying this view.
  std::span<const RowGroupInfo> groups_;
  uint64_t start_, end_;
  size_t index_ = 0;
};

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> DeltaLakeFormatReader::read_with_range(const uint64_t& start,
                                                                                                const uint64_t& end) {
  if (start > end || end > file_->kernel->LiveRows())
    return arrow::Status::Invalid("Delta range out of bounds");
  return std::make_shared<RangeReader>(shared_from_this(), output_schema_, file_->groups, start, end);
}

arrow::Result<std::shared_ptr<FormatReader>> DeltaLakeFormatReader::clone_reader() {
  auto reader = std::shared_ptr<DeltaLakeFormatReader>(new DeltaLakeFormatReader(metadata_, file_, plan_));
  ARROW_RETURN_NOT_OK(reader->open());
  return reader;
}

}  // namespace milvus_storage::deltalake
