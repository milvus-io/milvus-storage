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

#include "milvus-storage/format/lance/lance_table_reader.h"

#include <algorithm>
#include <compare>
#include <condition_variable>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include <arrow/chunked_array.h>  // keep this line before other arrow header
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/status.h>
#include <arrow/result.h>
#include <fmt/format.h>

#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "bridge_util.h"

namespace milvus_storage::lance {

// Lance metadata follows the Dataset/fragment hierarchy and uses three cache
// levels with distinct identities and ownership:
//
//   LanceDatasetCache (process-wide)
//     key: {dataset version, base URI, filesystem cache key}
//     value: weak_ptr<BlockingDataset>
//
//   FormatReaderMetadataCache (owned by one top-level Reader)
//     key: base URI
//     value: Metadata -> Payload -> shared_ptr<BlockingDataset>
//
//   Payload::FragmentMetadataCache
//     key: fragment ID
//     value: immutable fragment schema, row groups, deletion state, and
//            memory estimates
//
// LanceFormat::explore() records the already-open Dataset's version in each
// ColumnGroupFile. A reader can therefore query LanceDatasetCache before the
// expensive Dataset open. Legacy files without that property resolve only the
// latest manifest location first. Exact-version singleflight ensures concurrent
// misses decode and retain one Dataset snapshot. The process cache owns only
// weak pointers, so top-level reader metadata determines Dataset lifetime.
//
//   LanceDatasetCache
//     `-- weak_ptr<BlockingDataset> -------------------------+
//                                                            |
//   ReaderImpl                                               |
//     `-- MetadataCache                                      |
//           `-- FormatReaderMetadataCache<LanceTableReader>  |
//                 `-- Metadata -> Payload                    |
//                       +-- shared_ptr<BlockingDataset> ------+
//                       `-- FragmentMetadataCache
//                             +-- fragment 0 -> metadata[0]
//                             `-- fragment 1 -> metadata[1]
//
// BlockingFragmentReader is projection-specific and stateful, so every
// LanceTableReader creates its own instance rather than caching it.
class LanceDatasetCache final {
  public:
  using DatasetPtr = std::shared_ptr<BlockingDataset>;

  struct Key {
    uint64_t version;
    std::string base_uri;
    std::string filesystem_cache_key;

    bool operator==(const Key&) const = default;
    auto operator<=>(const Key&) const = default;
  };

  static LanceDatasetCache& Instance() {
    static LanceDatasetCache cache;
    return cache;
  }

  template <typename DatasetLoader>
  arrow::Result<DatasetPtr> GetOrOpen(const Key& key, DatasetLoader&& load_fn) {
    std::shared_ptr<InFlightOpen> in_flight_open;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      auto cached = datasets_.get(key);
      if (cached.has_value()) {
        if (auto dataset = cached->lock()) {
          return dataset;
        }
        datasets_.remove(key);
      }

      const auto existing_open = in_flight_opens_.find(key);
      if (existing_open != in_flight_opens_.end()) {
        in_flight_open = existing_open->second;
        in_flight_open->cv.wait(lock, [&in_flight_open]() { return in_flight_open->done; });
        if (!in_flight_open->status.ok()) {
          return in_flight_open->status;
        }
        return in_flight_open->dataset;
      }

      in_flight_open = std::make_shared<InFlightOpen>();
      in_flight_opens_.emplace(key, in_flight_open);
    }

    auto status = arrow::Status::OK();
    DatasetPtr dataset;
    try {
      auto load_result = load_fn();
      status = load_result.status();
      if (load_result.ok()) {
        dataset = std::move(load_result).ValueOrDie();
        if (!dataset) {
          status = arrow::Status::Invalid("Lance dataset loader returned null for base URI: ", key.base_uri,
                                          ", version: ", key.version);
        }
      }
    } catch (const std::exception& e) {
      status = arrow::Status::UnknownError("Exception while opening Lance dataset for base URI ", key.base_uri,
                                           ", version ", key.version, ": ", e.what());
    } catch (...) {
      status = arrow::Status::UnknownError("Unknown exception while opening Lance dataset for base URI: ", key.base_uri,
                                           ", version: ", key.version);
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (status.ok()) {
        try {
          datasets_.put(key, std::weak_ptr<BlockingDataset>(dataset));
        } catch (...) {
          // Publication is best effort. The opened dataset must still reach all
          // waiters so an allocation failure cannot strand the in-flight open.
        }
      }
      in_flight_open->status = status;
      in_flight_open->dataset = dataset;
      in_flight_open->done = true;

      auto in_flight_it = in_flight_opens_.find(key);
      if (in_flight_it != in_flight_opens_.end() && in_flight_it->second == in_flight_open) {
        in_flight_opens_.erase(in_flight_it);
      }
    }
    in_flight_open->cv.notify_all();

    if (!status.ok()) {
      return status;
    }
    return dataset;
  }

  private:
  struct InFlightOpen {
    bool done = false;
    arrow::Status status = arrow::Status::OK();
    DatasetPtr dataset;
    std::condition_variable cv;
  };

  // Weak entries are cheap, while this larger bound absorbs normal snapshot
  // churn without allowing versioned keys to grow for the process lifetime.
  static constexpr size_t kDatasetCacheCapacity = 4096;

  std::mutex mutex_;
  LRUCache<Key, std::weak_ptr<BlockingDataset>> datasets_{kDatasetCacheCapacity};
  std::map<Key, std::shared_ptr<InFlightOpen>> in_flight_opens_;
};

// Opening a fragment creates a shallow Rust Dataset clone:
//
//   C++ BlockingDataset::inner (Rust Dataset)
//     `-- manifest: Arc ----------------------------+
//   FileFragment[0] -> Arc<Dataset clone>           |
//     `-- manifest: Arc ----------------------------+--> one Manifest allocation
//   FileFragment[1] -> Arc<Dataset clone>           |
//     `-- manifest: Arc ----------------------------+
//
// The cloned Dataset structs keep fragment readers alive independently, while
// their Arc-backed ObjectStore, Session, caches, and Manifest remain shared.
struct LanceTableReader::MetaTrait::FragmentMetadata {
  std::shared_ptr<arrow::Schema> file_schema;
  std::vector<RowGroupInfo> row_group_infos;
  uint64_t num_deletions = 0;
  uint64_t logical_chunk_rows = 0;
  std::shared_ptr<const std::vector<uint64_t>> column_memory_weights;
};

class LanceTableReader::MetaTrait::FragmentMetadataCache final {
  public:
  using FragmentMetadataPtr = std::shared_ptr<const FragmentMetadata>;

  template <typename FragmentMetadataLoader>
  arrow::Result<FragmentMetadataPtr> get_or_load(uint64_t fragment_id, FragmentMetadataLoader&& load_fn) {
    std::shared_ptr<InFlightLoad> in_flight_load;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      auto cached = fragments_.find(fragment_id);
      if (cached != fragments_.end()) {
        return cached->second;
      }

      auto [it, inserted] = in_flight_loads_.try_emplace(fragment_id, std::make_shared<InFlightLoad>());
      in_flight_load = it->second;
      if (!inserted) {
        in_flight_load->cv.wait(lock, [&in_flight_load]() { return in_flight_load->done; });
        if (!in_flight_load->status.ok()) {
          return in_flight_load->status;
        }
        return in_flight_load->metadata;
      }
    }

    auto status = arrow::Status::OK();
    FragmentMetadataPtr metadata;
    try {
      auto load_result = load_fn();
      status = load_result.status();
      if (load_result.ok()) {
        metadata = std::move(load_result).ValueOrDie();
        if (!metadata) {
          status =
              arrow::Status::Invalid("Lance fragment metadata loader returned null for fragment ID: ", fragment_id);
        }
      }
    } catch (const std::exception& e) {
      status = arrow::Status::UnknownError("Exception while loading Lance fragment metadata for fragment ID ",
                                           fragment_id, ": ", e.what());
    } catch (...) {
      status = arrow::Status::UnknownError("Unknown exception while loading Lance fragment metadata for fragment ID: ",
                                           fragment_id);
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (status.ok()) {
        try {
          auto [it, inserted] = fragments_.try_emplace(fragment_id, metadata);
          if (!inserted) {
            metadata = it->second;
          }
        } catch (...) {
          // Publication is best effort. The loaded metadata must still be
          // delivered to every waiter so the in-flight operation can finish.
        }
      }
      in_flight_load->status = status;
      in_flight_load->metadata = metadata;
      in_flight_load->done = true;

      auto in_flight_it = in_flight_loads_.find(fragment_id);
      if (in_flight_it != in_flight_loads_.end() && in_flight_it->second == in_flight_load) {
        in_flight_loads_.erase(in_flight_it);
      }
    }
    in_flight_load->cv.notify_all();

    if (!status.ok()) {
      return status;
    }
    return metadata;
  }

  private:
  struct InFlightLoad {
    bool done = false;
    arrow::Status status = arrow::Status::OK();
    FragmentMetadataPtr metadata;
    std::condition_variable cv;
  };

  std::mutex mutex_;
  std::unordered_map<uint64_t, FragmentMetadataPtr> fragments_;
  std::unordered_map<uint64_t, std::shared_ptr<InFlightLoad>> in_flight_loads_;
};

LanceTableReader::LanceTableReader(MetaTrait::MetadataPtr metadata,
                                   uint64_t fragment_id,
                                   const std::shared_ptr<arrow::Schema>& schema,
                                   const std::vector<std::string>& needed_columns)
    : filesystem_(metadata->payload.filesystem),
      dataset_(metadata->payload.dataset),
      uri_(metadata->payload.base_uri),
      fragment_id_(fragment_id),
      read_schema_(schema),
      properties_(metadata->payload.properties),
      needed_columns_(needed_columns),
      fragment_reader_(nullptr) {}

LanceTableReader::LanceTableReader(const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
                                   const std::string& uri,
                                   uint64_t fragment_id,
                                   const std::shared_ptr<arrow::Schema>& schema,
                                   const milvus_storage::api::Properties& properties,
                                   const std::vector<std::string>& needed_columns,
                                   uint64_t dataset_version)
    : filesystem_(filesystem),
      uri_(uri),
      fragment_id_(fragment_id),
      dataset_version_(dataset_version),
      read_schema_(schema),
      properties_(properties),
      needed_columns_(needed_columns),
      fragment_reader_(nullptr) {}

static arrow::Result<std::vector<uint64_t>> estimate_fragment_column_memory_sizes(const BlockingDataset& dataset,
                                                                                  uint64_t fragment_id,
                                                                                  size_t num_columns) {
  FIU_RETURN_ON(FIUKEY_MEMORY_SIZE_ESTIMATION_FAIL,
                arrow::Status::NotImplemented("Injected fault: ", FIUKEY_MEMORY_SIZE_ESTIMATION_FAIL));

  // Lance 7 returns both this estimate and FileFragment::schema() in current
  // dataset-schema order. Fields not physically present in the fragment have a
  // zero estimate, so schema evolution does not require positional remapping.
  ARROW_ASSIGN_OR_RAISE(auto memory_sizes, dataset.EstimateFragmentColumnMemory(fragment_id));
  if (memory_sizes.size() != num_columns) {
    return arrow::Status::Invalid("Lance column memory estimate count does not match the file schema: ",
                                  memory_sizes.size(), " != ", num_columns);
  }

  uint64_t total_size = 0;
  for (auto memory_size : memory_sizes) {
    if (memory_size > std::numeric_limits<uint64_t>::max() - total_size) {
      return arrow::Status::Invalid("Lance column memory estimates exceed the uint64_t range");
    }
    total_size += memory_size;
  }
  return memory_sizes;
}

static arrow::Result<std::vector<RowGroupInfo>> create_row_group_infos(
    uint64_t rows_in_file,
    uint64_t logical_chunk_rows,
    const std::vector<uint64_t>& fragment_column_memory_sizes,
    bool memory_size_available) {
  if (rows_in_file == 0) {
    return std::vector<RowGroupInfo>{};
  }
  assert(logical_chunk_rows > 0);

  uint64_t fragment_memory_size = 0;
  if (memory_size_available) {
    for (auto column_memory_size : fragment_column_memory_sizes) {
      fragment_memory_size += column_memory_size;
    }
  }

  std::vector<RowGroupInfo> result;
  uint64_t last_offset = 0;
  uint64_t last_memory_offset = 0;

  while (last_offset < rows_in_file) {
    uint64_t end_offset = std::min(last_offset + logical_chunk_rows, rows_in_file);
    // end_offset <= rows_in_file, so the quotient is at most fragment_memory_size and is safe to cast to uint64_t.
    auto memory_offset =
        static_cast<uint64_t>((static_cast<unsigned __int128>(fragment_memory_size) * end_offset) / rows_in_file);
    auto memory_size = memory_offset - last_memory_offset;
    result.emplace_back(RowGroupInfo{
        .start_offset = last_offset,
        .end_offset = end_offset,
        .memory_size = memory_size,
        .memory_size_available = memory_size_available,
    });
    last_offset = end_offset;
    last_memory_offset = memory_offset;
  }

  return result;
}

static arrow::Result<std::shared_ptr<arrow::Schema>> build_read_schema(
    const std::shared_ptr<arrow::Schema>& file_schema,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& needed_columns) {
  if (read_schema) {
    return read_schema;
  }
  if (!file_schema) {
    return arrow::Status::Invalid("Lance file schema is not available");
  }
  if (needed_columns.empty()) {
    return file_schema;
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto& col : needed_columns) {
    auto field = file_schema->GetFieldByName(col);
    if (!field) {
      return arrow::Status::Invalid(
          fmt::format("Lance column '{}' not found in fragment schema: {}", col, file_schema->ToString()));
    }
    fields.push_back(field);
  }
  return arrow::schema(fields);
}

std::string LanceTableReader::MetaTrait::cache_key(const milvus_storage::api::ColumnGroupFile& file) {
  auto parsed_uri = ParseLanceUri(file.path);
  if (!parsed_uri.ok()) {
    // load_metadata() will return the detailed URI error. Keep malformed URIs
    // distinct here so cache lookup itself remains infallible.
    LOG_STORAGE_WARNING_ << "Failed to parse Lance URI while building metadata cache key"
                         << ", path=" << file.path << ", status=" << parsed_uri.status().ToString();
    return fmt::format("lance-table|invalid-uri:{}", file.path);
  }
  return fmt::format("lance-table|base-uri:{}", parsed_uri->first);
}

static arrow::Result<std::shared_ptr<const LanceTableReader::MetaTrait::FragmentMetadata>> load_fragment_metadata(
    uint64_t fragment_id,
    const milvus_storage::api::Properties& properties,
    const std::shared_ptr<BlockingDataset>& dataset) {
  std::shared_ptr<arrow::Schema> file_schema;
  {
    ArrowSchema c_fragment_schema{};
    ARROW_RETURN_NOT_OK(dataset->GetFragmentSchema(fragment_id, c_fragment_schema));
    ARROW_ASSIGN_OR_RAISE(file_schema, arrow::ImportSchema(&c_fragment_schema));
  }

  ARROW_ASSIGN_OR_RAISE(auto logical_rows, dataset->GetFragmentRowCount(fragment_id));
  ARROW_ASSIGN_OR_RAISE(auto physical_rows, dataset->GetFragmentPhysicalRowCount(fragment_id));
  if (physical_rows < logical_rows) {
    return arrow::Status::Invalid("Fragment ", fragment_id, " has inconsistent metadata: physical_rows (",
                                  physical_rows, ") < logical_rows (", logical_rows, ")");
  }

  ARROW_ASSIGN_OR_RAISE(auto logical_chunk_rows,
                        milvus_storage::api::GetValue<uint64_t>(properties, PROPERTY_READER_LOGICAL_CHUNK_ROWS));

  auto column_memory_sizes_result =
      estimate_fragment_column_memory_sizes(*dataset, fragment_id, static_cast<size_t>(file_schema->num_fields()));
  const bool memory_size_available = column_memory_sizes_result.ok();
  std::vector<uint64_t> fragment_column_memory_sizes;
  if (memory_size_available) {
    fragment_column_memory_sizes = std::move(column_memory_sizes_result).ValueOrDie();
  } else {
    // Memory statistics are optional. Do not retain the underlying failure in
    // metadata: estimate APIs return a generic NotImplemented status instead.
    // Keep the detailed reason in the debug log for diagnostics only.
    LOG_STORAGE_DEBUG_ << "Lance column memory estimation is unavailable while loading metadata"
                       << ", fragment_id=" << fragment_id
                       << ", status=" << column_memory_sizes_result.status().ToString();
  }
  ARROW_ASSIGN_OR_RAISE(
      auto row_group_infos,
      create_row_group_infos(logical_rows, logical_chunk_rows, fragment_column_memory_sizes, memory_size_available));

  auto fragment_metadata = std::make_shared<LanceTableReader::MetaTrait::FragmentMetadata>();
  fragment_metadata->file_schema = std::move(file_schema);
  fragment_metadata->row_group_infos = std::move(row_group_infos);
  fragment_metadata->num_deletions = physical_rows - logical_rows;
  fragment_metadata->logical_chunk_rows = logical_chunk_rows;
  fragment_metadata->column_memory_weights =
      memory_size_available ? std::make_shared<const std::vector<uint64_t>>(std::move(fragment_column_memory_sizes))
                            : nullptr;

  std::shared_ptr<const LanceTableReader::MetaTrait::FragmentMetadata> result = fragment_metadata;
  return result;
}

arrow::Result<LanceTableReader::MetaTrait::MetadataPtr> LanceTableReader::MetaTrait::load_metadata(
    const milvus_storage::api::ColumnGroupFile& file,
    const milvus_storage::api::Properties& properties,
    const KeyRetriever& key_retriever) {
  (void)key_retriever;

  ARROW_ASSIGN_OR_RAISE(auto parsed_uri, ParseLanceUri(file.path));
  auto base_uri = std::move(parsed_uri.first);
  const auto fragment_id = parsed_uri.second;

  ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties, base_uri));
  ARROW_ASSIGN_OR_RAISE(auto fs_config, FilesystemCache::resolve_config(properties, base_uri));
  const auto lance_uri = ToStandardLanceUri(base_uri);
  const auto reader_options = ToReaderOptions(fs_config);

  // New manifests persist the Dataset snapshot selected by explore(), letting
  // this path identify the global cache entry before any heavyweight open.
  // Legacy manifests have no version property, so resolve only Lance's latest
  // manifest location; this fallback does not load or decode the manifest.
  uint64_t dataset_version = 0;
  const auto version_it = file.properties.find(kDatasetVersionProperty);
  if (version_it != file.properties.end()) {
    const auto [valid, version] = api::convert::convertFunc<uint64_t>(version_it->second);
    if (!valid) {
      return arrow::Status::Invalid("Invalid Lance dataset version for file ", file.path, ": ", version_it->second);
    }
    dataset_version = version;
  }
  if (dataset_version == 0) {
    ARROW_ASSIGN_OR_RAISE(dataset_version, BlockingDataset::ResolveLatestVersion(lance_uri, fs, reader_options));
  }

  // URI alone is insufficient: the same table can have multiple live snapshots,
  // and one URI may resolve through filesystems with different credential
  // identities. A cache miss opens the exact resolved version so a concurrent
  // commit cannot change which snapshot is published under this key.
  const LanceDatasetCache::Key dataset_cache_key{
      .version = dataset_version,
      .base_uri = base_uri,
      .filesystem_cache_key = fs_config.GetCacheKey(),
  };
  ARROW_ASSIGN_OR_RAISE(auto dataset, LanceDatasetCache::Instance().GetOrOpen(dataset_cache_key, [&]() {
    return BlockingDataset::Open(lance_uri, fs, reader_options, dataset_version);
  }));

  auto fragment_metadata_cache = std::make_shared<FragmentMetadataCache>();
  ARROW_ASSIGN_OR_RAISE(auto fragment_metadata, fragment_metadata_cache->get_or_load(fragment_id, [&]() {
    return load_fragment_metadata(fragment_id, properties, dataset);
  }));

  auto metadata = std::make_shared<Metadata>();
  metadata->cache_key = cache_key(file);
  metadata->path = base_uri;
  // FileFragment::schema() is the current Dataset schema in Lance 7, so it is
  // valid at the Dataset-level outer metadata. Fragment row groups live only
  // in FragmentMetadata and are selected by create_from_metadata().
  metadata->file_schema = fragment_metadata->file_schema;
  metadata->cache_size = sizeof(Metadata);
  metadata->payload = Payload{
      .base_uri = std::move(base_uri),
      .filesystem = std::move(fs),
      .dataset = std::move(dataset),
      .fragment_metadata_cache = std::move(fragment_metadata_cache),
      .properties = properties,
  };

  MetadataPtr result = metadata;
  return result;
}

arrow::Result<std::shared_ptr<LanceTableReader>> LanceTableReader::MetaTrait::create_from_metadata(
    MetadataPtr metadata,
    const milvus_storage::api::ColumnGroupFile& file,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& needed_columns,
    const std::string& predicate) {
  (void)predicate;
  if (!metadata) {
    return arrow::Status::Invalid("Cannot open Lance reader from null metadata");
  }
  if (!metadata->payload.filesystem || !metadata->payload.dataset || !metadata->payload.fragment_metadata_cache) {
    return arrow::Status::Invalid("Cannot open Lance reader from incomplete metadata");
  }

  ARROW_ASSIGN_OR_RAISE(auto parsed_uri, ParseLanceUri(file.path));
  const auto& base_uri = parsed_uri.first;
  const auto fragment_id = parsed_uri.second;
  if (base_uri != metadata->payload.base_uri) {
    return arrow::Status::Invalid("Lance metadata base URI does not match file URI: ", metadata->payload.base_uri,
                                  " != ", base_uri);
  }

  // Verify that the file's Dataset version matches the cached metadata.
  const auto version_it = file.properties.find(kDatasetVersionProperty);
  if (version_it != file.properties.end()) {
    const auto [valid, version] = api::convert::convertFunc<uint64_t>(version_it->second);
    if (!valid) {
      return arrow::Status::Invalid("Invalid Lance dataset version for file ", file.path, ": ", version_it->second);
    }
    if (version != 0 && version != metadata->payload.dataset->Version()) {
      return arrow::Status::Invalid("Lance dataset version does not match cached metadata for file ", file.path, ": ",
                                    version, " != ", metadata->payload.dataset->Version());
    }
  }

  ARROW_ASSIGN_OR_RAISE(
      auto fragment_metadata, metadata->payload.fragment_metadata_cache->get_or_load(fragment_id, [&]() {
        return load_fragment_metadata(fragment_id, metadata->payload.properties, metadata->payload.dataset);
      }));

  auto reader =
      std::shared_ptr<LanceTableReader>(new LanceTableReader(metadata, fragment_id, read_schema, needed_columns));
  reader->file_schema_ = fragment_metadata->file_schema;
  reader->logical_chunk_rows_ = fragment_metadata->logical_chunk_rows;
  reader->num_deletions_ = fragment_metadata->num_deletions;
  reader->column_memory_weights_ = fragment_metadata->column_memory_weights;
  reader->row_group_infos_ = fragment_metadata->row_group_infos;

  ARROW_ASSIGN_OR_RAISE(auto requested_schema, build_read_schema(reader->file_schema_, read_schema, needed_columns));
  ArrowSchema c_arrow_schema{};
  ARROW_RETURN_NOT_OK(arrow::ExportSchema(*requested_schema, &c_arrow_schema));
  ARROW_ASSIGN_OR_RAISE(reader->fragment_reader_,
                        BlockingFragmentReader::Open(*metadata->payload.dataset, fragment_id, c_arrow_schema));

  return reader;
}

arrow::Status LanceTableReader::open() {
  assert(!fragment_reader_);

  if (!dataset_) {
    // uri_ is in Milvus format (scheme://address/bucket/key) so extfs.<alias>.*
    // can be resolved by address+bucket. Strip the address back to standard form
    // (scheme://bucket/key) before handing to Lance, whose object_store treats
    // the host as the bucket.
    ARROW_ASSIGN_OR_RAISE(auto fs_config, FilesystemCache::resolve_config(properties_, uri_));
    const auto lance_uri = ToStandardLanceUri(uri_);
    const auto reader_options = ToReaderOptions(fs_config);

    // Version zero means latest, but it cannot identify a stable global cache
    // entry. Resolve only the latest manifest location before the cache lookup.
    if (dataset_version_ == 0) {
      ARROW_ASSIGN_OR_RAISE(dataset_version_,
                            BlockingDataset::ResolveLatestVersion(lance_uri, filesystem_, reader_options));
    }

    const LanceDatasetCache::Key dataset_cache_key{
        .version = dataset_version_,
        .base_uri = uri_,
        .filesystem_cache_key = fs_config.GetCacheKey(),
    };
    ARROW_ASSIGN_OR_RAISE(dataset_, LanceDatasetCache::Instance().GetOrOpen(dataset_cache_key, [&]() {
      return BlockingDataset::Open(lance_uri, filesystem_, reader_options, dataset_version_);
    }));
  }

  // Lance 7 exposes the current dataset schema through FileFragment::schema().
  {
    ArrowSchema c_fragment_schema{};
    ARROW_RETURN_NOT_OK(dataset_->GetFragmentSchema(fragment_id_, c_fragment_schema));
    ARROW_ASSIGN_OR_RAISE(file_schema_, arrow::ImportSchema(&c_fragment_schema));
  }

  // Build the read schema for fragment reader:
  // use user-provided schema if available, otherwise project file schema by needed_columns
  ARROW_ASSIGN_OR_RAISE(auto read_schema, build_read_schema(file_schema_, read_schema_, needed_columns_));

  ARROW_ASSIGN_OR_RAISE(logical_chunk_rows_, api::GetValue<uint64_t>(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS));

  ArrowSchema c_arrow_schema{};
  ARROW_RETURN_NOT_OK(arrow::ExportSchema(*read_schema, &c_arrow_schema));
  ARROW_ASSIGN_OR_RAISE(fragment_reader_, BlockingFragmentReader::Open(*dataset_, fragment_id_, c_arrow_schema));

  // Lance's read_range accepts logical indices (post-deletion) and internally
  // patches the range to skip deleted rows. So row_group_infos uses logical row count.
  // However, read_range's batch_size is applied to the *physical* range after
  // patch_range_for_deletions, so we add num_deletions_ to batch_size to ensure
  // each read produces a single output batch.
  ARROW_ASSIGN_OR_RAISE(auto logical_rows, fragment_reader_->RowCount());
  ARROW_ASSIGN_OR_RAISE(auto physical_rows, dataset_->GetFragmentPhysicalRowCount(fragment_id_));
  if (physical_rows < logical_rows) {
    return arrow::Status::Invalid("Fragment ", fragment_id_, " has inconsistent metadata: physical_rows (",
                                  physical_rows, ") < logical_rows (", logical_rows, ")");
  }
  num_deletions_ = physical_rows - logical_rows;

  auto column_memory_sizes_result =
      estimate_fragment_column_memory_sizes(*dataset_, fragment_id_, static_cast<size_t>(file_schema_->num_fields()));
  const bool memory_size_available = column_memory_sizes_result.ok();
  std::vector<uint64_t> fragment_column_memory_sizes;
  if (memory_size_available) {
    fragment_column_memory_sizes = std::move(column_memory_sizes_result).ValueOrDie();
  } else {
    // Memory statistics are optional. Do not retain the underlying failure in
    // row-group metadata: estimate APIs return a generic NotImplemented status
    // instead. Keep the detailed reason in the debug log for diagnostics only.
    LOG_STORAGE_DEBUG_ << "Lance column memory estimation is unavailable while opening the reader"
                       << ", fragment_id=" << fragment_id_
                       << ", status=" << column_memory_sizes_result.status().ToString();
  }
  ARROW_ASSIGN_OR_RAISE(row_group_infos_, create_row_group_infos(logical_rows, logical_chunk_rows_,
                                                                 fragment_column_memory_sizes, memory_size_available));
  column_memory_weights_ = memory_size_available
                               ? std::make_shared<const std::vector<uint64_t>>(std::move(fragment_column_memory_sizes))
                               : nullptr;

  return arrow::Status::OK();
}

std::shared_ptr<arrow::Schema> LanceTableReader::get_schema() const { return file_schema_; }

arrow::Result<std::vector<RowGroupInfo>> LanceTableReader::get_row_group_infos() {
  assert(fragment_reader_);
  return row_group_infos_;
}

arrow::Result<std::vector<uint64_t>> LanceTableReader::get_rg_column_memsz(int64_t row_group_index) const {
  if (row_group_index < 0 || static_cast<size_t>(row_group_index) >= row_group_infos_.size()) {
    return arrow::Status::Invalid("Lance row group index out of range: ", row_group_index);
  }
  if (!row_group_infos_[row_group_index].memory_size_available || !column_memory_weights_) {
    return arrow::Status::NotImplemented("Lance column memory size statistics are not available");
  }
  return DistributeMemorySizes(row_group_infos_[row_group_index].memory_size, *column_memory_weights_);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> LanceTableReader::get_chunk(const int& row_group_index) {
  assert(fragment_reader_);
  auto start_idx = row_group_infos_[row_group_index].start_offset;
  auto end_idx = row_group_infos_[row_group_index].end_offset;
  // FIXME: Lance's read_range may produce multiple output batches for two reasons:
  // 1. batch_size is applied to the *physical* range (after patch_range_for_deletions),
  //    so deletions cause the physical range to exceed batch_size.
  // 2. Lance may split at internal page boundaries regardless of batch_size.
  // We add num_deletions_ to mitigate (1), but (2) is not addressed — if Lance
  // splits at page boundaries, chunk(0) will silently lose trailing rows in Release
  // builds (assert is a no-op). A robust fix would combine all chunks here.
  ARROW_ASSIGN_OR_RAISE(auto array_stream,
                        fragment_reader_->ReadRangesAsStream(start_idx, end_idx, end_idx - start_idx + num_deletions_));
  auto chunkedarray_result = arrow::ImportChunkedArray(&array_stream);
  if (!chunkedarray_result.ok()) {
    return MakeBridgeErrorStatus("Failed to import Lance chunked array", chunkedarray_result.status());
  }
  auto chunkedarray = chunkedarray_result.ValueOrDie();
  assert(chunkedarray != nullptr && chunkedarray->num_chunks() == 1);
  return arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0));
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> LanceTableReader::get_chunks(
    const std::vector<int>& rg_indices_in_file) {
  assert(fragment_reader_);
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;

#ifndef NDEBUG
  // verify rg_indices_in_file have been sorted
  for (size_t i = 1; i < rg_indices_in_file.size(); ++i) {
    assert(rg_indices_in_file[i] >= rg_indices_in_file[i - 1]);
  }
#endif

  std::vector<std::pair<uint64_t, uint64_t>> rg_idx_ranges;

  // calc continuous ranges
  // ex. [1, 2, 3, 5] -> [(1, 3), (5, 5)]
  size_t start_idx = 0;
  for (size_t i = 1; i < rg_indices_in_file.size(); ++i) {
    if (rg_indices_in_file[i] != rg_indices_in_file[i - 1] + 1) {
      rg_idx_ranges.emplace_back(rg_indices_in_file[start_idx], rg_indices_in_file[i - 1]);
      start_idx = i;
    }
  }

  if (start_idx < rg_indices_in_file.size()) {
    rg_idx_ranges.emplace_back(rg_indices_in_file[start_idx], rg_indices_in_file.back());
  }

  for (const auto& rg_range : rg_idx_ranges) {
    // load continuous chunks in one read
    const auto& start_rg_info = row_group_infos_[rg_range.first];
    const auto& end_rg_info = row_group_infos_[rg_range.second];

    // batch_size adds num_deletions_ for the same reason as get_chunk — see comment there.
    ARROW_ASSIGN_OR_RAISE(auto array_stream, fragment_reader_->ReadRangesAsStream(
                                                 start_rg_info.start_offset, end_rg_info.end_offset,
                                                 end_rg_info.end_offset - start_rg_info.start_offset + num_deletions_));
    auto chunkedarray_result = arrow::ImportChunkedArray(&array_stream);
    if (!chunkedarray_result.ok()) {
      return MakeBridgeErrorStatus("Failed to import Lance chunked array", chunkedarray_result.status());
    }
    auto chunkedarray = chunkedarray_result.ValueOrDie();
    assert(chunkedarray != nullptr);

    // assign to rbs
    for (size_t j = 0; j < chunkedarray->num_chunks(); ++j) {
      ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(j)));
      rbs.emplace_back(rb);
    }
  }

  return rbs;
}

arrow::Result<std::shared_ptr<arrow::Table>> LanceTableReader::take(const std::vector<int64_t>& row_indices) {
  assert(fragment_reader_);
  ARROW_ASSIGN_OR_RAISE(auto array_stream, fragment_reader_->TakeAsStream(row_indices, row_indices.size()));
  auto chunkedarray_result = arrow::ImportChunkedArray(&array_stream);
  if (!chunkedarray_result.ok()) {
    return MakeBridgeErrorStatus("Failed to import Lance take result", chunkedarray_result.status());
  }
  auto chunkedarray = chunkedarray_result.ValueOrDie();

  // out of range
  if (chunkedarray->num_chunks() == 0) {
    ARROW_ASSIGN_OR_RAISE(auto row_count, fragment_reader_->RowCount());
    return arrow::Status::Invalid(fmt::format("out of row range [0, {}]", row_count));
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
  for (size_t i = 0; i < chunkedarray->num_chunks(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(i)));
    rbs.emplace_back(rb);
  }

  return arrow::Table::FromRecordBatches(rbs);
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> LanceTableReader::read_with_range(const uint64_t& start_offset,
                                                                                           const uint64_t& end_offset) {
  assert(fragment_reader_);
  // Lance's read_range accepts logical indices directly.
  // batch_size adds num_deletions_ for the same reason as get_chunk — see comment there.
  ARROW_ASSIGN_OR_RAISE(auto array_stream, fragment_reader_->ReadRangesAsStream(
                                               start_offset, end_offset, end_offset - start_offset + num_deletions_));
  ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ImportRecordBatchReader(&array_stream));
  return internal::WrapLanceRecordBatchReader(std::move(reader));
}

arrow::Result<std::shared_ptr<FormatReader>> LanceTableReader::clone_reader() {
  assert(fragment_reader_);  // already opened
  return this->shared_from_this();
}

}  // namespace milvus_storage::lance
