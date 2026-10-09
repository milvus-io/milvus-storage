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
#include <atomic>
#include <condition_variable>
#include <exception>
#include <limits>
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
#include <glog/raw_logging.h>
#include <folly/ScopeGuard.h>
#include <folly/Try.h>
#include <folly/futures/Promise.h>
#include <folly/futures/SharedPromise.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "runtime/bridge_util.h"

namespace milvus_storage::lance {

template <typename T, typename ImportFn>
struct LanceAsyncContext {
  folly::Promise<arrow::Result<T>> promise;
  const char* const operation;
  ImportFn import_result;
  ArrowArrayStream stream{};

  LanceAsyncContext(const char* operation, ImportFn import_result)
      : operation(operation), import_result(std::move(import_result)) {}

  ~LanceAsyncContext() {
    // Arrow clears release after import; otherwise the context still owns the stream.
    if (stream.release) {
      stream.release(&stream);
    }
  }

  static void Complete(void* raw, uint64_t value, const char* error) noexcept {
    // Consume every result, even if the caller has dropped its future.
    std::unique_ptr<LanceAsyncContext> context(static_cast<LanceAsyncContext*>(raw));
    try {
      // Convert import failures before fulfilling the promise; never retry setValue.
      auto imported = folly::makeTryWith([&]() -> arrow::Result<T> {
        if (error) {
          return MakeBridgeErrorStatus(context->operation, error);
        }
        return context->import_result(value, context->stream);
      });
      auto result =
          imported.hasException()
              ? arrow::Result<T>(MakeBridgeErrorStatus(context->operation, imported.exception().what().toStdString()))
              : std::move(imported).value();
      context->promise.setValue(std::move(result));
    } catch (const std::exception& e) {
      // No exception may cross the Rust callback boundary. An unfulfilled
      // promise reports BrokenPromise on destruction; retain the cause here.
      RAW_LOG(ERROR, "%s: failed to publish callback result: %s", context->operation, e.what());
    } catch (...) {
      RAW_LOG(ERROR, "%s: unknown exception publishing callback result", context->operation);
    }
  }
};

template <typename T, typename ImportFn, typename SubmitFn>
static folly::SemiFuture<arrow::Result<T>> submit_lance_async(const char* operation,
                                                              ImportFn import_result,
                                                              SubmitFn&& submit) {
  using Context = LanceAsyncContext<T, ImportFn>;
  auto context = std::make_unique<Context>(operation, std::move(import_result));
  auto future = context->promise.getSemiFuture();
  // Completion can race with submission returning. Only failed submission
  // leaves ownership here; otherwise the callback reclaims the context.
  auto* raw = context.release();
  auto status = arrow::Status::OK();
  try {
    status = submit(&Context::Complete, raw);
  } catch (const std::exception& e) {
    status = MakeBridgeErrorStatus(operation, e.what());
  } catch (...) {
    status = arrow::Status::IOError(operation, ": unknown exception during submission");
  }
  if (!status.ok()) {
    std::unique_ptr<Context> failed(raw);
    failed->promise.setValue(std::move(status));
  }
  return future;
}

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

class LanceTableReader::MetaTrait::FragmentMetadataCache final
    : public std::enable_shared_from_this<LanceTableReader::MetaTrait::FragmentMetadataCache> {
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
      if (!inserted && !in_flight_load->async_leader) {
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

    if (in_flight_load->async_leader) {
      if (!status.ok()) {
        return status;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      return fragments_.try_emplace(fragment_id, std::move(metadata)).first->second;
    }
    return complete_load(fragment_id, in_flight_load,
                         status.ok() ? arrow::Result<FragmentMetadataPtr>(metadata) : status);
  }

  template <typename FragmentMetadataLoader>
  folly::SemiFuture<arrow::Result<FragmentMetadataPtr>> get_or_load_async(uint64_t fragment_id,
                                                                          FragmentMetadataLoader load_fn) {
    auto self = shared_from_this();
    return folly::makeSemiFuture().deferValue(
        [self = std::move(self), fragment_id,
         load_fn = std::move(load_fn)](folly::Unit) -> folly::SemiFuture<arrow::Result<FragmentMetadataPtr>> {
          std::shared_ptr<InFlightLoad> flight;
          {
            std::lock_guard<std::mutex> lock(self->mutex_);
            auto cached = self->fragments_.find(fragment_id);
            if (cached != self->fragments_.end()) {
              return folly::makeSemiFuture(arrow::Result<FragmentMetadataPtr>(cached->second));
            }
            auto [it, inserted] = self->in_flight_loads_.try_emplace(fragment_id, std::make_shared<InFlightLoad>());
            flight = it->second;
            if (!inserted) {
              return flight->async_result.getSemiFuture();
            }
            flight->async_leader = true;
          }
          try {
            auto future = load_fn();
            auto abandoned = folly::makeGuard([self, fragment_id, flight]() noexcept {
              if (!flight->continuation_attached.load(std::memory_order_acquire)) {
                return;
              }
              auto cleanup = folly::makeTryWith([&] {
                return self->complete_load(fragment_id, flight,
                                           arrow::Status::Cancelled("Lance fragment metadata load was abandoned"));
              });
              if (cleanup.hasException()) {
                const auto* error = cleanup.exception().get_exception();
                RAW_LOG(ERROR, "Failed to abandon Lance fragment metadata load: %s",
                        error ? error->what() : "unknown exception");
              }
            });
            auto pending =
                std::move(future).defer([self, fragment_id, flight, abandoned = std::move(abandoned)](
                                            folly::Try<arrow::Result<FragmentMetadataPtr>>&& result) mutable {
                  auto loaded = result.hasException()
                                    ? arrow::Result<FragmentMetadataPtr>(arrow::Status::UnknownError(
                                          "Exception while asynchronously loading Lance fragment metadata: ",
                                          result.exception().what().toStdString()))
                                    : std::move(result).value();
                  auto completed = self->complete_load(fragment_id, flight, std::move(loaded));
                  abandoned.dismiss();
                  return completed;
                });
            // Attachment failures belong to the catch below, not to cancellation.
            flight->continuation_attached.store(true, std::memory_order_release);
            return pending;
          } catch (const std::exception& error) {
            return folly::makeSemiFuture(self->complete_load(
                fragment_id, flight,
                arrow::Status::UnknownError("Failed to submit Lance fragment metadata load: ", error.what())));
          } catch (...) {
            return folly::makeSemiFuture(self->complete_load(
                fragment_id, flight,
                arrow::Status::UnknownError("Unknown exception submitting Lance fragment metadata load")));
          }
        });
  }

  private:
  struct InFlightLoad {
    bool done = false;
    bool async_leader = false;
    std::atomic<bool> continuation_attached{false};
    arrow::Status status = arrow::Status::OK();
    FragmentMetadataPtr metadata;
    std::condition_variable cv;
    folly::SharedPromise<arrow::Result<FragmentMetadataPtr>> async_result;
  };

  arrow::Result<FragmentMetadataPtr> complete_load(uint64_t fragment_id,
                                                   const std::shared_ptr<InFlightLoad>& in_flight_load,
                                                   arrow::Result<FragmentMetadataPtr> result) {
    auto status = result.status();
    auto metadata = result.ok() ? std::move(result).ValueOrDie() : nullptr;
    if (status.ok() && !metadata) {
      status = arrow::Status::Invalid("Lance fragment metadata loader returned null for fragment ID: ", fragment_id);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (in_flight_load->done) {
        return in_flight_load->status.ok() ? arrow::Result<FragmentMetadataPtr>(in_flight_load->metadata)
                                           : in_flight_load->status;
      }
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
    arrow::Result<FragmentMetadataPtr> completed = status.ok() ? arrow::Result<FragmentMetadataPtr>(metadata) : status;
    in_flight_load->async_result.setValue(completed);
    return completed;
  }

  std::mutex mutex_;
  std::unordered_map<uint64_t, FragmentMetadataPtr> fragments_;
  std::unordered_map<uint64_t, std::shared_ptr<InFlightLoad>> in_flight_loads_;
};

folly::SemiFuture<arrow::Result<std::shared_ptr<BlockingDataset>>> LanceTableReader::load_dataset_async(
    const std::string& lance_uri,
    const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
    const StorageOptions& read_options,
    uint64_t version) {
  return submit_lance_async<std::shared_ptr<BlockingDataset>>(
      "Failed to open Lance dataset",
      [](uint64_t handle, ArrowArrayStream&) -> arrow::Result<std::shared_ptr<BlockingDataset>> {
        return BlockingDataset::FromHandle(handle);
      },
      [&](LanceAsyncCallback callback, void* context) {
        return BlockingDataset::OpenAsync(lance_uri, filesystem, read_options, version, callback, context);
      });
}

folly::SemiFuture<arrow::Result<std::shared_ptr<BlockingDataset>>> LanceTableReader::open_dataset_async(
    const std::string& base_uri,
    const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
    const api::Properties& properties,
    uint64_t version) {
  FOLLY_ARROW_ASSIGN_OR_RAISE(auto fs_config, FilesystemCache::resolve_config(properties, base_uri));
  const auto lance_uri = ToStandardLanceUri(base_uri);
  const auto options = ToReaderOptions(fs_config);
  auto resolved_version =
      version == 0
          ? submit_lance_async<uint64_t>(
                "Failed to resolve latest Lance dataset version",
                [](uint64_t resolved, ArrowArrayStream&) -> arrow::Result<uint64_t> { return resolved; },
                [&](LanceAsyncCallback callback, void* context) {
                  return BlockingDataset::ResolveLatestVersionAsync(lance_uri, filesystem, options, callback, context);
                })
          : folly::makeSemiFuture(arrow::Result<uint64_t>(version));
  return std::move(resolved_version)
      .deferValue(
          [base_uri, filesystem, options, filesystem_key = fs_config.GetCacheKey()](
              arrow::Result<uint64_t> result) -> folly::SemiFuture<arrow::Result<std::shared_ptr<BlockingDataset>>> {
            FOLLY_ARROW_ASSIGN_OR_RAISE(const auto version, std::move(result));
            return get_or_open_dataset_async(base_uri, filesystem, options, filesystem_key, version);
          });
}

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

static folly::SemiFuture<arrow::Result<std::shared_ptr<const LanceTableReader::MetaTrait::FragmentMetadata>>>
load_fragment_metadata_async(uint64_t fragment_id,
                             const milvus_storage::api::Properties& properties,
                             const std::shared_ptr<BlockingDataset>& dataset) {
  // These getters inspect the loaded manifest. Only the memory estimator needs
  // footer/page I/O, which is submitted to the shared Rust runtime below.
  std::shared_ptr<arrow::Schema> file_schema;
  {
    ArrowSchema c_fragment_schema{};
    FOLLY_ARROW_RETURN_NOT_OK(dataset->GetFragmentSchema(fragment_id, c_fragment_schema));
    FOLLY_ARROW_ASSIGN_OR_RAISE(file_schema, arrow::ImportSchema(&c_fragment_schema));
  }
  FOLLY_ARROW_ASSIGN_OR_RAISE(auto logical_rows, dataset->GetFragmentRowCount(fragment_id));
  FOLLY_ARROW_ASSIGN_OR_RAISE(auto physical_rows, dataset->GetFragmentPhysicalRowCount(fragment_id));
  if (physical_rows < logical_rows) {
    return folly::makeSemiFuture(arrow::Result<std::shared_ptr<const LanceTableReader::MetaTrait::FragmentMetadata>>(
        arrow::Status::Invalid("Fragment ", fragment_id, " has inconsistent metadata: physical_rows (", physical_rows,
                               ") < logical_rows (", logical_rows, ")")));
  }
  FOLLY_ARROW_ASSIGN_OR_RAISE(auto logical_chunk_rows,
                              milvus_storage::api::GetValue<uint64_t>(properties, PROPERTY_READER_LOGICAL_CHUNK_ROWS));
  auto metadata = std::make_shared<LanceTableReader::MetaTrait::FragmentMetadata>();
  metadata->file_schema = std::move(file_schema);
  metadata->num_deletions = physical_rows - logical_rows;
  metadata->logical_chunk_rows = logical_chunk_rows;
  auto estimates = [&]() -> folly::SemiFuture<arrow::Result<std::vector<uint64_t>>> {
    FIU_RETURN_ON(FIUKEY_MEMORY_SIZE_ESTIMATION_FAIL,
                  folly::makeSemiFuture(arrow::Result<std::vector<uint64_t>>(
                      arrow::Status::NotImplemented("Injected fault: ", FIUKEY_MEMORY_SIZE_ESTIMATION_FAIL))));
    return submit_lance_async<std::vector<uint64_t>>(
               "Failed to estimate Lance column memory",
               [](uint64_t handle, ArrowArrayStream&) -> arrow::Result<std::vector<uint64_t>> {
                 return BlockingDataset::TakeColumnMemoryResult(handle);
               },
               [&](LanceAsyncCallback callback, void* context) {
                 return dataset->EstimateFragmentColumnMemoryAsync(fragment_id, callback, context);
               })
        .deferValue([](arrow::Result<std::vector<uint64_t>> result) -> arrow::Result<std::vector<uint64_t>> {
          if (!result.ok()) {
            return arrow::Status::NotImplemented("Lance column memory size estimation is not available: ",
                                                 result.status().message());
          }
          return result;
        });
  }();
  return std::move(estimates).deferValue(
      [dataset, fragment_id, logical_rows, metadata = std::move(metadata)](arrow::Result<std::vector<uint64_t>> result)
          -> arrow::Result<std::shared_ptr<const LanceTableReader::MetaTrait::FragmentMetadata>> {
        if (result.ok()) {
          const auto& sizes = *result;
          if (sizes.size() != static_cast<size_t>(metadata->file_schema->num_fields())) {
            result = arrow::Status::Invalid("Lance column memory estimate count does not match the file schema");
          } else {
            uint64_t total = 0;
            for (const auto size : sizes) {
              if (size > std::numeric_limits<uint64_t>::max() - total) {
                result = arrow::Status::Invalid("Lance column memory estimates exceed the uint64_t range");
                break;
              }
              total += size;
            }
          }
        }
        const bool memory_size_available = result.ok();
        std::vector<uint64_t> sizes;
        if (memory_size_available) {
          sizes = std::move(result).ValueOrDie();
        } else {
          // Preserve optional-estimate behavior and retain the failure in logs.
          LOG_STORAGE_DEBUG_ << "Lance column memory estimation is unavailable while loading metadata"
                             << ", fragment_id=" << fragment_id << ", status=" << result.status().ToString();
        }
        ARROW_ASSIGN_OR_RAISE(
            metadata->row_group_infos,
            create_row_group_infos(logical_rows, metadata->logical_chunk_rows, sizes, memory_size_available));
        if (memory_size_available) {
          metadata->column_memory_weights = std::make_shared<const std::vector<uint64_t>>(std::move(sizes));
        }
        std::shared_ptr<const LanceTableReader::MetaTrait::FragmentMetadata> immutable = std::move(metadata);
        return immutable;
      });
}

static arrow::Result<std::shared_ptr<const LanceTableReader::MetaTrait::FragmentMetadata>> load_fragment_metadata(
    uint64_t fragment_id,
    const milvus_storage::api::Properties& properties,
    const std::shared_ptr<BlockingDataset>& dataset) {
  return load_fragment_metadata_async(fragment_id, properties, dataset).get();
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
  ARROW_ASSIGN_OR_RAISE(auto dataset,
                        get_or_open_dataset(base_uri, fs, reader_options, fs_config.GetCacheKey(), dataset_version));

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

folly::SemiFuture<arrow::Result<LanceTableReader::MetaTrait::MetadataPtr>>
LanceTableReader::MetaTrait::load_metadata_async(const api::ColumnGroupFile& file,
                                                 const api::Properties& properties,
                                                 const KeyRetriever& /*key_retriever*/) {
  return folly::makeSemiFuture().deferValue(
      [file, properties](folly::Unit) -> folly::SemiFuture<arrow::Result<MetadataPtr>> {
        FOLLY_ARROW_ASSIGN_OR_RAISE(auto parsed_uri, ParseLanceUri(file.path));
        auto base_uri = std::move(parsed_uri.first);
        const auto fragment_id = parsed_uri.second;
        FOLLY_ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties, base_uri));
        uint64_t version = 0;
        const auto version_it = file.properties.find(kDatasetVersionProperty);
        if (version_it != file.properties.end()) {
          const auto [valid, parsed_version] = api::convert::convertFunc<uint64_t>(version_it->second);
          if (!valid) {
            return folly::makeSemiFuture(arrow::Result<MetadataPtr>(arrow::Status::Invalid(
                "Invalid Lance dataset version for file ", file.path, ": ", version_it->second)));
          }
          version = parsed_version;
        }
        return open_dataset_async(base_uri, fs, properties, version)
            .deferValue(
                [file, properties, base_uri, fs, fragment_id](arrow::Result<std::shared_ptr<BlockingDataset>> result)
                    -> folly::SemiFuture<arrow::Result<MetadataPtr>> {
                  FOLLY_ARROW_ASSIGN_OR_RAISE(auto dataset, std::move(result));
                  auto cache = std::make_shared<FragmentMetadataCache>();
                  return cache
                      ->get_or_load_async(fragment_id,
                                          [fragment_id, properties, dataset]() {
                                            return load_fragment_metadata_async(fragment_id, properties, dataset);
                                          })
                      .deferValue([file, properties, base_uri, fs, dataset,
                                   cache](arrow::Result<std::shared_ptr<const FragmentMetadata>> fragment_result)
                                      -> arrow::Result<MetadataPtr> {
                        ARROW_ASSIGN_OR_RAISE(auto fragment, std::move(fragment_result));
                        auto metadata = std::make_shared<Metadata>();
                        metadata->cache_key = cache_key(file);
                        metadata->path = base_uri;
                        metadata->file_schema = fragment->file_schema;
                        metadata->cache_size = sizeof(Metadata);
                        metadata->payload = Payload{base_uri, fs, dataset, cache, properties};
                        MetadataPtr immutable = std::move(metadata);
                        return immutable;
                      });
                });
      });
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
  // Preserve legacy output schema metadata semantics; field metadata remains intact.
  reader->read_schema_ = requested_schema->metadata() ? requested_schema->RemoveMetadata() : requested_schema;
  ARROW_ASSIGN_OR_RAISE(reader->fragment_reader_,
                        BlockingFragmentReader::Open(*metadata->payload.dataset, fragment_id, c_arrow_schema));

  return reader;
}

folly::SemiFuture<arrow::Result<std::shared_ptr<LanceTableReader>>>
LanceTableReader::MetaTrait::create_from_metadata_async(MetadataPtr metadata,
                                                        const api::ColumnGroupFile& file,
                                                        const std::shared_ptr<arrow::Schema>& read_schema,
                                                        const std::vector<std::string>& needed_columns,
                                                        const std::string& /*predicate*/) {
  return folly::makeSemiFuture().deferValue([metadata = std::move(metadata), file, read_schema,
                                             needed_columns](folly::Unit)
                                                -> folly::SemiFuture<arrow::Result<std::shared_ptr<LanceTableReader>>> {
    if (!metadata || !metadata->payload.filesystem || !metadata->payload.dataset ||
        !metadata->payload.fragment_metadata_cache) {
      return folly::makeSemiFuture(arrow::Result<std::shared_ptr<LanceTableReader>>(
          arrow::Status::Invalid("Cannot open Lance reader from incomplete metadata")));
    }
    FOLLY_ARROW_ASSIGN_OR_RAISE(auto parsed_uri, ParseLanceUri(file.path));
    if (parsed_uri.first != metadata->payload.base_uri) {
      return folly::makeSemiFuture(arrow::Result<std::shared_ptr<LanceTableReader>>(arrow::Status::Invalid(
          "Lance metadata base URI does not match file URI: ", metadata->payload.base_uri, " != ", parsed_uri.first)));
    }
    const auto version_it = file.properties.find(kDatasetVersionProperty);
    if (version_it != file.properties.end()) {
      const auto [valid, version] = api::convert::convertFunc<uint64_t>(version_it->second);
      if (!valid) {
        return folly::makeSemiFuture(arrow::Result<std::shared_ptr<LanceTableReader>>(
            arrow::Status::Invalid("Invalid Lance dataset version for file ", file.path, ": ", version_it->second)));
      }
      if (version != 0 && version != metadata->payload.dataset->Version()) {
        return folly::makeSemiFuture(arrow::Result<std::shared_ptr<LanceTableReader>>(
            arrow::Status::Invalid("Lance dataset version does not match cached metadata for file ", file.path, ": ",
                                   version, " != ", metadata->payload.dataset->Version())));
      }
    }
    const auto fragment_id = parsed_uri.second;
    auto reader =
        std::shared_ptr<LanceTableReader>(new LanceTableReader(metadata, fragment_id, read_schema, needed_columns));
    return metadata->payload.fragment_metadata_cache
        ->get_or_load_async(fragment_id,
                            [metadata, fragment_id]() {
                              return load_fragment_metadata_async(fragment_id, metadata->payload.properties,
                                                                  metadata->payload.dataset);
                            })
        .deferValue([reader](arrow::Result<std::shared_ptr<const FragmentMetadata>> result)
                        -> folly::SemiFuture<arrow::Result<std::shared_ptr<LanceTableReader>>> {
          FOLLY_ARROW_ASSIGN_OR_RAISE(auto fragment, std::move(result));
          return reader->open_fragment_async(fragment).deferValue(
              [reader](arrow::Status status) -> arrow::Result<std::shared_ptr<LanceTableReader>> {
                ARROW_RETURN_NOT_OK(status);
                return reader;
              });
        });
  });
}

folly::SemiFuture<arrow::Status> LanceTableReader::open_fragment_async(
    const std::shared_ptr<const MetaTrait::FragmentMetadata>& metadata) {
  auto self = shared_from_this();
  FOLLY_ARROW_ASSIGN_OR_RAISE(auto requested_schema,
                              build_read_schema(metadata->file_schema, read_schema_, needed_columns_));
  ArrowSchema c_schema{};
  FOLLY_ARROW_RETURN_NOT_OK(arrow::ExportSchema(*requested_schema, &c_schema));
  ArrowCDataReleaseGuard schema_guard(&c_schema);
  // OpenAsync consumes the C schema before returning. Publish reader state only
  // after the native fragment open succeeds; all captures outlive the operation.
  return submit_lance_async<std::unique_ptr<BlockingFragmentReader>>(
             "Failed to open Lance fragment reader",
             [](uint64_t handle, ArrowArrayStream&) -> arrow::Result<std::unique_ptr<BlockingFragmentReader>> {
               return BlockingFragmentReader::FromHandle(handle);
             },
             [&](LanceAsyncCallback callback, void* context) {
               return BlockingFragmentReader::OpenAsync(*dataset_, fragment_id_, c_schema, callback, context);
             })
      .deferValue([self, metadata,
                   requested_schema](arrow::Result<std::unique_ptr<BlockingFragmentReader>> result) -> arrow::Status {
        ARROW_ASSIGN_OR_RAISE(auto fragment_reader, std::move(result));
        self->file_schema_ = metadata->file_schema;
        self->logical_chunk_rows_ = metadata->logical_chunk_rows;
        self->num_deletions_ = metadata->num_deletions;
        self->column_memory_weights_ = metadata->column_memory_weights;
        self->row_group_infos_ = metadata->row_group_infos;
        self->read_schema_ = requested_schema->metadata() ? requested_schema->RemoveMetadata() : requested_schema;
        self->fragment_reader_ = std::move(fragment_reader);
        return arrow::Status::OK();
      });
}

folly::SemiFuture<arrow::Status> LanceTableReader::open_async() {
  assert(!fragment_reader_);
  auto self = shared_from_this();
  auto dataset = dataset_ ? folly::makeSemiFuture(arrow::Result<std::shared_ptr<BlockingDataset>>(dataset_))
                          : open_dataset_async(uri_, filesystem_, properties_, dataset_version_);
  return std::move(dataset).deferValue(
      [self](arrow::Result<std::shared_ptr<BlockingDataset>> result) -> folly::SemiFuture<arrow::Status> {
        FOLLY_ARROW_ASSIGN_OR_RAISE(self->dataset_, std::move(result));
        self->dataset_version_ = self->dataset_->Version();
        return load_fragment_metadata_async(self->fragment_id_, self->properties_, self->dataset_)
            .deferValue([self](arrow::Result<std::shared_ptr<const MetaTrait::FragmentMetadata>> metadata)
                            -> folly::SemiFuture<arrow::Status> {
              FOLLY_ARROW_ASSIGN_OR_RAISE(auto fragment, std::move(metadata));
              return self->open_fragment_async(fragment);
            });
      });
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

    ARROW_ASSIGN_OR_RAISE(
        dataset_, get_or_open_dataset(uri_, filesystem_, reader_options, fs_config.GetCacheKey(), dataset_version_));
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
  read_schema_ = read_schema->metadata() ? read_schema->RemoveMetadata() : read_schema;
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

folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> LanceTableReader::take_async(
    const std::vector<int64_t>& row_indices) {
  assert(fragment_reader_);
  if (row_indices.empty()) {
    return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(
        arrow::Status::Invalid("Lance take_async requires a nonempty row selection")));
  }
  std::vector<uint32_t> converted;
  converted.reserve(row_indices.size());
  for (const auto index : row_indices) {
    if (index < 0 || static_cast<uint64_t>(index) > std::numeric_limits<uint32_t>::max()) {
      return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(
          arrow::Status::Invalid("Lance row index is outside the uint32 range")));
    }
    converted.push_back(static_cast<uint32_t>(index));
  }
  return submit_lance_async<std::shared_ptr<arrow::Table>>(
      "Failed to take Lance rows asynchronously",
      [reader = shared_from_this()](uint64_t,
                                    ArrowArrayStream& stream) -> arrow::Result<std::shared_ptr<arrow::Table>> {
        // Keep the reader alive until its output has been imported.
        (void)reader;
        ARROW_ASSIGN_OR_RAISE(auto chunks, arrow::ImportChunkedArray(&stream));
        if (chunks->num_chunks() == 0) {
          return arrow::Status::Invalid("Lance take_async returned no rows");
        }
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        batches.reserve(chunks->num_chunks());
        for (const auto& chunk : chunks->chunks()) {
          ARROW_ASSIGN_OR_RAISE(auto batch, arrow::RecordBatch::FromStructArray(chunk));
          batches.push_back(std::move(batch));
        }
        return arrow::Table::FromRecordBatches(batches);
      },
      [&](LanceAsyncCallback callback, auto* context) {
        return fragment_reader_->TakeAsync(converted, &context->stream, callback, context);
      });
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

folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::RecordBatchReader>>> LanceTableReader::read_with_range_async(
    uint64_t start_offset, uint64_t end_offset) {
  assert(fragment_reader_);
  if (start_offset > end_offset || end_offset > std::numeric_limits<uint32_t>::max()) {
    return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::RecordBatchReader>>(
        arrow::Status::Invalid("Lance row range must be ordered and fit in uint32")));
  }
  // Empty ranges use the Rust projection schema too, preserving the same
  // schema-level metadata as nonempty ranges without storing another schema.
  const auto batch_size = std::max<uint64_t>(
      1, std::min<uint64_t>(end_offset - start_offset + num_deletions_, std::numeric_limits<uint32_t>::max()));
  // Rust owns the fragment reader and materializes all batches before completing
  // this future, so the caller can drain the result without waiting for I/O.
  return submit_lance_async<std::shared_ptr<arrow::RecordBatchReader>>(
      "Failed to read Lance fragment range",
      [](uint64_t handle, ArrowArrayStream&) -> arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> {
        ArrowArrayStream stream{};
        ArrowCDataReleaseGuard stream_guard(&stream);
        BlockingFragmentReader::TakeRecordBatchStream(handle, stream);
        ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ImportRecordBatchReader(&stream));
        return internal::WrapLanceRecordBatchReader(std::move(reader));
      },
      [&](LanceAsyncCallback callback, void* context) {
        return fragment_reader_->ReadRangesAsync(static_cast<uint32_t>(start_offset), static_cast<uint32_t>(end_offset),
                                                 static_cast<uint32_t>(batch_size), callback, context);
      });
}

arrow::Result<std::shared_ptr<FormatReader>> LanceTableReader::clone_reader() {
  assert(fragment_reader_);  // already opened
  return this->shared_from_this();
}

}  // namespace milvus_storage::lance
