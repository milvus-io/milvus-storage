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

#include "paimon_bridge.h"

#include <cerrno>
#include <string_view>

#include <arrow/util/io_util.h>

#include "bridge_util.h"
#include "milvus-storage/common/extend_status.h"
#include "rust/cxx.h"
#include "rust-bridge/lib.h"

namespace milvus_storage::paimon {
namespace {

constexpr std::string_view kInvalidMarker = "[paimon:error=invalid]";
constexpr std::string_view kNotImplementedMarker = "[paimon:error=not-implemented]";
constexpr std::string_view kNotFoundMarker = "[paimon:error=not-found]";
constexpr std::string_view kTransientThrottlingMarker = "[paimon:error=transient-throttling]";
constexpr std::string_view kTransientServiceMarker = "[paimon:error=transient-service]";

std::string StripMarker(std::string_view message, std::string_view marker) {
  auto position = message.find(marker);
  if (position == std::string_view::npos) {
    return std::string(message);
  }
  auto suffix = position + marker.size();
  if (suffix < message.size() && message[suffix] == ' ') {
    ++suffix;
  }
  std::string result;
  result.reserve(message.size() - marker.size());
  result.append(message.substr(0, position));
  result.append(message.substr(suffix));
  return result.empty() ? "Unknown Paimon error" : result;
}

template <typename T, typename Fn>
arrow::Result<T> CatchRustResult(Fn&& fn) {
  try {
    return fn();
  } catch (const rust::cxxbridge1::Error& error) {
    return MakePaimonBridgeErrorStatus(error.what());
  }
}

}  // namespace

arrow::Status MakePaimonBridgeErrorStatus(std::string_view message) {
  if (message.find(kInvalidMarker) != std::string_view::npos) {
    return arrow::Status::Invalid(StripMarker(message, kInvalidMarker));
  }
  if (message.find(kNotImplementedMarker) != std::string_view::npos) {
    return arrow::Status::NotImplemented(StripMarker(message, kNotImplementedMarker));
  }
  if (message.find(kNotFoundMarker) != std::string_view::npos) {
    return arrow::Status::IOError(StripMarker(message, kNotFoundMarker))
        .WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
  }
  if (message.find(kTransientThrottlingMarker) != std::string_view::npos) {
    auto error = StripMarker(message, kTransientThrottlingMarker);
    return MakeExtendError(ExtendStatusCode::StorageTransientThrottling, error, error);
  }
  if (message.find(kTransientServiceMarker) != std::string_view::npos) {
    auto error = StripMarker(message, kTransientServiceMarker);
    return MakeExtendError(ExtendStatusCode::StorageTransientService, error, error);
  }
  return arrow::Status::IOError(message);
}

arrow::Result<std::vector<PaimonFileInfo>> PlanFiles(const std::string& table_location,
                                                     int64_t snapshot_id,
                                                     const std::string& scan_mode,
                                                     const StorageOptions& storage_options) {
  return CatchRustResult<std::vector<PaimonFileInfo>>([&]() {
    rust::Vec<rust::String> keys;
    rust::Vec<rust::String> values;
    ConvertStorageOptions(storage_options, keys, values);
    auto planned = ffi::paimon_plan_files(rust::Str(table_location), snapshot_id, rust::Str(scan_mode), std::move(keys),
                                          std::move(values));
    std::vector<PaimonFileInfo> result;
    result.reserve(planned.size());
    for (const auto& info : planned) {
      result.push_back(PaimonFileInfo{std::string(info.path), info.file_size, std::string(info.metadata_json)});
    }
    return result;
  });
}

arrow::Result<std::vector<uint64_t>> ReadDeletionVector(const std::string& path,
                                                        uint64_t offset,
                                                        uint64_t length,
                                                        int64_t expected_cardinality,
                                                        const StorageOptions& storage_options) {
  return CatchRustResult<std::vector<uint64_t>>([&]() {
    rust::Vec<rust::String> keys;
    rust::Vec<rust::String> values;
    ConvertStorageOptions(storage_options, keys, values);
    auto positions = ffi::paimon_read_deletion_vector(rust::Str(path), offset, length, expected_cardinality,
                                                      std::move(keys), std::move(values));
    return std::vector<uint64_t>(positions.begin(), positions.end());
  });
}

arrow::Result<int64_t> CreateTestTable(const std::string& table_location,
                                       uint64_t num_rows,
                                       const std::string& mode,
                                       const std::vector<int64_t>& deleted_positions,
                                       const std::string& file_format,
                                       uint32_t dimension) {
  return CatchRustResult<int64_t>([&]() {
    rust::Vec<int64_t> positions;
    positions.reserve(deleted_positions.size());
    for (auto position : deleted_positions) {
      positions.push_back(position);
    }
    return ffi::paimon_create_test_table(rust::Str(table_location), num_rows, rust::Str(mode), std::move(positions),
                                         rust::Str(file_format), dimension);
  });
}

}  // namespace milvus_storage::paimon
