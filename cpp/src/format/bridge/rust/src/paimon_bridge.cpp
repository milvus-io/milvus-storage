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
#include "bridge_error.h"

#include <cerrno>
#include <string_view>
#include <utility>

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/util/io_util.h>

#include "bridge_util.h"
#include "milvus-storage/common/extend_status.h"
#include "rust/cxx.h"
#include "rust-bridge/lib.h"

namespace milvus_storage::paimon {
// The classified-error side channel is shared with every bridge (see
// bridge_error.h); paimon reads the same slot through the bridge namespace.
namespace bridge_ffi = ::milvus_storage::bridge::ffi;
namespace {

template <typename T, typename Fn>
arrow::Result<T> CatchRustResult(Fn&& fn) {
  milvus_storage::bridge::ClearBridgeErrorChannel();
  try {
    return fn();
  } catch (const rust::cxxbridge1::Error& error) {
    // The typed side channel wins when the Rust side recorded a code; markers
    // stay only as the fallback for messages that were never published.
    auto info = bridge_ffi::take_last_bridge_error();
    if (info.code != 0) {
      return milvus_storage::bridge::MakeBridgeErrorStatus(
          info.code,
          info.message.size() != 0 ? std::string_view(info.message.data(), info.message.size())
                                   : std::string_view(error.what()));
    }
        // No slot code: fall through to THE shared decoder, which reads the
    // universal transport tag the Rust side embeds and otherwise degrades to
    // the conservative non-retriable IOError.
    return milvus_storage::bridge::MakeBridgeErrorStatus(error.what());
  }
}

arrow::Status TranslatePaimonStreamStatus(arrow::Status status) {
  if (status.ok()) {
    return status;
  }
  // Stream errors surface as strings through the Arrow C ABI; the Rust side
  // embeds the same universal transport tag every other bridge uses.
  if (auto decoded = milvus_storage::bridge::DecodeBridgeErrorStatus(status.message())) {
    return *decoded;
  }
  return status;
}

class PaimonStreamReader final : public arrow::RecordBatchReader {
  public:
  PaimonStreamReader(std::shared_ptr<arrow::RecordBatchReader> inner, std::shared_ptr<arrow::Schema> output_schema)
      : inner_(std::move(inner)), output_schema_(std::move(output_schema)) {}

  std::shared_ptr<arrow::Schema> schema() const override { return output_schema_; }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
    ARROW_RETURN_NOT_OK(TranslatePaimonStreamStatus(inner_->ReadNext(batch)));
    if (!*batch) {
      return arrow::Status::OK();
    }
    if ((*batch)->num_columns() != output_schema_->num_fields()) {
      return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated, "Paimon data-split stream returned an unexpected column count");
    }
    for (int index = 0; index < output_schema_->num_fields(); ++index) {
      const auto& actual = (*batch)->schema()->field(index);
      const auto& expected = output_schema_->field(index);
      if (actual->name() != expected->name() || !actual->type()->Equals(expected->type())) {
        return arrow::Status::Invalid("Paimon data-split stream schema mismatch at column ", index, ": expected ",
                                      expected->ToString(), ", got ", actual->ToString());
      }
      if (!expected->nullable() && (*batch)->column(index)->null_count() != 0) {
        return arrow::Status::Invalid("Paimon data-split stream returned nulls for non-nullable column: ",
                                      expected->name());
      }
    }
    *batch = arrow::RecordBatch::Make(output_schema_, (*batch)->num_rows(), (*batch)->columns());
    return arrow::Status::OK();
  }

  arrow::Status Close() override { return TranslatePaimonStreamStatus(inner_->Close()); }

  private:
  std::shared_ptr<arrow::RecordBatchReader> inner_;
  std::shared_ptr<arrow::Schema> output_schema_;
};

}  // namespace


namespace internal {
std::shared_ptr<arrow::RecordBatchReader> WrapPaimonRecordBatchReader(std::shared_ptr<arrow::RecordBatchReader> inner,
                                                                      std::shared_ptr<arrow::Schema> output_schema) {
  return std::make_shared<PaimonStreamReader>(std::move(inner), std::move(output_schema));
}
}  // namespace internal

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

arrow::Result<PaimonTestTableInfo> CreateTestTableInfo(const std::string& table_location,
                                                       uint64_t num_rows,
                                                       const std::string& mode,
                                                       const std::vector<int64_t>& deleted_positions,
                                                       const StorageOptions& storage_options,
                                                       const std::string& file_format,
                                                       uint32_t dimension) {
  return CatchRustResult<PaimonTestTableInfo>([&]() {
    rust::Vec<int64_t> positions;
    positions.reserve(deleted_positions.size());
    for (auto position : deleted_positions) {
      positions.push_back(position);
    }
    rust::Vec<rust::String> keys;
    rust::Vec<rust::String> values;
    ConvertStorageOptions(storage_options, keys, values);
    auto info =
        ffi::paimon_create_test_table(rust::Str(table_location), num_rows, rust::Str(mode), std::move(positions),
                                      std::move(keys), std::move(values), rust::Str(file_format), dimension);
    return PaimonTestTableInfo{{info.snapshot_ids.begin(), info.snapshot_ids.end()}};
  });
}

arrow::Result<int64_t> CreateTestTable(const std::string& table_location,
                                       uint64_t num_rows,
                                       const std::string& mode,
                                       const std::vector<int64_t>& deleted_positions,
                                       const std::string& file_format,
                                       uint32_t dimension) {
  ARROW_ASSIGN_OR_RAISE(
      auto info, CreateTestTableInfo(table_location, num_rows, mode, deleted_positions, {}, file_format, dimension));
  if (info.snapshot_ids.empty()) {
    return arrow::Status::Invalid("Paimon test table has no committed snapshot");
  }
  return info.snapshot_ids.back();
}

arrow::Result<std::shared_ptr<BlockingPaimonDataSplitReader>> BlockingPaimonDataSplitReader::Open(
    const std::string& metadata_json,
    const std::string& expected_table_location,
    const StorageOptions& storage_options) {
  return CatchRustResult<std::shared_ptr<BlockingPaimonDataSplitReader>>([&]() {
    rust::Vec<rust::String> keys;
    rust::Vec<rust::String> values;
    ConvertStorageOptions(storage_options, keys, values);
    return std::make_shared<BlockingPaimonDataSplitReader>(ffi::paimon_open_data_split_reader(
        rust::Str(metadata_json), rust::Str(expected_table_location), std::move(keys), std::move(values)));
  });
}

arrow::Status BlockingPaimonDataSplitReader::ExportSchema(ArrowSchema* schema) const {
  if (schema == nullptr) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated, "cannot export Paimon schema into a null pointer");
  }
  try {
    impl_->export_schema(reinterpret_cast<uint8_t*>(schema));
    return arrow::Status::OK();
  } catch (const rust::cxxbridge1::Error& error) {
        // No slot code: fall through to THE shared decoder, which reads the
    // universal transport tag the Rust side embeds and otherwise degrades to
    // the conservative non-retriable IOError.
    return milvus_storage::bridge::MakeBridgeErrorStatus(error.what());
  }
}

arrow::Result<ArrowArrayStream> BlockingPaimonDataSplitReader::OpenStream(
    const std::vector<std::string>& projected_columns) const {
  return CatchRustResult<ArrowArrayStream>([&]() {
    rust::Vec<rust::String> columns;
    columns.reserve(projected_columns.size());
    for (const auto& column : projected_columns) {
      columns.push_back(rust::String(column));
    }
    ArrowArrayStream stream{};
    impl_->open_stream(std::move(columns), reinterpret_cast<uint8_t*>(&stream));
    return stream;
  });
}

}  // namespace milvus_storage::paimon
