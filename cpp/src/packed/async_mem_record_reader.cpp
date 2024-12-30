// Copyright 2024 Zilliz
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

#include "packed/async_mem_record_reader.h"
#include <parquet/properties.h>
#include <memory>
#include <limits>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <arrow/util/logging.h>
#include "common/log.h"
#include "common/arrow_util.h"
#include "arrow/util/future.h"
#include <arrow/util/thread_pool.h>

namespace milvus_storage {

AsyncMemRecordBatchReader::AsyncMemRecordBatchReader(arrow::fs::FileSystem& fs,
                                                     const std::string& path,
                                                     const std::shared_ptr<arrow::Schema>& schema,
                                                     int64_t total_buffer_size)
    : fs_(fs),
      path_(path),
      schema_(schema),
      total_buffer_size_(total_buffer_size),
      cpu_thread_pool_size_(arrow::internal::GetCpuThreadPool()->GetCapacity()) {
  auto result = MakeArrowFileReader(fs, path);
  if (!result.ok()) {
    LOG_STORAGE_ERROR_ << "Error making file reader:" << result.status().ToString();
    throw std::runtime_error(result.status().ToString());
  }
  auto file_reader = std::move(result.value());
  total_row_groups_ = file_reader->parquet_reader()->metadata()->num_row_groups();
}

std::vector<std::pair<size_t, size_t>> AsyncMemRecordBatchReader::CreateBatches(size_t batch_size) const {
  std::vector<std::pair<size_t, size_t>> batches;
  size_t remaining = total_row_groups_;
  size_t offset = 0;

  while (remaining > 0) {
    size_t current_batch_size = std::min(batch_size, remaining);
    batches.emplace_back(offset, current_batch_size);
    offset += current_batch_size;
    remaining -= current_batch_size;
  }
  return batches;
}

arrow::Status AsyncMemRecordBatchReader::Execute() {
  size_t batch_size = cpu_thread_pool_size_;
  int64_t reader_buffer_size = total_buffer_size_ / cpu_thread_pool_size_;

  auto batches = CreateBatches(batch_size);
  results_.resize(batches.size());
  std::vector<arrow::Future<>> futures;

  for (size_t reader_index = 0; reader_index < batches.size(); ++reader_index) {
    const auto& [offset, num_row_groups] = batches[reader_index];

    auto reader =
        std::make_shared<MemRecordBatchReader>(fs_, path_, schema_, offset, num_row_groups, reader_buffer_size);
    readers_.push_back(reader);

    auto submit_result = arrow::internal::GetCpuThreadPool()->Submit([this, reader, reader_index]() -> arrow::Status {
      // Collect RecordBatches
      std::shared_ptr<arrow::RecordBatch> batch;
      while (true) {
        ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));
        if (!batch) {
          break;  // Finished reading
        }
        results_[reader_index].push_back(std::move(batch));
      }
      return arrow::Status::OK();
    });

    if (submit_result.ok()) {
      futures.push_back(submit_result.ValueOrDie());
    } else {
      return submit_result.status();  // Return error if Submit failed
    }
  }

  arrow::AllFinished(futures).Wait();
  return arrow::Status::OK();
}

const std::vector<std::shared_ptr<MemRecordBatchReader>> AsyncMemRecordBatchReader::Readers() { return readers_; }

const std::shared_ptr<arrow::Table> AsyncMemRecordBatchReader::Table() {
  std::vector<std::shared_ptr<arrow::RecordBatch>> all_batches;
  for (const auto& reader_batches : results_) {
    for (const auto& batch : reader_batches) {
      all_batches.push_back(batch);
    }
  }
  return arrow::Table::FromRecordBatches(all_batches).ValueOrDie();
}

}  // namespace milvus_storage
