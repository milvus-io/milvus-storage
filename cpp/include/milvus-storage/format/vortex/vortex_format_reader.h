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
#ifdef BUILD_VORTEX_BRIDGE
#pragma once

#include <arrow/chunked_array.h>

#include "bridgeimpl.hpp"  // from cpp/src/format/vortex/vx-bridge/src/include

#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::vortex {

class VortexFormatReader final {
  public:
  VortexFormatReader(const ObjectStoreWrapper& obsw_ref_,
                     const std::shared_ptr<arrow::Schema>& schema,
                     const std::string& path,
                     std::vector<std::string> needed_columns);

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::ChunkedArray>> read(uint64_t row_start, uint64_t row_end);

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices);

  // get the row ranges(splits) of the file
  inline std::vector<uint64_t> row_ranges() const { return vxfile_.Splits(); }

  // get the total rows of the file
  inline size_t rows() const { return vxfile_.RowCount(); }

  // get the total memory usage(uncompressed memory) of the file
  uint64_t total_mem_usage();

  // get the memory usage(uncompressed memory) of the column
  uint64_t mem_usage(size_t idx_in_column_group);

  private:
  const ObjectStoreWrapper& obsw_ref_;
  VortexFile vxfile_;

  std::vector<std::string> proj_cols_;
  std::shared_ptr<arrow::Schema> schema_;
};

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE