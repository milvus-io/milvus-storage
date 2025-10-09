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

#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/common/metadata.h"
#include "parquet/arrow/reader.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/filesystem/fs.h"
#include "bridgeimpl.hpp"  // from cpp/src/format/vortex/vx-bridge/src/include

namespace milvus_storage::vortex {

class VortexFormatReader final {
  public:
  VortexFormatReader(const ObjectStoreWrapper& obsw_ref_,
                     const std::string& path,
                     std::vector<std::string> needed_columns);

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> readall();

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(const std::vector<int64_t>& row_indices);

  inline size_t rows() const { return vxfile_.RowCount(); }

  private:
  const ObjectStoreWrapper& obsw_ref_;
  VortexFile vxfile_;

  std::vector<std::string> proj_cols_;
};

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE