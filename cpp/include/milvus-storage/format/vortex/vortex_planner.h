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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <arrow/result.h>
#include <arrow/status.h>

#include "milvus-storage/format/vortex/vortex_types.h"

namespace milvus_storage::vortex {

class VortexFooterReader;

struct VortexReadPlan {
  struct RangeScan {
    std::vector<RowRange> ranges;
  };

  struct Take {
    std::vector<int64_t> row_indices;
    std::vector<RowRange> ranges;
  };

  std::variant<RangeScan, Take> op = RangeScan{};
  std::string predicate;
  bool apply_predicate = true;
};

struct VortexPlan {
  std::vector<uint64_t> cell_ids;
  VortexReadPlan read_plan;
};

// Build local-format cache cell metadata from Vortex footer metadata. Planner
// and translator should be initialized from the same cell metas so planning and
// cache loading use identical physical-unit boundaries.
arrow::Result<VortexCellMetasPtr> BuildVortexCellMetas(const std::shared_ptr<VortexFooterReader>& footer_reader,
                                                       const std::string& field_name);

arrow::Result<VortexCellMetasPtr> BuildVortexGroupCellMetas(const std::shared_ptr<VortexFooterReader>& footer_reader,
                                                            const std::vector<std::string>& field_names);

class VortexPlanner {
  public:
  static arrow::Result<std::shared_ptr<VortexPlanner>> Make(const std::shared_ptr<VortexFooterReader>& footer_reader,
                                                            std::string field_name,
                                                            VortexCellMetasPtr cell_metas);

  static arrow::Result<std::shared_ptr<VortexPlanner>> MakeGroup(
      const std::shared_ptr<VortexFooterReader>& footer_reader, VortexCellMetasPtr cell_metas);

  const std::string& field_name() const { return field_name_; }

  uint64_t rows() const { return rows_; }

  uint64_t memory_bytes() const { return memory_bytes_; }

  size_t num_cells() const { return cell_metas_ ? cell_metas_->size() : 0; }

  const VortexCellMetas& cell_metas() const { return *cell_metas_; }

  arrow::Result<VortexPlan> PlanForRowRange(uint64_t row_start,
                                            uint64_t row_end,
                                            const std::string& predicate = "") const;

  arrow::Result<VortexPlan> PlanForOffsets(const std::vector<int64_t>& offsets) const;

  private:
  VortexPlanner(std::shared_ptr<VortexFooterReader> footer_reader,
                std::string field_name,
                uint64_t rows,
                VortexCellMetasPtr cell_metas,
                uint64_t memory_bytes);

  arrow::Result<std::vector<uint64_t>> SelectCellIdsForRowRange(uint64_t row_start, uint64_t row_end) const;

  arrow::Result<std::vector<uint64_t>> SelectCellIdsForOffsets(const std::vector<int64_t>& offsets) const;

  arrow::Result<std::vector<RowRange>> BuildReadRangesForCells(const std::vector<uint64_t>& cell_ids,
                                                               uint64_t row_start,
                                                               uint64_t row_end) const;

  arrow::Result<std::vector<uint64_t>> PruneCellsByPredicate(const std::vector<uint64_t>& cell_ids,
                                                             const std::string& predicate) const;

  std::shared_ptr<VortexFooterReader> footer_reader_;
  std::string field_name_;
  uint64_t rows_ = 0;
  VortexCellMetasPtr cell_metas_;
  uint64_t memory_bytes_ = 0;
};

}  // namespace milvus_storage::vortex
