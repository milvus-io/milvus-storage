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

#include "milvus-storage/format/vortex/vortex_planner.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <unordered_set>
#include <utility>

#include <fmt/format.h>

#include "milvus-storage/common/log.h"
#include "milvus-storage/format/vortex/vortex_field_layout_internal.h"
#include "milvus-storage/format/vortex/vortex_footer_reader.h"
#include "milvus-storage/format/vortex/vortex_types.h"

namespace milvus_storage::vortex {

namespace {

static arrow::Status CheckedAdd(uint64_t* total, uint64_t value, const std::string& key) {
  if (*total > std::numeric_limits<uint64_t>::max() - value) {
    return arrow::Status::Invalid(fmt::format("Vortex field {} byte size overflows", key));
  }
  *total += value;
  return arrow::Status::OK();
}

static arrow::Result<uint64_t> SumFlatSegmentBytes(const std::vector<VortexFlatSegmentRange>& ranges,
                                                   const std::string& key) {
  uint64_t bytes = 0;
  for (const auto& range : ranges) {
    ARROW_RETURN_NOT_OK(CheckedAdd(&bytes, range.byte_range.length, key));
  }
  return bytes;
}

static arrow::Result<uint64_t> SumByteRanges(const std::vector<ByteRange>& ranges, const std::string& key) {
  uint64_t bytes = 0;
  for (const auto& range : ranges) {
    ARROW_RETURN_NOT_OK(CheckedAdd(&bytes, range.length, key));
  }
  return bytes;
}

static arrow::Result<uint64_t> SumCellBytes(const std::vector<VortexCellMeta>& metas, const std::string& key) {
  uint64_t bytes = 0;
  for (const auto& meta : metas) {
    ARROW_RETURN_NOT_OK(CheckedAdd(&bytes, std::max<uint64_t>(meta.memory_bytes, meta.storage_bytes), key));
  }
  return bytes;
}

static arrow::Status AppendUniqueFlatSegments(VortexCellMeta* meta,
                                              const std::vector<VortexFlatSegmentRange>& flat_segments,
                                              std::unordered_set<uint64_t>* seen_flat_segments) {
  if (meta == nullptr || seen_flat_segments == nullptr) {
    return arrow::Status::Invalid("Vortex group cell merge got null state");
  }
  for (const auto& segment : flat_segments) {
    if (!seen_flat_segments->emplace(segment.flat_segment_id).second) {
      continue;
    }
    meta->flat_segment_ids.emplace_back(segment.flat_segment_id);
    meta->flat_segment_ranges.emplace_back(segment.byte_range);
  }
  return arrow::Status::OK();
}

static uint64_t PhysicalUnitIndex(const VortexFlatUnit& unit) { return unit.flat_id; }

static uint64_t PhysicalUnitIndex(const VortexRowGroupUnit& unit) { return unit.row_group_id; }

static arrow::Result<std::vector<VortexCellMeta>> BuildGroupCellMetas(std::vector<VortexFieldLayout> layouts,
                                                                      const std::vector<std::string>& field_names,
                                                                      const std::string& key) {
  if (field_names.empty()) {
    return arrow::Status::Invalid("Vortex group planner requires at least one field");
  }
  if (layouts.size() != field_names.size()) {
    return arrow::Status::Invalid(
        fmt::format("Vortex group {} has {} layouts, expected {}", key, layouts.size(), field_names.size()));
  }

  auto& first_layout = layouts.front();
  std::vector<VortexCellMeta> metas;
  std::vector<std::unordered_set<uint64_t>> seen_flat_segments;

  const auto init_meta = [&](const auto& unit) {
    return VortexCellMeta{
        .granularity = first_layout.granularity,
        .physical_unit_index = PhysicalUnitIndex(unit),
        .row_offset = unit.row_offset,
        .row_count = unit.row_count,
    };
  };

  if (first_layout.granularity == VortexPhysicalGranularity::kFlat) {
    metas.reserve(first_layout.flats.size());
    seen_flat_segments.resize(first_layout.flats.size());
    for (auto& flat : first_layout.flats) {
      auto meta = init_meta(flat);
      ARROW_RETURN_NOT_OK(AppendUniqueFlatSegments(&meta, flat.flat_segments, &seen_flat_segments[metas.size()]));
      metas.emplace_back(std::move(meta));
    }
  } else if (first_layout.granularity == VortexPhysicalGranularity::kRowGroup) {
    metas.reserve(first_layout.row_groups.size());
    seen_flat_segments.resize(first_layout.row_groups.size());
    for (auto& row_group : first_layout.row_groups) {
      auto meta = init_meta(row_group);
      ARROW_RETURN_NOT_OK(AppendUniqueFlatSegments(&meta, row_group.flat_segments, &seen_flat_segments[metas.size()]));
      metas.emplace_back(std::move(meta));
    }
  } else {
    return arrow::Status::Invalid(fmt::format("Unsupported vortex group granularity for {}", key));
  }

  const auto merge_units = [&](const auto& units, const std::string& field_name) -> arrow::Status {
    if (units.size() != metas.size()) {
      return arrow::Status::Invalid(fmt::format("Vortex group {} field {} has {} units, expected {}", key, field_name,
                                                units.size(), metas.size()));
    }
    for (size_t i = 0; i < units.size(); ++i) {
      const auto& unit = units[i];
      auto& meta = metas[i];
      if (unit.row_offset != meta.row_offset || unit.row_count != meta.row_count) {
        return arrow::Status::Invalid(fmt::format(
            "Vortex group {} field {} unit {} row range [{}, {}) does not match group [{}, {})", key, field_name, i,
            unit.row_offset, unit.row_offset + unit.row_count, meta.row_offset, meta.row_offset + meta.row_count));
      }
      if (first_layout.granularity == VortexPhysicalGranularity::kRowGroup &&
          PhysicalUnitIndex(unit) != meta.physical_unit_index) {
        return arrow::Status::Invalid(
            fmt::format("Vortex group {} field {} unit {} physical index {} does not match group {}", key, field_name,
                        i, PhysicalUnitIndex(unit), meta.physical_unit_index));
      }
      ARROW_RETURN_NOT_OK(AppendUniqueFlatSegments(&meta, unit.flat_segments, &seen_flat_segments[i]));
    }
    return arrow::Status::OK();
  };

  for (size_t field_idx = 1; field_idx < field_names.size(); ++field_idx) {
    const auto& field_name = field_names[field_idx];
    auto& layout = layouts[field_idx];
    if (layout.granularity != first_layout.granularity) {
      return arrow::Status::Invalid(
          fmt::format("Vortex group {} field {} granularity does not match group", key, field_name));
    }
    if (layout.granularity == VortexPhysicalGranularity::kFlat) {
      ARROW_RETURN_NOT_OK(merge_units(layout.flats, field_name));
    } else if (layout.granularity == VortexPhysicalGranularity::kRowGroup) {
      ARROW_RETURN_NOT_OK(merge_units(layout.row_groups, field_name));
    } else {
      return arrow::Status::Invalid(
          fmt::format("Unsupported vortex group granularity for {} field {}", key, field_name));
    }
  }

  for (auto& meta : metas) {
    ARROW_ASSIGN_OR_RAISE(meta.storage_bytes, SumByteRanges(meta.flat_segment_ranges, key));
    meta.memory_bytes = meta.storage_bytes;
  }
  return metas;
}

static size_t FindCellIndexForRow(const std::vector<VortexCellMeta>& metas, uint64_t row) {
  auto it = std::upper_bound(metas.begin(), metas.end(), row,
                             [](uint64_t value, const VortexCellMeta& meta) { return value < meta.row_offset; });
  if (it == metas.begin()) {
    return 0;
  }
  return static_cast<size_t>(std::distance(metas.begin(), std::prev(it)));
}

static arrow::Result<std::vector<VortexCellMeta>> BuildCellMetasFromFlats(std::vector<VortexFlatUnit> flats,
                                                                          uint64_t total_rows,
                                                                          const std::string& key);

static arrow::Result<std::vector<VortexCellMeta>> BuildCellMetasFromRowGroups(
    std::vector<VortexRowGroupUnit> row_groups, uint64_t total_rows, const std::string& key);

static arrow::Status ValidateCellMetas(std::vector<VortexCellMeta>* metas, uint64_t total_rows, const std::string& key);

static arrow::Status ValidateCellMetasReady(const VortexCellMetasPtr& metas,
                                            uint64_t total_rows,
                                            const std::string& key);

static arrow::Status ValidateSortedUniqueOffsets(const std::vector<int64_t>& offsets, uint64_t total_rows) {
  for (size_t i = 0; i < offsets.size(); ++i) {
    const auto offset = offsets[i];
    if (offset < 0 || static_cast<uint64_t>(offset) >= total_rows) {
      return arrow::Status::Invalid(fmt::format("Vortex take offset {} out of file rows {}", offset, total_rows));
    }
    if (i > 0 && offset <= offsets[i - 1]) {
      return arrow::Status::Invalid(
          fmt::format("Vortex take offsets must be sorted and unique, index={}, previous={}, current={}", i,
                      offsets[i - 1], offset));
    }
  }
  return arrow::Status::OK();
}

}  // namespace

arrow::Result<VortexCellMetasPtr> BuildVortexCellMetas(const std::shared_ptr<VortexFooterReader>& footer_reader,
                                                       const std::string& field_name) {
  if (!footer_reader) {
    return arrow::Status::Invalid("BuildVortexCellMetas requires a non-null footer reader");
  }
  const auto key = fmt::format("{}:{}", footer_reader->path(), field_name);
  ARROW_ASSIGN_OR_RAISE(auto layout, footer_reader->GetFieldLayout(field_name));

  std::vector<VortexCellMeta> cell_metas;
  if (layout.granularity == VortexPhysicalGranularity::kFlat) {
    ARROW_ASSIGN_OR_RAISE(cell_metas, BuildCellMetasFromFlats(std::move(layout.flats), footer_reader->rows(), key));
  } else if (layout.granularity == VortexPhysicalGranularity::kRowGroup) {
    ARROW_ASSIGN_OR_RAISE(cell_metas,
                          BuildCellMetasFromRowGroups(std::move(layout.row_groups), footer_reader->rows(), key));
  } else {
    return arrow::Status::Invalid(fmt::format("Unsupported vortex field granularity for {}", key));
  }
  return std::make_shared<const VortexCellMetas>(std::move(cell_metas));
}

arrow::Result<VortexCellMetasPtr> BuildVortexGroupCellMetas(const std::shared_ptr<VortexFooterReader>& footer_reader,
                                                            const std::vector<std::string>& field_names) {
  if (!footer_reader) {
    return arrow::Status::Invalid("BuildVortexGroupCellMetas requires a non-null footer reader");
  }
  if (field_names.empty()) {
    return arrow::Status::Invalid("Vortex group cell meta builder requires at least one field");
  }
  const auto key = fmt::format("{}:<column_group>", footer_reader->path());
  std::vector<VortexFieldLayout> layouts;
  layouts.reserve(field_names.size());
  for (const auto& field_name : field_names) {
    ARROW_ASSIGN_OR_RAISE(auto layout, footer_reader->GetFieldLayout(field_name));
    layouts.emplace_back(std::move(layout));
  }
  ARROW_ASSIGN_OR_RAISE(auto cell_metas, BuildGroupCellMetas(std::move(layouts), field_names, key));
  ARROW_RETURN_NOT_OK(ValidateCellMetas(&cell_metas, footer_reader->rows(), key));
  return std::make_shared<const VortexCellMetas>(std::move(cell_metas));
}

arrow::Result<std::shared_ptr<VortexPlanner>> VortexPlanner::Make(
    const std::shared_ptr<VortexFooterReader>& footer_reader, std::string field_name, VortexCellMetasPtr cell_metas) {
  if (!footer_reader) {
    return arrow::Status::Invalid("VortexPlanner requires a non-null footer reader");
  }
  const auto key = fmt::format("{}:{}", footer_reader->path(), field_name);
  ARROW_RETURN_NOT_OK(ValidateCellMetasReady(cell_metas, footer_reader->rows(), key));
  ARROW_ASSIGN_OR_RAISE(auto memory_bytes, SumCellBytes(*cell_metas, key));
  return std::shared_ptr<VortexPlanner>(new VortexPlanner(footer_reader, std::move(field_name), footer_reader->rows(),
                                                          std::move(cell_metas), memory_bytes));
}

arrow::Result<std::shared_ptr<VortexPlanner>> VortexPlanner::MakeGroup(
    const std::shared_ptr<VortexFooterReader>& footer_reader, VortexCellMetasPtr cell_metas) {
  if (!footer_reader) {
    return arrow::Status::Invalid("VortexPlanner requires a non-null footer reader");
  }
  const auto key = fmt::format("{}:<column_group>", footer_reader->path());
  ARROW_RETURN_NOT_OK(ValidateCellMetasReady(cell_metas, footer_reader->rows(), key));
  ARROW_ASSIGN_OR_RAISE(auto memory_bytes, SumCellBytes(*cell_metas, key));
  return std::shared_ptr<VortexPlanner>(
      new VortexPlanner(footer_reader, "<column_group>", footer_reader->rows(), std::move(cell_metas), memory_bytes));
}

VortexPlanner::VortexPlanner(std::shared_ptr<VortexFooterReader> footer_reader,
                             std::string field_name,
                             uint64_t rows,
                             VortexCellMetasPtr cell_metas,
                             uint64_t memory_bytes)
    : footer_reader_(std::move(footer_reader)),
      field_name_(std::move(field_name)),
      rows_(rows),
      cell_metas_(std::move(cell_metas)),
      memory_bytes_(memory_bytes) {}

arrow::Result<std::vector<uint64_t>> VortexPlanner::SelectCellIdsForRowRange(uint64_t row_start,
                                                                             uint64_t row_end) const {
  if (row_start > row_end || row_end > rows_) {
    return arrow::Status::Invalid(
        fmt::format("Vortex row range [{}, {}) is out of rows {}", row_start, row_end, rows_));
  }

  std::vector<uint64_t> cell_ids;
  if (row_start == row_end) {
    return cell_ids;
  }

  const auto& metas = cell_metas();
  for (auto cell_idx = FindCellIndexForRow(metas, row_start); cell_idx < metas.size(); ++cell_idx) {
    const auto& meta = metas[cell_idx];
    const auto cell_start = meta.row_offset;
    const auto cell_end = meta.row_offset + meta.row_count;
    if (cell_start >= row_end) {
      break;
    }
    if (cell_start < row_end && cell_end > row_start) {
      cell_ids.emplace_back(meta.cell_id);
    }
  }
  if (cell_ids.empty()) {
    return arrow::Status::Invalid(fmt::format("Vortex row range [{}, {}) has no resident cells", row_start, row_end));
  }
  return cell_ids;
}

arrow::Result<VortexPlan> VortexPlanner::PlanForRowRange(uint64_t row_start,
                                                         uint64_t row_end,
                                                         const std::string& predicate) const {
  ARROW_ASSIGN_OR_RAISE(auto candidate_cell_ids, SelectCellIdsForRowRange(row_start, row_end));
  ARROW_ASSIGN_OR_RAISE(auto cell_ids, PruneCellsByPredicate(candidate_cell_ids, predicate));
  ARROW_ASSIGN_OR_RAISE(auto read_ranges, BuildReadRangesForCells(cell_ids, row_start, row_end));
  LOG_STORAGE_INFO_ << fmt::format(
      "[VortexPlanner] row range [{}, {}) -> candidate_cells={}/{} filter_pruned_cells={} kept_cells={} read_ranges={} "
      "predicate={}",
      row_start, row_end, candidate_cell_ids.size(), num_cells(), candidate_cell_ids.size() - cell_ids.size(),
      cell_ids.size(), read_ranges.size(), predicate.empty() ? "<none>" : predicate);
  return VortexPlan{
      .cell_ids = std::move(cell_ids),
      .read_plan =
          VortexReadPlan{
              .op = VortexReadPlan::RangeScan{.ranges = std::move(read_ranges)},
              .predicate = predicate,
              .apply_predicate = true,
          },
  };
}

arrow::Result<std::vector<uint64_t>> VortexPlanner::SelectCellIdsForOffsets(const std::vector<int64_t>& offsets) const {
  std::vector<uint64_t> cell_ids;
  if (offsets.empty()) {
    return cell_ids;
  }

  ARROW_RETURN_NOT_OK(ValidateSortedUniqueOffsets(offsets, rows_));

  const auto& metas = cell_metas();
  size_t cell_idx = 0;
  for (auto offset : offsets) {
    const auto row = static_cast<uint64_t>(offset);
    while (cell_idx < metas.size() && metas[cell_idx].row_offset + metas[cell_idx].row_count <= row) {
      ++cell_idx;
    }
    if (cell_idx >= metas.size()) {
      return arrow::Status::Invalid(fmt::format("Vortex take offset {} has no resident cell", offset));
    }

    const auto& meta = metas[cell_idx];
    if (!(meta.row_offset <= row && row < meta.row_offset + meta.row_count)) {
      return arrow::Status::Invalid(fmt::format("Vortex take offset {} is outside selected cell [{}, {})", offset,
                                                meta.row_offset, meta.row_offset + meta.row_count));
    }
    if (cell_ids.empty() || cell_ids.back() != meta.cell_id) {
      cell_ids.emplace_back(meta.cell_id);
    }
  }
  return cell_ids;
}

arrow::Result<std::vector<RowRange>> VortexPlanner::BuildReadRangesForCells(const std::vector<uint64_t>& cell_ids,
                                                                            uint64_t row_start,
                                                                            uint64_t row_end) const {
  if (row_start > row_end || row_end > rows_) {
    return arrow::Status::Invalid(
        fmt::format("Vortex read range [{}, {}) is out of rows {}", row_start, row_end, rows_));
  }

  std::vector<RowRange> ranges;
  ranges.reserve(cell_ids.size());
  const auto& metas = cell_metas();
  for (auto cell_id : cell_ids) {
    if (cell_id >= metas.size()) {
      return arrow::Status::Invalid(fmt::format("Vortex cell id {} out of range {}", cell_id, metas.size()));
    }
    const auto& meta = metas[cell_id];
    const auto cell_start = meta.row_offset;
    const auto cell_end = meta.row_offset + meta.row_count;
    const auto start = std::max(cell_start, row_start);
    const auto end = std::min(cell_end, row_end);
    if (start >= end) {
      continue;
    }
    ranges.emplace_back(RowRange{.start = start, .end = end});
  }
  return ranges;
}

arrow::Result<std::vector<uint64_t>> VortexPlanner::PruneCellsByPredicate(const std::vector<uint64_t>& cell_ids,
                                                                          const std::string& predicate) const {
  if (predicate.empty() || cell_ids.empty()) {
    return cell_ids;
  }
  if (!footer_reader_) {
    return arrow::Status::Invalid("VortexPlanner requires footer reader for predicate pruning");
  }

  std::vector<uint64_t> candidate_row_group_ids;
  candidate_row_group_ids.reserve(cell_ids.size());
  const auto& metas = cell_metas();
  for (auto cell_id : cell_ids) {
    if (cell_id >= metas.size()) {
      return arrow::Status::Invalid(fmt::format("Vortex cell id {} out of range {}", cell_id, metas.size()));
    }
    const auto& meta = metas[cell_id];
    if (meta.granularity != VortexPhysicalGranularity::kRowGroup) {
      return cell_ids;
    }
    candidate_row_group_ids.emplace_back(meta.physical_unit_index);
  }

  ARROW_ASSIGN_OR_RAISE(auto kept_row_group_ids, footer_reader_->PruneRowGroups(predicate, candidate_row_group_ids));
  std::unordered_set<uint64_t> kept_row_group_set(kept_row_group_ids.begin(), kept_row_group_ids.end());

  std::vector<uint64_t> kept_cell_ids;
  kept_cell_ids.reserve(cell_ids.size());
  for (auto cell_id : cell_ids) {
    const auto& meta = metas[cell_id];
    if (kept_row_group_set.find(meta.physical_unit_index) != kept_row_group_set.end()) {
      kept_cell_ids.emplace_back(cell_id);
    }
  }
  return kept_cell_ids;
}

arrow::Result<VortexPlan> VortexPlanner::PlanForOffsets(const std::vector<int64_t>& offsets) const {
  ARROW_ASSIGN_OR_RAISE(auto cell_ids, SelectCellIdsForOffsets(offsets));
  ARROW_ASSIGN_OR_RAISE(auto read_ranges, BuildReadRangesForCells(cell_ids, 0, rows_));
  LOG_STORAGE_INFO_ << fmt::format("[VortexPlanner] offsets input={} -> cells {}/{} read_ranges={}", offsets.size(),
                                   cell_ids.size(), num_cells(), read_ranges.size());
  return VortexPlan{
      .cell_ids = std::move(cell_ids),
      .read_plan =
          VortexReadPlan{
              .op = VortexReadPlan::Take{.row_indices = offsets, .ranges = std::move(read_ranges)},
              .apply_predicate = false,
          },
  };
}

namespace {

static arrow::Result<std::vector<VortexCellMeta>> BuildCellMetasFromFlats(std::vector<VortexFlatUnit> flats,
                                                                          uint64_t total_rows,
                                                                          const std::string& key) {
  std::vector<VortexCellMeta> metas;
  metas.reserve(flats.size());
  for (auto& flat : flats) {
    VortexCellMeta meta{
        .granularity = VortexPhysicalGranularity::kFlat,
        .physical_unit_index = flat.flat_id,
        .row_offset = flat.row_offset,
        .row_count = flat.row_count,
    };
    ARROW_ASSIGN_OR_RAISE(meta.storage_bytes, SumFlatSegmentBytes(flat.flat_segments, key));
    meta.memory_bytes = meta.storage_bytes;
    meta.flat_segment_ids.reserve(flat.flat_segments.size());
    meta.flat_segment_ranges.reserve(flat.flat_segments.size());
    for (auto& segment : flat.flat_segments) {
      meta.flat_segment_ids.emplace_back(segment.flat_segment_id);
      meta.flat_segment_ranges.emplace_back(segment.byte_range);
    }
    metas.emplace_back(std::move(meta));
  }
  ARROW_RETURN_NOT_OK(ValidateCellMetas(&metas, total_rows, key));
  return metas;
}

static arrow::Result<std::vector<VortexCellMeta>> BuildCellMetasFromRowGroups(
    std::vector<VortexRowGroupUnit> row_groups, uint64_t total_rows, const std::string& key) {
  std::vector<VortexCellMeta> metas;
  metas.reserve(row_groups.size());
  for (auto& row_group : row_groups) {
    VortexCellMeta meta{
        .granularity = VortexPhysicalGranularity::kRowGroup,
        .physical_unit_index = row_group.row_group_id,
        .row_offset = row_group.row_offset,
        .row_count = row_group.row_count,
    };
    ARROW_ASSIGN_OR_RAISE(meta.storage_bytes, SumFlatSegmentBytes(row_group.flat_segments, key));
    meta.memory_bytes = meta.storage_bytes;
    meta.flat_segment_ids.reserve(row_group.flat_segments.size());
    meta.flat_segment_ranges.reserve(row_group.flat_segments.size());
    for (auto& segment : row_group.flat_segments) {
      meta.flat_segment_ids.emplace_back(segment.flat_segment_id);
      meta.flat_segment_ranges.emplace_back(segment.byte_range);
    }
    metas.emplace_back(std::move(meta));
  }
  ARROW_RETURN_NOT_OK(ValidateCellMetas(&metas, total_rows, key));
  return metas;
}

static arrow::Status ValidateCellMetas(std::vector<VortexCellMeta>* metas,
                                       uint64_t total_rows,
                                       const std::string& key) {
  if (metas == nullptr) {
    return arrow::Status::Invalid("Vortex cell meta validation got null metas");
  }
  if (metas->empty()) {
    if (total_rows == 0) {
      return arrow::Status::OK();
    }
    return arrow::Status::Invalid(fmt::format("Vortex field {} has no cells", key));
  }

  std::unordered_set<uint64_t> seen_flat_segments;
  for (auto& meta : *metas) {
    if (meta.flat_segment_ids.size() != meta.flat_segment_ranges.size()) {
      return arrow::Status::Invalid(fmt::format(
          "Vortex field {} physical unit {} has mismatched flat segment ids and byte ranges, ids={}, ranges={}", key,
          meta.physical_unit_index, meta.flat_segment_ids.size(), meta.flat_segment_ranges.size()));
    }
    if (meta.flat_segment_ids.empty() && !(total_rows == 0 && meta.row_count == 0)) {
      return arrow::Status::Invalid(
          fmt::format("Vortex field {} physical unit {} has no flat segments", key, meta.physical_unit_index));
    }
    for (auto flat_segment_id : meta.flat_segment_ids) {
      if (!seen_flat_segments.emplace(flat_segment_id).second) {
        return arrow::Status::Invalid(
            fmt::format("Vortex field {} flat segment {} is shared by multiple cells", key, flat_segment_id));
      }
    }
  }

  std::sort(metas->begin(), metas->end(),
            [](const auto& lhs, const auto& rhs) { return lhs.row_offset < rhs.row_offset; });

  uint64_t next_row_offset = 0;
  for (uint64_t i = 0; i < metas->size(); ++i) {
    auto& meta = (*metas)[i];
    if (meta.row_count == 0 && total_rows != 0) {
      return arrow::Status::Invalid(fmt::format("Vortex field {} cell {} has zero rows", key, i));
    }
    if (meta.row_offset != next_row_offset) {
      return arrow::Status::Invalid(
          fmt::format("Vortex field {} cell rows are not contiguous at cell {}, got row_offset={}, expected={}", key, i,
                      meta.row_offset, next_row_offset));
    }
    if (meta.row_count > std::numeric_limits<uint64_t>::max() - next_row_offset) {
      return arrow::Status::Invalid(
          fmt::format("Vortex field {} cell {} row range overflows, row_offset={}, row_count={}", key, i,
                      meta.row_offset, meta.row_count));
    }
    next_row_offset += meta.row_count;
    meta.cell_id = i;
  }
  if (next_row_offset != total_rows) {
    return arrow::Status::Invalid(
        fmt::format("Vortex field {} cells cover {} rows, expected {}", key, next_row_offset, total_rows));
  }
  return arrow::Status::OK();
}

static arrow::Status ValidateCellMetasReady(const VortexCellMetasPtr& metas,
                                            uint64_t total_rows,
                                            const std::string& key) {
  if (!metas) {
    return arrow::Status::Invalid("Vortex cell meta validation got null metas");
  }
  if (metas->empty()) {
    if (total_rows == 0) {
      return arrow::Status::OK();
    }
    return arrow::Status::Invalid(fmt::format("Vortex field {} has no cells", key));
  }

  std::unordered_set<uint64_t> seen_flat_segments;
  uint64_t next_row_offset = 0;
  for (uint64_t i = 0; i < metas->size(); ++i) {
    const auto& meta = (*metas)[i];
    if (meta.cell_id != i) {
      return arrow::Status::Invalid(
          fmt::format("Vortex field {} cell {} has mismatched cell id {}", key, i, meta.cell_id));
    }
    if (meta.row_count == 0 && total_rows != 0) {
      return arrow::Status::Invalid(fmt::format("Vortex field {} cell {} has zero rows", key, i));
    }
    if (meta.row_offset != next_row_offset) {
      return arrow::Status::Invalid(
          fmt::format("Vortex field {} cell rows are not contiguous at cell {}, got row_offset={}, expected={}", key, i,
                      meta.row_offset, next_row_offset));
    }
    if (meta.row_count > std::numeric_limits<uint64_t>::max() - next_row_offset) {
      return arrow::Status::Invalid(
          fmt::format("Vortex field {} cell {} row range overflows, row_offset={}, row_count={}", key, i,
                      meta.row_offset, meta.row_count));
    }
    if (meta.flat_segment_ids.size() != meta.flat_segment_ranges.size()) {
      return arrow::Status::Invalid(fmt::format(
          "Vortex field {} physical unit {} has mismatched flat segment ids and byte ranges, ids={}, ranges={}", key,
          meta.physical_unit_index, meta.flat_segment_ids.size(), meta.flat_segment_ranges.size()));
    }
    if (meta.flat_segment_ids.empty() && !(total_rows == 0 && meta.row_count == 0)) {
      return arrow::Status::Invalid(
          fmt::format("Vortex field {} physical unit {} has no flat segments", key, meta.physical_unit_index));
    }
    for (auto flat_segment_id : meta.flat_segment_ids) {
      if (!seen_flat_segments.emplace(flat_segment_id).second) {
        return arrow::Status::Invalid(
            fmt::format("Vortex field {} flat segment {} is shared by multiple cells", key, flat_segment_id));
      }
    }
    next_row_offset += meta.row_count;
  }
  if (next_row_offset != total_rows) {
    return arrow::Status::Invalid(
        fmt::format("Vortex field {} cells cover {} rows, expected {}", key, next_row_offset, total_rows));
  }
  return arrow::Status::OK();
}

}  // namespace

}  // namespace milvus_storage::vortex
