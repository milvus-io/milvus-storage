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

#include "milvus-storage/format/vortex/vortex_translater.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include <arrow/io/interfaces.h>
#include <fmt/format.h>

namespace milvus_storage::vortex {

namespace {

static uint64_t CheckedAddByteSize(uint64_t total, uint64_t value, const char* label) {
  if (total > std::numeric_limits<uint64_t>::max() - value) {
    throw std::overflow_error(fmt::format("{} overflows uint64_t", label));
  }
  return total + value;
}

static int64_t ToResourceBytes(uint64_t bytes, const char* label) {
  if (bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error(fmt::format("{} overflows int64_t", label));
  }
  return static_cast<int64_t>(bytes);
}

static uint64_t MergedByteSize(const std::vector<ByteRange>& flat_segment_ranges) {
  uint64_t total = 0;
  for (const auto& range : MergeByteRanges(flat_segment_ranges)) {
    total = CheckedAddByteSize(total, range.length, "Merged vortex byte range size");
  }
  return total;
}

static uint64_t PinnedByteSize(const VortexCellMeta& meta) {
  return std::max(meta.storage_bytes, MergedByteSize(meta.flat_segment_ranges));
}

static std::vector<ByteRange> MergeCellByteRanges(const VortexCellMetas& cell_metas,
                                                  const std::vector<milvus::cachinglayer::cid_t>& cids) {
  std::vector<ByteRange> ranges;
  for (auto cid : cids) {
    const auto& meta = cell_metas[cid];
    ranges.insert(ranges.end(), meta.flat_segment_ranges.begin(), meta.flat_segment_ranges.end());
  }
  return MergeByteRanges(std::move(ranges));
}

}  // namespace

VortexCellGuard::VortexCellGuard(VortexCellMetasPtr cell_metas,
                                 uint64_t cell_id,
                                 std::shared_ptr<VortexRangeFile> range_file)
    : cell_metas_(std::move(cell_metas)),
      cell_id_(cell_id),
      range_file_(std::move(range_file)),
      pinned_bytes_(PinnedByteSize(meta())) {}

VortexCellGuard::~VortexCellGuard() {
  if (!range_file_) {
    return;
  }
  for (const auto& byte_range : MergeByteRanges(meta().flat_segment_ranges)) {
    range_file_->Punch(byte_range.offset, byte_range.length);
  }
}

milvus::cachinglayer::ResourceUsage VortexCellGuard::CellByteSize() const {
  const auto bytes = pinned_bytes_ > 0 ? pinned_bytes_ : meta().memory_bytes;
  return {ToResourceBytes(bytes, "Vortex cell byte size"), 0};
}

arrow::Result<std::unique_ptr<VortexTranslater>> VortexTranslater::Make(
    VortexCellMetasPtr cell_metas,
    std::shared_ptr<arrow::fs::FileSystem> source_fs,
    std::string source_path,
    std::shared_ptr<arrow::fs::FileSystem> sparse_fs,
    std::string sparse_path,
    CacheWarmupPolicy cache_warmup_policy) {
  if (!source_fs) {
    return arrow::Status::Invalid("VortexTranslater requires a non-null source filesystem");
  }
  if (!sparse_fs) {
    return arrow::Status::Invalid("VortexTranslater requires a non-null range filesystem");
  }
  if (!cell_metas) {
    return arrow::Status::Invalid("VortexTranslater requires non-null cell metas");
  }
  ARROW_ASSIGN_OR_RAISE(auto range_file, GetVortexRangeFile(sparse_fs, sparse_path));
  if (!range_file) {
    return arrow::Status::Invalid("VortexTranslater requires a non-null range file");
  }

  ARROW_ASSIGN_OR_RAISE(auto input_file, source_fs->OpenInputFile(source_path));

  auto loader = [input_file, range_file,
                 cell_metas](const std::vector<milvus::cachinglayer::cid_t>& cids) -> arrow::Status {
    for (const auto& byte_range : MergeCellByteRanges(*cell_metas, cids)) {
      ARROW_RETURN_NOT_OK(FillVortexRangeFile(input_file, range_file, byte_range.offset, byte_range.length));
    }
    return arrow::Status::OK();
  };

  const auto key = source_path;
  return std::unique_ptr<VortexTranslater>(
      new VortexTranslater(std::move(cell_metas), std::move(loader), std::move(range_file), key, cache_warmup_policy));
}

VortexTranslater::VortexTranslater(VortexCellMetasPtr cell_metas,
                                   CellLoader cell_loader,
                                   std::shared_ptr<VortexRangeFile> range_file,
                                   std::string key,
                                   CacheWarmupPolicy cache_warmup_policy)
    : cell_metas_(std::move(cell_metas)),
      cell_loader_(std::move(cell_loader)),
      range_file_(std::move(range_file)),
      key_(std::move(key)),
      meta_(milvus::cachinglayer::StorageType::MEMORY,
            milvus::cachinglayer::CellIdMappingMode::IDENTICAL,
            milvus::cachinglayer::CellDataType::SCALAR_FIELD,
            cache_warmup_policy,
            true) {}

milvus::cachinglayer::cid_t VortexTranslater::cell_id_of(milvus::cachinglayer::uid_t uid) const { return uid; }

std::pair<milvus::cachinglayer::ResourceUsage, milvus::cachinglayer::ResourceUsage>
VortexTranslater::estimated_byte_size_of_cell(milvus::cachinglayer::cid_t cid) const {
  const auto& cell_metas = *cell_metas_;
  if (cid < 0 || static_cast<size_t>(cid) >= cell_metas.size()) {
    throw std::out_of_range(fmt::format("Vortex cell id {} is out of range, num_cells={}", cid, cell_metas.size()));
  }
  const auto bytes = ToResourceBytes(PinnedByteSize(cell_metas[cid]), "Vortex cell byte size");
  milvus::cachinglayer::ResourceUsage loaded(bytes, 0);
  milvus::cachinglayer::ResourceUsage loading_overhead(bytes, 0);
  return {loaded, loading_overhead};
}

int64_t VortexTranslater::cells_storage_bytes(const std::vector<milvus::cachinglayer::cid_t>& cids) const {
  uint64_t total = 0;
  const auto& cell_metas = *cell_metas_;
  for (auto cid : cids) {
    if (cid < 0 || static_cast<size_t>(cid) >= cell_metas.size()) {
      throw std::out_of_range(fmt::format("Vortex cell id {} is out of range, num_cells={}", cid, cell_metas.size()));
    }
    total = CheckedAddByteSize(total, cell_metas[cid].storage_bytes, "Vortex cells storage byte size");
  }
  return ToResourceBytes(total, "Vortex cells storage byte size");
}

std::vector<std::pair<milvus::cachinglayer::cid_t, std::unique_ptr<VortexCellGuard>>> VortexTranslater::get_cells(
    milvus::OpContext* ctx, const std::vector<milvus::cachinglayer::cid_t>& cids) {
  (void)ctx;

  std::vector<std::pair<milvus::cachinglayer::cid_t, std::unique_ptr<VortexCellGuard>>> cells;
  cells.reserve(cids.size());
  const auto& cell_metas = *cell_metas_;
  for (auto cid : cids) {
    if (cid < 0 || static_cast<size_t>(cid) >= cell_metas.size()) {
      throw std::out_of_range(fmt::format("Vortex cell id {} is out of range, num_cells={}", cid, cell_metas.size()));
    }
  }
  auto status = cell_loader_(cids);
  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }
  for (auto cid : cids) {
    cells.emplace_back(cid, std::make_unique<VortexCellGuard>(cell_metas_, static_cast<uint64_t>(cid), range_file_));
  }
  return cells;
}

}  // namespace milvus_storage::vortex
