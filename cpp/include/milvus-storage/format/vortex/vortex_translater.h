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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <arrow/filesystem/filesystem.h>
#include <arrow/result.h>

#include "milvus-storage/format/vortex/vortex_types.h"
#include "cachinglayer/Translator.h"

namespace milvus_storage::vortex {

class VortexCellGuard {
  public:
  VortexCellGuard(VortexCellMetasPtr cell_metas, uint64_t cell_id, std::shared_ptr<VortexRangeFile> range_file);

  VortexCellGuard(const VortexCellGuard&) = delete;
  VortexCellGuard& operator=(const VortexCellGuard&) = delete;
  VortexCellGuard(VortexCellGuard&&) = delete;
  VortexCellGuard& operator=(VortexCellGuard&&) = delete;

  ~VortexCellGuard();

  const VortexCellMeta& meta() const { return (*cell_metas_)[cell_id_]; }

  uint64_t pinned_bytes() const { return pinned_bytes_; }

  milvus::cachinglayer::ResourceUsage CellByteSize() const;

  private:
  VortexCellMetasPtr cell_metas_;
  uint64_t cell_id_ = 0;
  std::shared_ptr<VortexRangeFile> range_file_;
  uint64_t pinned_bytes_ = 0;
};

class VortexTranslater : public milvus::cachinglayer::Translator<VortexCellGuard> {
  public:
  using CellLoader = std::function<arrow::Status(const std::vector<milvus::cachinglayer::cid_t>&)>;

  static arrow::Result<std::unique_ptr<VortexTranslater>> Make(
      VortexCellMetasPtr cell_metas,
      std::shared_ptr<arrow::fs::FileSystem> source_fs,
      std::string source_path,
      std::shared_ptr<arrow::fs::FileSystem> sparse_fs,
      std::string sparse_path,
      CacheWarmupPolicy cache_warmup_policy = CacheWarmupPolicy::CacheWarmupPolicy_Disable);

  VortexTranslater(const VortexTranslater&) = delete;
  VortexTranslater& operator=(const VortexTranslater&) = delete;

  size_t num_cells() const override { return cell_metas_ ? cell_metas_->size() : 0; }

  milvus::cachinglayer::cid_t cell_id_of(milvus::cachinglayer::uid_t uid) const override;

  std::pair<milvus::cachinglayer::ResourceUsage, milvus::cachinglayer::ResourceUsage> estimated_byte_size_of_cell(
      milvus::cachinglayer::cid_t cid) const override;

  const std::string& key() const override { return key_; }

  milvus::cachinglayer::Meta* meta() override { return &meta_; }

  int64_t cells_storage_bytes(const std::vector<milvus::cachinglayer::cid_t>& cids) const override;

  std::vector<std::pair<milvus::cachinglayer::cid_t, std::unique_ptr<VortexCellGuard>>> get_cells(
      milvus::OpContext* ctx, const std::vector<milvus::cachinglayer::cid_t>& cids) override;

  private:
  VortexTranslater(VortexCellMetasPtr cell_metas,
                   CellLoader cell_loader,
                   std::shared_ptr<VortexRangeFile> range_file,
                   std::string key,
                   CacheWarmupPolicy cache_warmup_policy);

  VortexCellMetasPtr cell_metas_;
  CellLoader cell_loader_;
  std::shared_ptr<VortexRangeFile> range_file_;
  std::string key_;
  milvus::cachinglayer::Meta meta_;
};

}  // namespace milvus_storage::vortex
