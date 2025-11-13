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

#include "milvus-storage/transaction/transupdate.h"
#include "milvus-storage/transaction/manifest.h"

#include <cassert>
#include <memory>

#include <arrow/status.h>
#include <arrow/result.h>

namespace milvus_storage::api::transaction {

class PendingAppendFiles : public PendingUpdate<Manifest> {
  public:
  PendingAppendFiles(const ManifestPtr& base_cgs, const ManifestPtr& new_cg) : base_cgs_(base_cgs), new_cg_(new_cg) {}

  arrow::Result<ManifestPtr> apply() override {
    ARROW_RETURN_NOT_OK(base_cgs_->append_files(new_cg_));
    return base_cgs_;
  }

  private:
  ManifestPtr base_cgs_;
  ManifestPtr new_cg_;
};

class PendingAddField : public PendingUpdate<Manifest> {
  public:
  PendingAddField(const ManifestPtr& base_cgs, const ManifestPtr& new_cg) : base_cgs_(base_cgs), new_cg_(new_cg) {}

  arrow::Result<ManifestPtr> apply() override {
    if (new_cg_->size() != 1) {
      return arrow::Status::Invalid("New column groups should contain exactly one column group for AddField.");
    }

    ARROW_RETURN_NOT_OK(base_cgs_->add_column_group(new_cg_->get_column_group(0)));
    return base_cgs_;
  }

  private:
  ManifestPtr base_cgs_;
  ManifestPtr new_cg_;
};

template <typename T>
std::shared_ptr<PendingUpdate<T>> PendingUpdate<T>::CreatePendingUpdate(const UpdateType update_type,
                                                                        const std::shared_ptr<T>& base_cgs,
                                                                        const std::shared_ptr<T>& new_cg) {
  switch (update_type) {
    case APPENDFILES:
      return std::make_shared<PendingAppendFiles>(base_cgs, new_cg);
    case ADDFIELD:
      return std::make_shared<PendingAddField>(base_cgs, new_cg);
    default: {
      assert(false);
      return nullptr;
    }
  }
  // unreachable
}

template class PendingUpdate<Manifest>;
}  // namespace milvus_storage::api::transaction