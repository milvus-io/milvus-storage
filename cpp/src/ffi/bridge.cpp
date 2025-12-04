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

#include "milvus-storage/ffi_internal/bridge.h"

#include <memory>
#include <optional>
#include <vector>
#include <assert.h>

namespace milvus_storage {
using namespace milvus_storage::api;

struct ColumnGroupFileExporter {
  public:
  void Export(const ColumnGroupFile* cgf, CColumnGroupFile* ccgf) {
    path_ = cgf->path;
    ccgf->path = path_.c_str();

    ccgf->start_index = cgf->start_index;
    ccgf->end_index = cgf->end_index;
    if (cgf->private_data.has_value()) {
      private_data_ = cgf->private_data.value();  // copy
      ccgf->private_data = private_data_.data();
      ccgf->private_data_size = private_data_.size();
    } else {
      ccgf->private_data = nullptr;
      ccgf->private_data_size = 0;
    }
  }

  private:
  std::string path_;
  std::vector<uint8_t> private_data_;
};

struct ColumnGroupExporter {
  public:
  void Export(const ColumnGroup* cg, CColumnGroup* ccg) {
    assert(cg != nullptr && ccg != nullptr);

    // export columns
    size_t num_of_columns = cg->columns.size();
    columns_holder_ = std::make_unique<const char*[]>(num_of_columns);
    colnames_.resize(num_of_columns);
    for (size_t i = 0; i < num_of_columns; i++) {
      colnames_[i] = cg->columns[i];
      columns_holder_[i] = colnames_[i].c_str();
    }
    ccg->columns = columns_holder_.get();
    ccg->num_of_columns = num_of_columns;

    // export format
    format_ = cg->format;
    ccg->format = format_.c_str();

    // export files
    size_t num_of_files = cg->files.size();
    files_holder_ = std::make_unique<CColumnGroupFile[]>(num_of_files);
    cgf_exporters_.resize(num_of_files);
    for (size_t i = 0; i < num_of_files; i++) {
      cgf_exporters_[i] = std::make_unique<ColumnGroupFileExporter>();
      cgf_exporters_[i]->Export(&cg->files[i], files_holder_.get() + i);
    }
    ccg->files = files_holder_.get();
    ccg->num_of_files = num_of_files;
  }

  private:
  std::vector<std::string> colnames_;
  std::unique_ptr<const char*[]> columns_holder_;

  std::string format_;

  std::unique_ptr<CColumnGroupFile[]> files_holder_;
  std::vector<std::unique_ptr<ColumnGroupFileExporter>> cgf_exporters_;
};

struct ColumnGroupsExporter {
  public:
  static arrow::Status Export(const ColumnGroups* cgs, CColumnGroups* out_ccgs) {
    ColumnGroupsExporter* exporter = new ColumnGroupsExporter();
    return exporter->ExportInternal(cgs, out_ccgs);
  }

  private:
  static void Release(CColumnGroups* ccg) {
    delete reinterpret_cast<ColumnGroupsExporter*>(ccg->private_data);
    ccg->release = nullptr;
    ccg->private_data = nullptr;
  }

  arrow::Status ExportInternal(const ColumnGroups* cgs, CColumnGroups* ccgs) {
    assert(cgs != nullptr && ccgs != nullptr);
    ccgp_holder_ = std::make_unique<CColumnGroup[]>(cgs->size());
    cg_exporters_.resize(cgs->size());
    for (size_t i = 0; i < cgs->size(); i++) {
      cg_exporters_[i] = std::make_unique<ColumnGroupExporter>();
      cg_exporters_[i]->Export(cgs->get_column_group(i).get(), ccgp_holder_.get() + i);
    }
    ccgs->column_group_array = ccgp_holder_.get();
    ccgs->num_of_column_groups = cgs->size();

    ccgs->private_data = this;
    ccgs->release = ColumnGroupsExporter::Release;

    if (cgs->meta_size() > 0) {
      meta_keys_holder_ = std::make_unique<const char*[]>(cgs->meta_size());
      meta_values_holder_ = std::make_unique<const char*[]>(cgs->meta_size());
      metadata_holder_.resize(cgs->meta_size());
      for (size_t i = 0; i < cgs->meta_size(); i++) {
        ARROW_ASSIGN_OR_RAISE(auto meta, cgs->get_metadata(i))
        metadata_holder_[i] = std::make_pair(meta.first, meta.second);
        meta_keys_holder_[i] = metadata_holder_[i].first.c_str();
        meta_values_holder_[i] = metadata_holder_[i].second.c_str();
      }

      ccgs->meta_keys = meta_keys_holder_.get();
      ccgs->meta_values = meta_values_holder_.get();
      ccgs->meta_len = cgs->meta_size();
    } else {
      ccgs->meta_keys = nullptr;
      ccgs->meta_values = nullptr;
      ccgs->meta_len = 0;
    }

    return arrow::Status::OK();
  }

  private:
  std::unique_ptr<CColumnGroup[]> ccgp_holder_;
  std::vector<std::unique_ptr<ColumnGroupExporter>> cg_exporters_;

  std::vector<std::pair<std::string, std::string>> metadata_holder_;
  std::unique_ptr<const char*[]> meta_keys_holder_;
  std::unique_ptr<const char*[]> meta_values_holder_;
};

struct ColumnGroupsImporter {
  public:
  static void ImportColumnGroupFile(const CColumnGroupFile* in_ccgf, ColumnGroupFile* cgf) {
    assert(in_ccgf != nullptr && cgf != nullptr);
    cgf->path = std::string(in_ccgf->path);
    cgf->start_index = in_ccgf->start_index;
    cgf->end_index = in_ccgf->end_index;

    if (in_ccgf->private_data != nullptr) {
      cgf->private_data =
          std::vector<uint8_t>(in_ccgf->private_data, in_ccgf->private_data + in_ccgf->private_data_size);
    }
  }

  static void ImportColumnGroup(const CColumnGroup* in_ccg, ColumnGroup* cg) {
    assert(in_ccg != nullptr && cg != nullptr);
    for (size_t i = 0; i < in_ccg->num_of_columns; i++) {
      cg->columns.emplace_back(std::string(in_ccg->columns[i]));
    }
    cg->format = std::string(in_ccg->format);

    for (size_t i = 0; i < in_ccg->num_of_files; i++) {
      ColumnGroupFile cgf;
      ImportColumnGroupFile(&in_ccg->files[i], &cgf);
      cg->files.emplace_back(std::move(cgf));
    }
  }

  static arrow::Status Import(const CColumnGroups* in_ccgs, ColumnGroups* cgs) {
    assert(in_ccgs != nullptr && cgs != nullptr);
    for (size_t i = 0; i < in_ccgs->num_of_column_groups; i++) {
      std::shared_ptr<ColumnGroup> cg = std::make_shared<ColumnGroup>();
      ImportColumnGroup(&in_ccgs->column_group_array[i], cg.get());
      ARROW_RETURN_NOT_OK(cgs->add_column_group(cg));
    }

    std::vector<std::string_view> keys;
    std::vector<std::string_view> values;
    for (size_t i = 0; i < in_ccgs->meta_len; i++) {
      keys.emplace_back(in_ccgs->meta_keys[i]);
      values.emplace_back(in_ccgs->meta_values[i]);
    }
    ARROW_RETURN_NOT_OK(cgs->add_metadatas(keys, values));

    return arrow::Status::OK();
  }
};

arrow::Status export_column_groups(const ColumnGroups* cgs, CColumnGroups* out_ccgs) {
  return ColumnGroupsExporter::Export(cgs, out_ccgs);
}

arrow::Status import_column_groups(const CColumnGroups* ccgs, ColumnGroups* out_cgs) {
  return ColumnGroupsImporter::Import(ccgs, out_cgs);
}

}  // namespace milvus_storage
