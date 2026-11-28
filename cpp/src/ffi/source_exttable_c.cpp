// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus-storage/ffi_source_read_context_c.h"

#include <cstring>
#include <new>

#include "milvus-storage/common/layout.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/azure/source_sas_token_policy.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/transaction/transaction.h"

namespace {
bool ValidProperties(const LoonProperties* properties) {
  if (!properties || (properties->count && !properties->properties))
    return false;
  for (size_t i = 0; i < properties->count; ++i)
    if (!properties->properties[i].key || !properties->properties[i].value)
      return false;
  return true;
}
}  // namespace

LoonFFIResult loon_exttable_get_file_info_with_context(const char* format,
                                                       const char* file_path,
                                                       const LoonProperties* properties,
                                                       LoonSourceReadContextHandle context,
                                                       uint64_t* out_num_of_rows) {
  if (out_num_of_rows)
    *out_num_of_rows = 0;
  if (!format || !file_path || !out_num_of_rows)
    RETURN_ERROR(LOON_INVALID_ARGS, "Missing source file-info arguments");
  if (!ValidProperties(properties))
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Invalid source properties");
  try {
    auto source_context = milvus_storage::GetSourceReadContext(context);
    if (!source_context)
      RETURN_ERROR(LOON_SOURCE_AUTH_DENIED);
    std::shared_ptr<const milvus_storage::api::Properties> converted;
    if (milvus_storage::api::ConvertFFIProperties(converted, properties))
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Invalid source properties");
    auto fmt = milvus_storage::Format::get(format);
    RETURN_ARROW_ERROR_IF(fmt.status(), LOON_INVALID_ARGS, "Unsupported source format");
    milvus_storage::api::ColumnGroupFile file{file_path, 0, 0, {}};
    auto reader = (*fmt)->create_reader_with_context(nullptr, file, converted, {}, nullptr, source_context);
    RETURN_ARROW_ERROR_IF(reader.status(), LOON_ARROW_ERROR, "Source reader creation failed");
    auto rows = (*reader)->get_row_group_infos();
    RETURN_ARROW_ERROR_IF(rows.status(), LOON_ARROW_ERROR, "Source row metadata read failed");
    const auto status = milvus_storage::fs::SourceCredentialArrowStatus(source_context->CheckState());
    RETURN_ARROW_ERROR_IF(status, LOON_ARROW_ERROR, "Source context is no longer active");
    *out_num_of_rows = rows->empty() ? 0 : rows->back().end_offset;
    RETURN_SUCCESS();
  } catch (const std::bad_alloc&) {
    RETURN_ERROR(LOON_MEMORY_ERROR, "Source file-info allocation failed");
  } catch (...) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Source file-info failed");
  }
}

LoonFFIResult loon_exttable_explore_with_context(const char** columns,
                                                 size_t col_lens,
                                                 const char* format,
                                                 const char* base_dir,
                                                 const char* explore_dir,
                                                 const LoonProperties* properties,
                                                 LoonSourceReadContextHandle context,
                                                 uint64_t* out_num_of_files,
                                                 char** out_column_groups_file_path) {
  if (out_num_of_files)
    *out_num_of_files = 0;
  if (out_column_groups_file_path)
    *out_column_groups_file_path = nullptr;
  if (!columns || !col_lens || !format || !base_dir || !explore_dir || !out_num_of_files ||
      !out_column_groups_file_path)
    RETURN_ERROR(LOON_INVALID_ARGS, "Missing source exploration arguments");
  for (size_t i = 0; i < col_lens; ++i)
    if (!columns[i])
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid source column");
  if (!ValidProperties(properties))
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Invalid source properties");
  try {
    auto source_context = milvus_storage::GetSourceReadContext(context);
    if (!source_context)
      RETURN_ERROR(LOON_SOURCE_AUTH_DENIED);
    std::shared_ptr<const milvus_storage::api::Properties> converted;
    if (milvus_storage::api::ConvertFFIProperties(converted, properties))
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Invalid source properties");
    auto fmt = milvus_storage::Format::get(format);
    RETURN_ARROW_ERROR_IF(fmt.status(), LOON_INVALID_ARGS, "Unsupported source format");
    auto files = (*fmt)->explore_with_context(explore_dir, converted, source_context);
    RETURN_ARROW_ERROR_IF(files.status(), LOON_ARROW_ERROR, "Source exploration failed");
    milvus_storage::api::ColumnGroups groups;
    groups.push_back(std::make_shared<milvus_storage::api::ColumnGroup>(milvus_storage::api::ColumnGroup{
        .columns = std::vector<std::string>(columns, columns + col_lens), .format = format, .files = *files}));
    // Only source I/O receives the source context. Internal manifest writes use
    // the configured internal-storage identity, exactly as in the legacy API.
    auto internal_fs = milvus_storage::FilesystemCache::getInstance().get(converted);
    RETURN_ARROW_ERROR_IF(internal_fs.status(), LOON_ARROW_ERROR, "Internal filesystem resolution failed");
    auto transaction = milvus_storage::api::transaction::Transaction::Open(*internal_fs, base_dir);
    RETURN_ARROW_ERROR_IF(transaction.status(), LOON_LOGICAL_ERROR, "Internal manifest transaction failed");
    (*transaction)->AppendFiles(groups);
    const auto status = milvus_storage::fs::SourceCredentialArrowStatus(source_context->CheckState());
    RETURN_ARROW_ERROR_IF(status, LOON_ARROW_ERROR, "Source context is no longer active");
    auto version = (*transaction)->Commit();
    RETURN_ARROW_ERROR_IF(version.status(), LOON_LOGICAL_ERROR, "Internal manifest commit failed");
    auto path = milvus_storage::get_manifest_filepath(base_dir, *version);
    auto result = strdup(path.c_str());
    if (!result)
      RETURN_ERROR(LOON_MEMORY_ERROR, "Manifest path allocation failed");
    *out_column_groups_file_path = result;
    *out_num_of_files = files->size();
    RETURN_SUCCESS();
  } catch (const std::bad_alloc&) {
    RETURN_ERROR(LOON_MEMORY_ERROR, "Source exploration allocation failed");
  } catch (...) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Source exploration failed");
  }
}
