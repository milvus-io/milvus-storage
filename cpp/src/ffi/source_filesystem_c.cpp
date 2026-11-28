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

#include <new>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/source_read_context.h"

LoonFFIResult loon_filesystem_get_with_context(const LoonProperties* properties,
                                               const char* path,
                                               uint32_t path_len,
                                               LoonSourceReadContextHandle context,
                                               FileSystemHandle* out_handle) {
  if (out_handle)
    *out_handle = 0;
  if (!properties || !path || path_len == 0 || !out_handle)
    RETURN_ERROR(LOON_INVALID_ARGS, "Missing source filesystem arguments");
  if (properties->count && !properties->properties)
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Invalid source filesystem properties");
  for (size_t i = 0; i < properties->count; ++i) {
    if (!properties->properties[i].key || !properties->properties[i].value)
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Invalid source filesystem properties");
  }
  try {
    auto source_context = milvus_storage::GetSourceReadContext(context);
    if (!source_context)
      RETURN_ERROR(LOON_SOURCE_AUTH_DENIED);
    std::shared_ptr<const milvus_storage::api::Properties> converted;
    if (milvus_storage::api::ConvertFFIProperties(converted, properties))
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "Invalid source filesystem properties");
    auto result = milvus_storage::FilesystemCache::getInstance().get_with_context(
        converted, std::string(path, path_len), source_context);
    RETURN_ARROW_ERROR_IF(result.status(), LOON_ARROW_ERROR, "Source filesystem resolution failed");
    auto wrapper = std::make_unique<FileSystemWrapper>(result->filesystem);
    *out_handle = reinterpret_cast<FileSystemHandle>(wrapper.release());
    RETURN_SUCCESS();
  } catch (const std::bad_alloc&) {
    RETURN_ERROR(LOON_MEMORY_ERROR, "Source filesystem allocation failed");
  } catch (...) {
    // Neither exception text nor user paths are credential-safe diagnostics.
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Source filesystem construction failed");
  }
}
