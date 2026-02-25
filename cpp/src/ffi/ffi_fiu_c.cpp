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

#include "milvus-storage/ffi_fiu_c.h"

#include <mutex>
#include <string>

#include <fmt/core.h>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/common/fiu_local.h"

// ==================== Fault Point Name Definitions (C-linkage for FFI) ====================
// These provide C-linkage symbols for external bindings (Python, Java, Rust).
// String values are defined as FIUKEY_* macros in fiu_local.h (single source of truth).

// Writer fault points
const char* loon_fiukey_writer_write_fail = FIUKEY_WRITER_WRITE_FAIL;
const char* loon_fiukey_writer_flush_fail = FIUKEY_WRITER_FLUSH_FAIL;
const char* loon_fiukey_writer_close_fail = FIUKEY_WRITER_CLOSE_FAIL;

// Reader fault points (low-level)
const char* loon_fiukey_column_group_read_fail = FIUKEY_COLUMN_GROUP_READ_FAIL;
const char* loon_fiukey_take_rows_fail = FIUKEY_TAKE_ROWS_FAIL;
const char* loon_fiukey_chunk_reader_read_fail = FIUKEY_CHUNK_READER_READ_FAIL;
const char* loon_fiukey_reader_open_fail = FIUKEY_READER_OPEN_FAIL;

// Transaction/Manifest fault points
const char* loon_fiukey_manifest_commit_fail = FIUKEY_MANIFEST_COMMIT_FAIL;
const char* loon_fiukey_manifest_read_fail = FIUKEY_MANIFEST_READ_FAIL;
const char* loon_fiukey_manifest_write_fail = FIUKEY_MANIFEST_WRITE_FAIL;

// Filesystem fault points
const char* loon_fiukey_fs_open_output_fail = FIUKEY_FS_OPEN_OUTPUT_FAIL;
const char* loon_fiukey_fs_open_input_fail = FIUKEY_FS_OPEN_INPUT_FAIL;

// S3 Filesystem fault points
const char* loon_fiukey_s3fs_create_upload_fail = FIUKEY_S3FS_CREATE_UPLOAD_FAIL;
const char* loon_fiukey_s3fs_part_upload_fail = FIUKEY_S3FS_PART_UPLOAD_FAIL;
const char* loon_fiukey_s3fs_complete_upload_fail = FIUKEY_S3FS_COMPLETE_UPLOAD_FAIL;
const char* loon_fiukey_s3fs_read_fail = FIUKEY_S3FS_READ_FAIL;
const char* loon_fiukey_s3fs_readat_fail = FIUKEY_S3FS_READAT_FAIL;

// ColumnGroup fault points
const char* loon_fiukey_column_group_write_fail = FIUKEY_COLUMN_GROUP_WRITE_FAIL;

#ifdef BUILD_WITH_FIU

static std::once_flag fiu_init_flag;

static inline void ensure_fiu_init() {
  std::call_once(fiu_init_flag, []() {
    int ret = FIU_INIT();
    if (ret != 0) {
      throw std::runtime_error(fmt::format("fiu_init failed with code: {}", ret));
    }
  });
}

LoonFFIResult loon_fiu_enable(const char* name, uint32_t name_len, int one_time) {
  try {
    if (name == nullptr || name_len == 0) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Fault point name cannot be empty");
    }

    ensure_fiu_init();

    std::string fault_name(name, name_len);
    int ret = one_time ? FIU_ENABLE_FAULT_ONETIME(fault_name.c_str()) : FIU_ENABLE_FAULT_ALWAYS(fault_name.c_str());
    if (ret != 0) {
      RETURN_ERROR(LOON_LOGICAL_ERROR, "Failed to enable fault point: " + fault_name);
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }
}

LoonFFIResult loon_fiu_disable(const char* name, uint32_t name_len) {
  try {
    if (name == nullptr || name_len == 0) {
      RETURN_ERROR(LOON_INVALID_ARGS, "Fault point name cannot be empty");
    }

    ensure_fiu_init();

    std::string fault_name(name, name_len);

    int ret = FIU_DISABLE_FAULT(fault_name.c_str());
    if (ret != 0) {
      // fiu_disable returns -1 if the point was not enabled, which is not an error
      // Just ignore this case
    }

    RETURN_SUCCESS();
  } catch (const std::exception& e) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, e.what());
  }
}

void loon_fiu_disable_all(void) {
  static const char* fault_points[] = {
      // Writer
      FIUKEY_WRITER_WRITE_FAIL,
      FIUKEY_WRITER_FLUSH_FAIL,
      FIUKEY_WRITER_CLOSE_FAIL,
      // Reader (low-level)
      FIUKEY_COLUMN_GROUP_READ_FAIL,
      FIUKEY_TAKE_ROWS_FAIL,
      FIUKEY_CHUNK_READER_READ_FAIL,
      FIUKEY_READER_OPEN_FAIL,
      // Transaction/Manifest
      FIUKEY_MANIFEST_COMMIT_FAIL,
      FIUKEY_MANIFEST_READ_FAIL,
      FIUKEY_MANIFEST_WRITE_FAIL,
      // Filesystem
      FIUKEY_FS_OPEN_OUTPUT_FAIL,
      FIUKEY_FS_OPEN_INPUT_FAIL,
      // S3 Filesystem
      FIUKEY_S3FS_CREATE_UPLOAD_FAIL,
      FIUKEY_S3FS_PART_UPLOAD_FAIL,
      FIUKEY_S3FS_COMPLETE_UPLOAD_FAIL,
      FIUKEY_S3FS_READ_FAIL,
      FIUKEY_S3FS_READAT_FAIL,
      // ColumnGroup
      FIUKEY_COLUMN_GROUP_WRITE_FAIL,
  };

  for (const auto* fp : fault_points) {
    FIU_DISABLE_FAULT(fp);
  }
}

int loon_fiu_is_enabled(void) { return 1; }

#else  // !BUILD_WITH_FIU

LoonFFIResult loon_fiu_enable(const char* /*name*/, uint32_t /*name_len*/, int /*one_time*/) {
  RETURN_ERROR(LOON_LOGICAL_ERROR, "Fault injection is not enabled. Rebuild with -DWITH_FIU=ON");
}

LoonFFIResult loon_fiu_disable(const char* /*name*/, uint32_t /*name_len*/) {
  RETURN_ERROR(LOON_LOGICAL_ERROR, "Fault injection is not enabled. Rebuild with -DWITH_FIU=ON");
}

void loon_fiu_disable_all() {}

int loon_fiu_is_enabled() { return 0; }

#endif  // BUILD_WITH_FIU
