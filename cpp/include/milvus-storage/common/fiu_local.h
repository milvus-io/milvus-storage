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

/**
 * @file fiu_local.h
 * @brief Fault injection macros and key definitions for milvus-storage
 *
 * This header provides:
 * 1. Fault injection point key definitions (constexpr strings)
 * 2. Convenient macros for adding fault injection points
 *
 * When BUILD_WITH_FIU is defined:
 *   - FIU_RETURN_ON: Returns specified value when fault point is triggered
 *   - FIU_DO_ON: Executes code when fault point is triggered
 *
 * When BUILD_WITH_FIU is not defined:
 *   - All macros compile to no-ops
 *
 * Usage:
 *   FIU_RETURN_ON(FIUKEY_WRITER_FLUSH_FAIL, Status::IOError("fault"));
 *   FIU_DO_ON(FIUKEY_FS_OPEN_OUTPUT_FAIL, { throw std::runtime_error("fault"); });
 */

// ==================== Fault Point Key Macros (Single Source of Truth) ====================
// Use these macros directly in C++ code and FFI exports.

// Writer fault points
#define FIUKEY_WRITER_WRITE_FAIL "writer.write.fail"
#define FIUKEY_WRITER_FLUSH_FAIL "writer.flush.fail"
#define FIUKEY_WRITER_CLOSE_FAIL "writer.close.fail"

// Reader fault points (low-level)
#define FIUKEY_COLUMN_GROUP_READ_FAIL "column_group.read.fail"
#define FIUKEY_TAKE_ROWS_FAIL "take_rows.fail"
#define FIUKEY_CHUNK_READER_READ_FAIL "chunk_reader.read.fail"
#define FIUKEY_READER_OPEN_FAIL "reader.open.fail"

// Transaction/Manifest fault points
#define FIUKEY_MANIFEST_COMMIT_FAIL "manifest.commit.fail"
#define FIUKEY_MANIFEST_READ_FAIL "manifest.read.fail"
#define FIUKEY_MANIFEST_WRITE_FAIL "manifest.write.fail"

// Filesystem fault points
#define FIUKEY_FS_OPEN_OUTPUT_FAIL "fs.open_output.fail"
#define FIUKEY_FS_OPEN_INPUT_FAIL "fs.open_input.fail"

// S3 Filesystem fault points
#define FIUKEY_S3FS_CREATE_UPLOAD_FAIL "s3fs.create_upload.fail"
#define FIUKEY_S3FS_PART_UPLOAD_FAIL "s3fs.part_upload.fail"
#define FIUKEY_S3FS_COMPLETE_UPLOAD_FAIL "s3fs.complete_upload.fail"
#define FIUKEY_S3FS_READ_FAIL "s3fs.read.fail"
#define FIUKEY_S3FS_READAT_FAIL "s3fs.readat.fail"

// ColumnGroup fault points
#define FIUKEY_COLUMN_GROUP_WRITE_FAIL "column_group.write.fail"

// ==================== Fault Injection Macros ====================

#ifdef BUILD_WITH_FIU

#include <fiu.h>
#include <fiu-control.h>

// Initialize fiu once (call in main or global init)
#define FIU_INIT() fiu_init(0)

// Return specified status when fault point is triggered
// Usage: FIU_RETURN_ON(FIUKEY_WRITER_FLUSH_FAIL, Status::IOError("Injected fault"));
#define FIU_RETURN_ON(name, retval) \
  do {                              \
    if (fiu_fail(name)) {           \
      return (retval);              \
    }                               \
  } while (0)

// Execute code block when fault point is triggered
// Usage: FIU_DO_ON(FIUKEY_FS_OPEN_OUTPUT_FAIL, { throw std::runtime_error("Injected fault"); });
#define FIU_DO_ON(name, action) \
  do {                          \
    if (fiu_fail(name)) {       \
      action;                   \
    }                           \
  } while (0)

// Enable a fault point once
// Usage: FIU_ENABLE_FAULT_ONETIME(FIUKEY_WRITER_FLUSH_FAIL);
#define FIU_ENABLE_FAULT_ONETIME(name) fiu_enable(name, -1, nullptr, FIU_ONETIME)

// Enable a fault point forever (until disabled)
// Usage: FIU_ENABLE_FAULT_ALWAYS(FIUKEY_WRITER_FLUSH_FAIL);
#define FIU_ENABLE_FAULT_ALWAYS(name) fiu_enable(name, -1, nullptr, 0 /* without FIU_ONETIME */)

// Disable a fault point
// Usage: FIU_DISABLE_FAULT(FIUKEY_WRITER_FLUSH_FAIL);
#define FIU_DISABLE_FAULT(name) fiu_disable(name)

#else  // BUILD_WITH_FIU not defined

// No-op implementations when FIU is disabled
#define FIU_INIT() ((void)0)
#define FIU_RETURN_ON(name, retval) ((void)0)
#define FIU_DO_ON(name, action) ((void)0)

#define FIU_ENABLE_FAULT_ONETIME(name) ((void)0)
#define FIU_ENABLE_FAULT_ALWAYS(name) ((void)0)
#define FIU_DISABLE_FAULT(name) ((void)0)

#endif  // BUILD_WITH_FIU
