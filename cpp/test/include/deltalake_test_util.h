// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include <arrow/result.h>
#include <arrow/type.h>

#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::deltalake::test {

// Write synthetic Parquet data and publish a Delta snapshot at version zero.
// Deletion positions are relative to the first file.
arrow::Result<std::shared_ptr<arrow::Schema>> WriteTable(const ArrowFileSystemPtr& fs,
                                                         const std::string& path,
                                                         int64_t rows_per_file,
                                                         int file_count = 1,
                                                         std::span<const int64_t> deleted_rows = {});

arrow::Result<std::string> ExploreUri(const ArrowFileSystemConfig& config, const std::string& path);

}  // namespace milvus_storage::deltalake::test
