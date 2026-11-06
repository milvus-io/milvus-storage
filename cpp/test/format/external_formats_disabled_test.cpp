// Copyright 2026 Zilliz
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

#ifdef MILVUS_STORAGE_NO_EXTERNAL_FORMATS

#include <gtest/gtest.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_talon_c.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/properties.h"

namespace milvus_storage {

TEST(ExternalFormatsDisabled, RemovedFormatsReturnNotImplemented) {
  for (const auto* format : {LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE, LOON_FORMAT_PAIMON_TABLE}) {
    auto result = Format::get(format);
    ASSERT_FALSE(result.ok()) << format;
    EXPECT_TRUE(result.status().IsNotImplemented()) << result.status().ToString();
    EXPECT_NE(result.status().message().find("WITH_EXTERNAL_TABLE_FORMATS=OFF"), std::string::npos);
  }
}

TEST(ExternalFormatsDisabled, ParquetAndVortexRemainAvailable) {
  EXPECT_TRUE(Format::get(LOON_FORMAT_PARQUET).ok());
  EXPECT_TRUE(Format::get(LOON_FORMAT_VORTEX).ok());
}

TEST(ExternalFormatsDisabled, TalonClientCreateReturnsError) {
  LoonTalonClientHandle client = 0;
  LoonFFIResult result = loon_talon_client_create(nullptr, &client);
  EXPECT_FALSE(loon_ffi_is_success(&result));
  loon_ffi_free_result(&result);
}

}  // namespace milvus_storage

#endif  // MILVUS_STORAGE_NO_EXTERNAL_FORMATS
