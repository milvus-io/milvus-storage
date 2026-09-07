// Copyright 2025 Zilliz
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

// Talon config-plumbing and factory-gating tests. These are pure configuration
// logic (no Talon SDK, no coordinator, no cloud), so they run in the default
// build regardless of WITH_TALON.

#include <gtest/gtest.h>

#include <string>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/properties.h"

namespace milvus_storage::test {

TEST(TalonConfigTest, PropertiesParseIntoConfig) {
  api::Properties properties;
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_ENABLED, "true"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_COORDINATOR, "talon-coord:7000"), std::nullopt);

  ArrowFileSystemConfig config;
  ASSERT_TRUE(ArrowFileSystemConfig::create_file_system_config(properties, config).ok());
  EXPECT_TRUE(config.talon_enabled);
  EXPECT_EQ(config.talon_coordinator, "talon-coord:7000");
}

TEST(TalonConfigTest, TalonDefaultsAreOff) {
  api::Properties properties;
  ArrowFileSystemConfig config;
  ASSERT_TRUE(ArrowFileSystemConfig::create_file_system_config(properties, config).ok());
  EXPECT_FALSE(config.talon_enabled);
  EXPECT_EQ(config.talon_coordinator, "");
}

TEST(TalonConfigTest, CacheKeyDistinguishesTalonRouting) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.cloud_provider = kCloudProviderAWS;
  base.address = "s3.amazonaws.com";
  base.bucket_name = "b";

  ArrowFileSystemConfig talon = base;
  talon.talon_enabled = true;
  talon.talon_coordinator = "coord-a:7000";

  // Talon vs no-Talon must not collide.
  EXPECT_NE(base.GetCacheKey(), talon.GetCacheKey());

  // Different coordinators must not share an instance.
  ArrowFileSystemConfig talon_other = talon;
  talon_other.talon_coordinator = "coord-b:7000";
  EXPECT_NE(talon.GetCacheKey(), talon_other.GetCacheKey());

  // Identical Talon routing is stable.
  ArrowFileSystemConfig talon_same = talon;
  EXPECT_EQ(talon.GetCacheKey(), talon_same.GetCacheKey());
}

// Enabling Talon in a build without WITH_TALON must fail loudly rather than
// silently ignore the request. The rejection happens before any backend is
// built, so storage_type is irrelevant here.
#ifndef WITH_TALON
TEST(TalonConfigTest, EnabledWithoutTalonBuildIsRejected) {
  ArrowFileSystemConfig config;
  config.storage_type = "local";
  config.talon_enabled = true;

  auto result = CreateArrowFileSystem(config);
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().ToString().find("Talon"), std::string::npos) << result.status().ToString();
}
#endif  // WITH_TALON

}  // namespace milvus_storage::test
