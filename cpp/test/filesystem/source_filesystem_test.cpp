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

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/azure/source_sas_token_policy.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/ffi_source_read_context_c.h"

#include <atomic>
#include <cstring>
#include <future>
#include <gtest/gtest.h>

namespace milvus_storage {
namespace {
constexpr auto kRoot = "https://acct.blob.core.windows.net/tables/team/table";
constexpr auto kFile = "https://acct.blob.core.windows.net/tables/team/table/data.parquet";
constexpr auto kPath = "team/table/data.parquet";

std::shared_ptr<api::Properties> MinimalProperties() {
  return std::make_shared<api::Properties>(api::Properties{{"extfs.catalog.cloud_provider", std::string("azure")},
                                                           {"extfs.catalog.access_key_id", std::string("acct")}});
}

class DenyingProvider final : public SourceCredentialProvider {
  public:
  mutable std::atomic<int> calls{0};
  SourceCredentialResponse Fetch(const SourceCredentialRequest&) const override {
    ++calls;
    SourceCredentialResponse result;
    result.status = LOON_SOURCE_CREDENTIAL_DENIED;
    return result;
  }
};

SourceReadContextPtr Context(const std::shared_ptr<DenyingProvider>& provider) {
  auto result = SourceReadContext::Create({"acct", {kRoot}}, provider);
  EXPECT_TRUE(result.ok()) << result.status();
  return result.ok() ? *result : nullptr;
}

void ExpectStatus(const arrow::Status& status, LoonSourceCredentialStatus expected) {
  auto detail = std::dynamic_pointer_cast<fs::SourceCredentialStatusDetail>(status.detail());
  ASSERT_TRUE(detail) << status;
  EXPECT_EQ(detail->status(), expected);
}

TEST(SourceFileSystemTest, MinimalConfigResolvesAliasesWithoutLegacyDefaultsAndDecodesOnce) {
  auto provider = std::make_shared<DenyingProvider>();
  auto context = Context(provider);
  auto& cache = FilesystemCache::getInstance();
  auto first = cache.get_with_context(MinimalProperties(), kFile, context);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->source_uri, kFile);
  EXPECT_EQ(first->path, kPath);
  for (const auto* uri :
       {"https://acct.dfs.core.windows.net/tables/team/table/data.parquet",
        "abfss://tables@acct.dfs.core.windows.net/team/table/data.parquet",
        "azure://core.windows.net/tables/team/table/data.parquet", "az://tables/team/table/data.parquet"}) {
    auto alias = cache.get_with_context(MinimalProperties(), uri, context);
    ASSERT_TRUE(alias.ok()) << alias.status();
    EXPECT_EQ(alias->source_uri, first->source_uri);
    EXPECT_EQ(alias->path, first->path);
    EXPECT_EQ(alias->cache_identity, first->cache_identity);
    EXPECT_EQ(alias->filesystem, first->filesystem);
  }
  auto encoded = cache.get_with_context(
      MinimalProperties(), "https://acct.blob.core.windows.net/tables/team/table/a%2520%2B%20%C3%A9.parquet", context);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(encoded->path, "team/table/a%20+ é.parquet");
  EXPECT_EQ(provider->calls, 0);
}

TEST(SourceFileSystemTest, NativeUriBindingValidatesAuthorityBeforeReturningDecodedKey) {
  auto provider = std::make_shared<DenyingProvider>();
  auto context = Context(provider);
  auto source = FilesystemCache::getInstance().get_with_context(MinimalProperties(), kFile, context);
  ASSERT_TRUE(source.ok()) << source.status();
  ASSERT_TRUE(std::dynamic_pointer_cast<SourceFileSystemProxy>(source->filesystem));
  auto encoded = source->filesystem->PathFromUri(std::string("https://acct.blob.core.windows.net/tables/team/table") +
                                                 "/a%2520%2B%20%C3%A9.parquet");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(*encoded, "team/table/a%20+ é.parquet");
  for (const auto* uri : {kFile, "abfss://tables@acct.dfs.core.windows.net/team/table/data.parquet",
                          "https://acct.dfs.core.windows.net/tables/team/table/data.parquet"}) {
    auto result = source->filesystem->PathFromUri(uri);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(*result, kPath);
  }
  for (const auto* uri :
       {"abfss://tables@other.dfs.core.windows.net/team/table/data.parquet",
        "https://acct.blob.core.windows.net/other/team/table/data.parquet",
        "https://acct.blob.core.windows.net/tables/team/other/data.parquet",
        "https://acct.blob.core.windows.net/tables/team/table/data.parquet?sig=hidden", "team/table/data.parquet"}) {
    auto result = source->filesystem->PathFromUri(uri);
    ExpectStatus(result.status(), LOON_SOURCE_CREDENTIAL_DENIED);
    EXPECT_EQ(result.status().ToString().find("hidden"), std::string::npos);
  }
  context->Cancel();
  ExpectStatus(source->filesystem->PathFromUri(kFile).status(), LOON_SOURCE_CREDENTIAL_CANCELLED);
  EXPECT_EQ(provider->calls, 0);
}

TEST(SourceFileSystemTest, ConcurrentReuseIsIsolatedFromOtherContextsAndDoesNotRetainEndedContext) {
  auto provider = std::make_shared<DenyingProvider>();
  auto context = Context(provider);
  std::weak_ptr<SourceReadContext> weak = context;
  auto& cache = FilesystemCache::getInstance();
  auto first = cache.get_with_context(MinimalProperties(), kFile, context);
  ASSERT_TRUE(first.ok());
  std::vector<std::future<arrow::Result<SourceReadFileSystem>>> workers;
  for (int i = 0; i < 10; ++i)
    workers.emplace_back(
        std::async(std::launch::async, [&] { return cache.get_with_context(MinimalProperties(), kFile, context); }));
  for (auto& worker : workers) {
    auto same = worker.get();
    ASSERT_TRUE(same.ok());
    EXPECT_EQ(same->filesystem, first->filesystem);
  }
  auto other = cache.get_with_context(MinimalProperties(), kFile, Context(provider));
  ASSERT_TRUE(other.ok());
  EXPECT_NE(first->filesystem, other->filesystem);
  EXPECT_NE(first->cache_identity, other->cache_identity);
  context.reset();
  EXPECT_FALSE(weak.expired());
  first->filesystem.reset();
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(provider->calls, 0);
}

TEST(SourceFileSystemTest, RejectsRelativeUnmatchedAndOutOfScopeWithoutPrimaryFilesystemFallback) {
  auto provider = std::make_shared<DenyingProvider>();
  auto context = Context(provider);
  auto props = MinimalProperties();
  (*props)[PROPERTY_FS_STORAGE_TYPE] = std::string("local");
  (*props)[PROPERTY_FS_ROOT_PATH] = std::string("/tmp");
  auto& cache = FilesystemCache::getInstance();
  for (const auto* uri : {kPath, "", "/tmp/data", "https://other.blob.core.windows.net/tables/team/table/data",
                          "https://acct.blob.core.windows.net/tables/team/other/data",
                          "https://acct.blob.core.windows.net/tables/team/table/a%2Fb",
                          "https://acct.blob.core.windows.net/tables/team/table/%2e%2e/other",
                          "azure://attacker.example/tables/team/table/data"}) {
    ExpectStatus(cache.get_with_context(props, uri, context).status(), LOON_SOURCE_CREDENTIAL_DENIED);
  }
  ExpectStatus(cache.get_with_context({}, kFile, context).status(), LOON_SOURCE_CREDENTIAL_DENIED);
  ExpectStatus(cache.get_with_context(props, kFile, nullptr).status(), LOON_SOURCE_CREDENTIAL_DENIED);
  auto active = cache.get_with_context(props, kFile, context);
  ASSERT_TRUE(active.ok());
  context->Cancel();
  ExpectStatus(cache.get_with_context(props, kFile, context).status(), LOON_SOURCE_CREDENTIAL_CANCELLED);
  EXPECT_EQ(provider->calls, 0);
}

TEST(SourceFileSystemTest, RejectsConflictingCredentialModesAndExplicitBindingOverrides) {
  auto provider = std::make_shared<DenyingProvider>();
  auto context = Context(provider);
  for (const auto& [key, value] :
       std::vector<std::pair<std::string, std::string>>{{"access_key_value", "secret"},
                                                        {"use_iam", "true"},
                                                        {"use_ssl", "false"},
                                                        {"storage_type", "local"},
                                                        {"credential_endpoint", "https://broker.example"},
                                                        {"azure_client_id", "client"},
                                                        {"address", "attacker.example"},
                                                        {"bucket_name", "other"}}) {
    SCOPED_TRACE(key);
    auto props = MinimalProperties();
    (*props)["extfs.catalog." + key] = value;
    ExpectStatus(FilesystemCache::getInstance().get_with_context(props, kFile, context).status(),
                 LOON_SOURCE_CREDENTIAL_DENIED);
  }
  EXPECT_EQ(provider->calls, 0);
}

struct CallbackState {
  LoonSourceCredentialStatus status = LOON_SOURCE_CREDENTIAL_DENIED;
  std::atomic<int> calls{0}, releases{0}, destroyed{0};
};

LoonSourceReadContextHandle CContext(CallbackState& state) {
  const char* roots[] = {kRoot};
  LoonSourceReadContextOptions options{1, sizeof(LoonSourceReadContextOptions), "acct", roots, 1, 0, 3000};
  LoonSourceCredentialCallbacks callbacks{
      1,
      sizeof(LoonSourceCredentialCallbacks),
      &state,
      [](void* data, const LoonSourceCredentialRequest* request, LoonSourceCredentialResult* result) {
        auto& state = *static_cast<CallbackState*>(data);
        ++state.calls;
        EXPECT_EQ(std::string(request->scope_root_uri, request->scope_root_uri_length), kRoot);
        result->status = state.status;
      },
      [](void* data, LoonSourceCredentialResult*) { ++static_cast<CallbackState*>(data)->releases; },
      [](void* data) { ++static_cast<CallbackState*>(data)->destroyed; }};
  LoonSourceReadContextHandle handle = 0;
  auto result = loon_source_read_context_create(&options, &callbacks, &handle);
  EXPECT_EQ(result.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&result);
  return handle;
}

struct CProperties {
  LoonProperties value{};
  CProperties() {
    const char* keys[] = {"extfs.catalog.cloud_provider", "extfs.catalog.access_key_id"};
    const char* values[] = {"azure", "acct"};
    auto result = loon_properties_create(keys, values, 2, &value);
    EXPECT_EQ(result.err_code, LOON_SUCCESS);
    loon_ffi_free_result(&result);
  }
  ~CProperties() { loon_properties_free(&value); }
};

void CheckResult(LoonFFIResult result, int expected) {
  EXPECT_EQ(result.err_code, expected) << (result.message ? result.message : "");
  loon_ffi_free_result(&result);
}

FileSystemHandle CFileSystem(const CProperties& properties, LoonSourceReadContextHandle context) {
  FileSystemHandle fs = 0;
  CheckResult(loon_filesystem_get_with_context(&properties.value, kFile, std::strlen(kFile), context, &fs),
              LOON_SUCCESS);
  EXPECT_NE(fs, 0);
  return fs;
}

TEST(SourceFileSystemTest, ActualCAbiPreservesProviderStatusAndRetryabilityAcrossMetadataListAndReads) {
  CProperties properties;
  const std::pair<LoonSourceCredentialStatus, int> failures[] = {
      {LOON_SOURCE_CREDENTIAL_DENIED, LOON_SOURCE_AUTH_DENIED},
      {LOON_SOURCE_CREDENTIAL_EXPIRED, LOON_SOURCE_AUTH_EXPIRED},
      {LOON_SOURCE_CREDENTIAL_TRANSIENT, LOON_SOURCE_AUTH_TRANSIENT},
      {LOON_SOURCE_CREDENTIAL_CANCELLED, LOON_SOURCE_AUTH_CANCELLED},
      {LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE, LOON_SOURCE_AUTH_INVALID_RESPONSE}};
  for (const auto& [failure, code] : failures) {
    for (int operation = 0; operation < 3; ++operation) {
      SCOPED_TRACE(failure);
      SCOPED_TRACE(operation);
      CallbackState state;
      state.status = failure;
      const auto context = CContext(state);
      auto fs = CFileSystem(properties, context);
      EXPECT_EQ(state.calls, 0);
      loon_source_read_context_release(context);
      EXPECT_EQ(state.destroyed, 0);
      if (operation == 0) {
        uint64_t size = 0;
        CheckResult(loon_filesystem_get_file_info(fs, kPath, std::strlen(kPath), &size), code);
      } else if (operation == 1) {
        LoonFileInfoList list{};
        CheckResult(loon_filesystem_list_dir(fs, "team/table", 10, true, &list), code);
        loon_filesystem_free_file_info_list(&list);
      } else {
        FileSystemReaderHandle reader = 0;
        CheckResult(loon_filesystem_open_reader(fs, kPath, std::strlen(kPath), 16, &reader), LOON_SUCCESS);
        loon_filesystem_destroy(fs);
        fs = 0;
        EXPECT_EQ(state.destroyed, 0);
        uint8_t bytes[4];
        CheckResult(loon_filesystem_reader_readat(reader, 0, 4, bytes), code);
        CheckResult(loon_filesystem_reader_close(reader), LOON_SUCCESS);
        loon_filesystem_reader_destroy(reader);
      }
      if (fs)
        loon_filesystem_destroy(fs);
      EXPECT_EQ(state.destroyed, 1);
      EXPECT_EQ(state.calls, failure == LOON_SOURCE_CREDENTIAL_TRANSIENT ? 4 : 1);
      EXPECT_EQ(state.releases, state.calls);
      EXPECT_EQ(loon_ffi_is_retryable_errcode(code), failure == LOON_SOURCE_CREDENTIAL_TRANSIENT);
      auto native = fs::SourceCredentialArrowStatus(failure);
      EXPECT_EQ(ToSegcoreError(native).get_error_code(),
                failure == LOON_SOURCE_CREDENTIAL_TRANSIENT ? milvus::StorageTransientError : milvus::StorageError);
    }
  }
}

TEST(SourceFileSystemTest, ReleasedOrCancelledCHandleCannotFallBackAndSourceErrorsDoNotEchoPaths) {
  CProperties properties;
  CallbackState state;
  const auto context = CContext(state);
  auto fs = CFileSystem(properties, context);
  const char* path = "team/table/data?sig=must-not-appear";
  uint64_t size = 0;
  auto result = loon_filesystem_get_file_info(fs, path, std::strlen(path), &size);
  EXPECT_EQ(result.err_code, LOON_SOURCE_AUTH_DENIED);
  ASSERT_NE(result.message, nullptr);
  EXPECT_EQ(std::string(result.message).find("must-not-appear"), std::string::npos);
  loon_ffi_free_result(&result);
  loon_filesystem_destroy(fs);
  loon_source_read_context_release(context);
  fs = 42;
  result = loon_filesystem_get_with_context(&properties.value, kFile, std::strlen(kFile), context, &fs);
  EXPECT_EQ(fs, 0);
  CheckResult(result, LOON_SOURCE_AUTH_DENIED);
  EXPECT_EQ(state.destroyed, 1);
  CallbackState cancelled;
  const auto handle = CContext(cancelled);
  fs = CFileSystem(properties, handle);
  loon_source_read_context_cancel(handle);
  size = 0;
  CheckResult(loon_filesystem_get_file_info(fs, kPath, std::strlen(kPath), &size), LOON_SOURCE_AUTH_CANCELLED);
  EXPECT_EQ(cancelled.calls, 0);
  loon_source_read_context_release(handle);
  loon_filesystem_destroy(fs);
  EXPECT_EQ(cancelled.destroyed, 1);
}

}  // namespace
}  // namespace milvus_storage
