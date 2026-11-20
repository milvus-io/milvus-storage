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

#include "milvus-storage/source_reader.h"
#include "milvus-storage/format/source_format_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/format/lance/lance_table_reader.h"

#include <atomic>
#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include "milvus-storage/ffi_internal/bridge.h"
#include <gtest/gtest.h>

namespace milvus_storage {
namespace {
constexpr auto kRoot = "https://acct.blob.core.windows.net/tables/team/table";
constexpr auto kFile = "https://acct.blob.core.windows.net/tables/team/table/data.parquet";

class DenyingProvider final : public SourceCredentialProvider {
  public:
  mutable std::atomic<int> calls{0};
  LoonSourceCredentialStatus failure = LOON_SOURCE_CREDENTIAL_DENIED;
  SourceCredentialResponse Fetch(const SourceCredentialRequest&) const override {
    ++calls;
    SourceCredentialResponse response;
    response.status = failure;
    return response;
  }
};

class SourceReaderTest : public ::testing::Test {
  protected:
  std::shared_ptr<DenyingProvider> provider = std::make_shared<DenyingProvider>();
  std::shared_ptr<arrow::Schema> schema = arrow::schema({arrow::field("value", arrow::int64())});
  SourceReadContextPtr Context() {
    auto result = SourceReadContext::Create({"acct", {kRoot}}, provider);
    EXPECT_TRUE(result.ok()) << result.status();
    return result.ok() ? *result : nullptr;
  }
  std::shared_ptr<const api::Properties> Properties(bool cached = true) {
    return std::make_shared<const api::Properties>(
        api::Properties{{"extfs.catalog.cloud_provider", std::string("azure")},
                        {"extfs.catalog.access_key_id", std::string("acct")},
                        {PROPERTY_READER_METADATA_CACHE_ENABLE, cached}});
  }
  std::shared_ptr<api::ColumnGroups> Groups(bool empty = false) {
    auto groups = std::make_shared<api::ColumnGroups>();
    if (!empty)
      groups->push_back(
          std::make_shared<api::ColumnGroup>(api::ColumnGroup{{"value"}, LOON_FORMAT_PARQUET, {{kFile, 0, 2, {}}}}));
    return groups;
  }
  void ExpectSource(const arrow::Status& status, int code) {
    EXPECT_EQ(FFIErrorCodeFromExtendStatus(status), code) << status;
  }
};

TEST_F(SourceReaderTest, NullContextDoesNotSelectLegacyReader) {
  auto result = api::Reader::create_with_context(Groups(), schema, nullptr, Properties(), nullptr);
  ExpectSource(result.status(), LOON_SOURCE_AUTH_DENIED);
  EXPECT_EQ(provider->calls, 0);
}

TEST_F(SourceReaderTest, EmptyResultsAndBufferedStreamsRetainAndCheckContext) {
  auto context = Context();
  std::weak_ptr<SourceReadContext> retained = context;
  auto result = api::Reader::create_with_context(Groups(true), schema, nullptr, Properties(), context);
  ASSERT_TRUE(result.ok()) << result.status();
  auto reader = std::move(*result);
  auto stream = reader->get_record_batch_reader();
  ASSERT_TRUE(stream.ok()) << stream.status();
  ASSERT_TRUE(reader->take({}).ok());
  context->Cancel();
  ExpectSource(reader->take({}).status(), LOON_SOURCE_AUTH_CANCELLED);
  reader.reset();
  context.reset();
  EXPECT_FALSE(retained.expired());
  std::shared_ptr<arrow::RecordBatch> batch;
  ExpectSource((*stream)->ReadNext(&batch), LOON_SOURCE_AUTH_CANCELLED);
  EXPECT_FALSE(batch);
  ASSERT_TRUE((*stream)->Close().ok());
  *stream = nullptr;
  EXPECT_TRUE(retained.expired());
  EXPECT_EQ(provider->calls, 0);
}

TEST_F(SourceReaderTest, DeferredTakeOwnsReaderAndChecksCancellationAtConsumption) {
  auto context = Context();
  auto result = api::Reader::create_with_context(Groups(true), schema, nullptr, Properties(), context);
  ASSERT_TRUE(result.ok());
  auto reader = std::move(*result);
  auto pending = reader->take_async({});
  reader.reset();
  context->Cancel();
  ExpectSource(std::move(pending).get().status(), LOON_SOURCE_AUTH_CANCELLED);
  EXPECT_EQ(provider->calls, 0);
}

TEST_F(SourceReaderTest, CacheStateIsRetainedByCopiesAndIsolatedAcrossContexts) {
  auto context = Context();
  MetadataCache first(true, context);
  auto copy = first;
  MetadataCache other(true, Context());
  using Reader = parquet::ParquetFormatReader;
  EXPECT_EQ(first.get<Reader>(), copy.get<Reader>());
  EXPECT_NE(first.get<Reader>(), other.get<Reader>());
  EXPECT_EQ(copy.source_context(), context);
  context->Cancel();
  api::ColumnGroupFile file{kFile, 0, 2, {}};
  auto result =
      OpenSourceFormatReader<Reader>(copy, LOON_FORMAT_PARQUET, file, Properties(), schema, {"value"}, {}, "");
  ExpectSource(result.status(), LOON_SOURCE_AUTH_CANCELLED);
  EXPECT_EQ(provider->calls, 0);
}

TEST_F(SourceReaderTest, NativeReaderRoutesAllFactoriesThroughSourceAuthorizationWithAndWithoutCache) {
  for (const bool cached : {false, true}) {
    for (const int operation : {0, 1, 2, 3, 4}) {
      SCOPED_TRACE(cached);
      SCOPED_TRACE(operation);
      provider->calls = 0;
      auto result = api::Reader::create_with_context(Groups(), schema, nullptr, Properties(cached), Context());
      ASSERT_TRUE(result.ok()) << result.status();
      auto reader = std::move(*result);
      arrow::Status status;
      if (operation == 0)
        status = reader->get_record_batch_reader().status();
      if (operation == 1)
        status = reader->get_chunk_reader(0).status();
      if (operation == 2)
        status = reader->take({0}).status();
      if (operation == 3)
        status = std::move(reader->get_chunk_reader_async(0)).get().status();
      if (operation == 4)
        status = std::move(reader->take_async({0})).get().status();
      ExpectSource(status, LOON_SOURCE_AUTH_DENIED);
      EXPECT_EQ(provider->calls, 1);
    }
  }
}

TEST_F(SourceReaderTest, LanceLocationRetainsScopeEncodingAndIsolatesSchedulerIdentity) {
  auto context = Context();
  const auto uri = std::string(kRoot) + "/a%2520%2B%20%C3%A9.lance";
  auto first = lance::ResolveSourceLanceLocation(uri, Properties(), context);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->lance_uri, "az://tables/team/table/a%2520%2B%20%C3%A9.lance");
  EXPECT_EQ(first->source.path, "team/table/a%20+ é.lance");
  EXPECT_EQ(first->options.at("milvus_fs_cache_key"), first->source.cache_identity);
  EXPECT_EQ(first->options.at("milvus_fs_is_local"), "false");
  auto same = lance::ResolveSourceLanceLocation(uri, Properties(), context);
  ASSERT_TRUE(same.ok()) << same.status();
  EXPECT_EQ(first->source.filesystem, same->source.filesystem);
  EXPECT_EQ(first->options, same->options);
  auto other = lance::ResolveSourceLanceLocation(uri, Properties(), Context());
  ASSERT_TRUE(other.ok()) << other.status();
  EXPECT_NE(first->source.filesystem, other->source.filesystem);
  EXPECT_NE(first->options.at("milvus_fs_cache_key"), other->options.at("milvus_fs_cache_key"));
  context->Cancel();
  ExpectSource(lance::ResolveSourceLanceLocation(uri, Properties(), context).status(), LOON_SOURCE_AUTH_CANCELLED);
  EXPECT_EQ(provider->calls, 0);
}

TEST_F(SourceReaderTest, LanceRejectsWrongAuthorityAndAmbiguousFragmentBeforeProvider) {
  auto format = Format::get(LOON_FORMAT_LANCE_TABLE);
  ASSERT_TRUE(format.ok());
  for (const auto* path : {"https://other.blob.core.windows.net/tables/team/table?fragment_id=1",
                           "https://acct.blob.core.windows.net/tables/team/other?fragment_id=1",
                           "https://acct.blob.core.windows.net/tables/team/table?fragment_id=1&sig=hidden",
                           "https://acct.blob.core.windows.net/tables/team/table?fragment_id=-1",
                           "https://acct.blob.core.windows.net/tables/team/table?fragment_id=18446744073709551616"}) {
    api::ColumnGroupFile file{path, 0, 2, {}};
    auto opened = (*format)->create_reader_with_context(schema, file, Properties(), {"value"}, {}, Context());
    ExpectSource(opened.status(), LOON_SOURCE_AUTH_DENIED);
    EXPECT_EQ(opened.status().ToString().find("hidden"), std::string::npos);
  }
  EXPECT_EQ(provider->calls, 0);
}

TEST_F(SourceReaderTest, LanceReaderUsesSourceProviderThroughNativeRustBridge) {
  for (auto failure : {LOON_SOURCE_CREDENTIAL_DENIED, LOON_SOURCE_CREDENTIAL_EXPIRED, LOON_SOURCE_CREDENTIAL_TRANSIENT,
                       LOON_SOURCE_CREDENTIAL_CANCELLED, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE}) {
    for (bool cached : {false, true}) {
      for (bool asynchronous : {false, true}) {
        SCOPED_TRACE(failure);
        SCOPED_TRACE(cached);
        SCOPED_TRACE(asynchronous);
        auto groups = Groups();
        (*groups)[0]->format = LOON_FORMAT_LANCE_TABLE;
        (*groups)[0]->files[0].path = lance::MakeLanceUri(kRoot, 0);
        (*groups)[0]->files[0].Set(lance::kDatasetVersionProperty, uint64_t{1});
        provider->calls = 0;
        provider->failure = failure;
        auto result = api::Reader::create_with_context(groups, schema, nullptr, Properties(cached), Context());
        ASSERT_TRUE(result.ok()) << result.status();
        auto status = asynchronous ? std::move((*result)->get_chunk_reader_async(0)).get().status()
                                   : (*result)->get_chunk_reader(0).status();
        ExpectSource(status, 116 + static_cast<int>(failure));
        EXPECT_EQ(provider->calls, failure == LOON_SOURCE_CREDENTIAL_TRANSIENT ? 4 : 1);
      }
    }
  }
}

TEST_F(SourceReaderTest, LanceRustAdapterValidatesCompleteUrisBeforeCredentialAcquisition) {
  for (const auto* uri :
       {"az://hidden@tables/team/table", "az://tables:443/team/table", "az://tables/team/table?sig=hidden",
        "az://tables/team/table#hidden", "az://other/team/table", "az://tables/team/other"}) {
    SCOPED_TRACE(uri);
    auto location = lance::ResolveSourceLanceLocation(kRoot, Properties(), Context());
    ASSERT_TRUE(location.ok()) << location.status();
    provider->calls = 0;
    auto opened = lance::BlockingDataset::Open(uri, location->source.filesystem, location->options, 1);
    ExpectSource(opened.status(), LOON_SOURCE_AUTH_DENIED);
    EXPECT_EQ(provider->calls, 0);
    EXPECT_EQ(opened.status().ToString().find("hidden"), std::string::npos);
  }
}

TEST_F(SourceReaderTest, IcebergPlannerPreservesEveryProviderFailureThroughRustAndCxx) {
  auto format = Format::get(LOON_FORMAT_ICEBERG_TABLE);
  ASSERT_TRUE(format.ok());
  for (auto failure : {LOON_SOURCE_CREDENTIAL_DENIED, LOON_SOURCE_CREDENTIAL_EXPIRED, LOON_SOURCE_CREDENTIAL_TRANSIENT,
                       LOON_SOURCE_CREDENTIAL_CANCELLED, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE}) {
    SCOPED_TRACE(failure);
    provider->calls = 0;
    provider->failure = failure;
    auto result =
        (*format)->explore_with_context(std::string(kRoot) + "/metadata/v1.metadata.json", Properties(), Context());
    ExpectSource(result.status(), 116 + static_cast<int>(failure));
    EXPECT_EQ(provider->calls, failure == LOON_SOURCE_CREDENTIAL_TRANSIENT ? 4 : 1);
  }
}

TEST_F(SourceReaderTest, IcebergReaderUsesSourceContextWithAndWithoutMetadataCache) {
  for (bool cached : {false, true}) {
    for (bool asynchronous : {false, true}) {
      SCOPED_TRACE(cached);
      SCOPED_TRACE(asynchronous);
      provider->calls = 0;
      auto groups = Groups();
      (*groups)[0]->format = LOON_FORMAT_ICEBERG_TABLE;
      auto result = api::Reader::create_with_context(groups, schema, nullptr, Properties(cached), Context());
      ASSERT_TRUE(result.ok()) << result.status();
      auto status = asynchronous ? std::move((*result)->get_chunk_reader_async(0)).get().status()
                                 : (*result)->get_chunk_reader(0).status();
      ExpectSource(status, LOON_SOURCE_AUTH_DENIED);
      EXPECT_EQ(provider->calls, 1);
    }
  }
}

TEST_F(SourceReaderTest, CReaderRetainsContextAfterHandleReleaseAndPreservesProviderFailures) {
  for (auto failure : {LOON_SOURCE_CREDENTIAL_DENIED, LOON_SOURCE_CREDENTIAL_EXPIRED, LOON_SOURCE_CREDENTIAL_TRANSIENT,
                       LOON_SOURCE_CREDENTIAL_CANCELLED, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE}) {
    struct State {
      LoonSourceCredentialStatus failure;
      int calls = 0, releases = 0, destroyed = 0;
    } state{failure};
    const char* roots[] = {kRoot};
    LoonSourceReadContextOptions options{1, sizeof(LoonSourceReadContextOptions), "acct", roots, 1, 0, 3000};
    LoonSourceCredentialCallbacks callbacks{
        1,
        sizeof(LoonSourceCredentialCallbacks),
        &state,
        [](void* data, const LoonSourceCredentialRequest*, LoonSourceCredentialResult* result) {
          auto& state = *static_cast<State*>(data);
          ++state.calls;
          result->status = state.failure;
        },
        [](void* data, LoonSourceCredentialResult*) { ++static_cast<State*>(data)->releases; },
        [](void* data) { ++static_cast<State*>(data)->destroyed; }};
    auto check = [](LoonFFIResult result, int expected) {
      EXPECT_EQ(result.err_code, expected) << (result.message ? result.message : "");
      loon_ffi_free_result(&result);
    };
    LoonSourceReadContextHandle context = 0;
    check(loon_source_read_context_create(&options, &callbacks, &context), LOON_SUCCESS);
    ASSERT_NE(context, 0);
    LoonColumnGroups* groups = nullptr;
    ASSERT_TRUE(column_groups_export(*Groups(), &groups).ok());
    ArrowSchema exported{};
    ASSERT_TRUE(arrow::ExportSchema(*schema, &exported).ok());
    const char* keys[] = {"extfs.catalog.cloud_provider", "extfs.catalog.access_key_id"};
    const char* values[] = {"azure", "acct"};
    LoonProperties properties{};
    check(loon_properties_create(keys, values, 2, &properties), LOON_SUCCESS);
    LoonReaderHandle reader = 0;
    check(loon_reader_new_with_context(groups, &exported, nullptr, 0, &properties, context, &reader), LOON_SUCCESS);
    ASSERT_NE(reader, 0);
    EXPECT_EQ(state.calls, 0);
    loon_source_read_context_release(context);
    loon_column_groups_destroy(groups);
    loon_properties_free(&properties);
    EXPECT_EQ(state.destroyed, 0);
    LoonChunkReaderHandle chunk = 0;
    check(loon_get_chunk_reader(reader, 0, nullptr, 0, &chunk), 116 + static_cast<int>(failure));
    EXPECT_EQ(chunk, 0);
    EXPECT_EQ(state.calls, failure == LOON_SOURCE_CREDENTIAL_TRANSIENT ? 4 : 1);
    EXPECT_EQ(state.calls, state.releases);
    loon_reader_destroy(reader);
    EXPECT_EQ(state.destroyed, 1);
  }
}

TEST_F(SourceReaderTest, ContextSurvivesCacheResetForEncryptionKeyRetriever) {
  auto context = Context();
  auto result = api::Reader::create_with_context(Groups(), schema, nullptr, Properties(), context);
  ASSERT_TRUE(result.ok());
  auto reader = std::move(*result);
  reader->set_keyretriever([](const std::string&) { return std::string(); });
  ExpectSource(reader->get_chunk_reader(0).status(), LOON_SOURCE_AUTH_DENIED);
  EXPECT_EQ(provider->calls, 1);
}
}  // namespace
}  // namespace milvus_storage
