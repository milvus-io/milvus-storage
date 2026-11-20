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

#include "milvus-storage/format/source_format_reader.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format.h"
#include "iceberg/iceberg_bridge.h"
#include "milvus-storage/format/lance/lance_table_writer.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/filesystem/azure/azurefs.h"
#include "milvus-storage/filesystem/azure/source_sas_token_policy.h"
#include "milvus-storage/ffi_internal/ffi_error_code.h"
#include "milvus-storage/ffi_internal/result.h"

#include <atomic>
#include <deque>
#include <filesystem>
#include <mutex>
#include <random>
#include <unordered_set>
#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/io/file.h>
#include <arrow/util/io_util.h>
#include <parquet/arrow/writer.h>
#include <azure/core/datetime.hpp>
#include <azure/core/io/body_stream.hpp>
#include <gtest/gtest.h>

namespace milvus_storage {
class SourceFilesystemTestAccess {
  public:
  static arrow::Status Install(const std::shared_ptr<const api::Properties>& properties,
                               const std::string& uri,
                               SourceReadContextPtr context,
                               ArrowFileSystemPtr filesystem) {
    auto& cache = FilesystemCache::getInstance();
    ARROW_ASSIGN_OR_RAISE(auto resolved, cache.get_with_context(properties, uri, context));
    std::lock_guard lock(cache.mutex_);
    cache.source_cache_[resolved.cache_identity] = std::move(filesystem);
    return arrow::Status::OK();
  }
};
namespace {
constexpr auto kRoot = "https://acct.blob.core.windows.net/tables/team/table";
constexpr auto kFile = "https://acct.blob.core.windows.net/tables/team/table/data.parquet";
constexpr int64_t kNow = 1893456000000;

class Provider final : public SourceCredentialProvider {
  public:
  mutable std::atomic<int> calls{0};
  std::atomic<int64_t> now{kNow};
  LoonSourceCredentialStatus failure = LOON_SOURCE_CREDENTIAL_OK;
  SourceCredentialResponse Fetch(const SourceCredentialRequest& request) const override {
    const auto count = ++calls;
    if (failure != LOON_SOURCE_CREDENTIAL_OK) {
      SourceCredentialResponse result;
      result.status = failure;
      return result;
    }
    const auto expires = now.load() + 120000;
    auto expiry = Azure::DateTime(std::chrono::system_clock::time_point(std::chrono::milliseconds(expires)));
    return {LOON_SOURCE_CREDENTIAL_OK,
            LOON_SOURCE_CREDENTIAL_KIND_AZURE_SAS,
            "acct",
            request.scope_root_uri,
            "sv=2023-11-03&sr=d&sdd=2&sp=rl&se=" + expiry.ToString(Azure::DateTime::DateFormat::Rfc3339) +
                "&sig=lease" + std::to_string(count),
            expires};
  }
};

SourceReadContextPtr Context(const std::shared_ptr<Provider>& provider) {
  auto result = SourceReadContext::Create({"acct", {kRoot}}, provider, [provider] { return provider->now.load(); });
  EXPECT_TRUE(result.ok()) << result.status();
  return result.ok() ? *result : nullptr;
}
void ExpectSource(const arrow::Status& status, LoonSourceCredentialStatus expected) {
  auto detail = std::dynamic_pointer_cast<fs::SourceCredentialStatusDetail>(status.detail());
  ASSERT_TRUE(detail) << status;
  EXPECT_EQ(detail->status(), expected);
}
std::shared_ptr<api::Properties> Properties() {
  return std::make_shared<api::Properties>(api::Properties{{"extfs.catalog.cloud_provider", std::string("azure")},
                                                           {"extfs.catalog.access_key_id", std::string("acct")}});
}

class ParquetTransport final : public Azure::Core::Http::HttpTransport {
  public:
  std::shared_ptr<arrow::Buffer> parquet;
  std::unordered_map<std::string, std::shared_ptr<arrow::Buffer>> files;
  std::unordered_set<std::string> missing_files;
  std::mutex mutex;
  std::deque<std::vector<uint8_t>> bodies;
  std::vector<std::string> signatures;
  std::unique_ptr<Azure::Core::Http::RawResponse> Send(Azure::Core::Http::Request& request,
                                                       const Azure::Core::Context&) override {
    std::lock_guard lock(mutex);
    signatures.push_back(request.GetUrl().GetQueryParameters().at("sig"));
    auto contents = parquet;
    if (files.empty()) {
      EXPECT_EQ(request.GetUrl().GetPath(), "tables/team/table/data.parquet");
    } else {
      auto file = files.find(request.GetUrl().GetPath());
      if (file == files.end()) {
        EXPECT_TRUE(missing_files.contains(request.GetUrl().GetPath()))
            << "Unexpected fixture path: " << request.GetUrl().GetPath();
        auto missing = std::make_unique<Azure::Core::Http::RawResponse>(
            1, 1, Azure::Core::Http::HttpStatusCode::NotFound, "fixture absent");
        missing->SetHeader("Content-Length", "0");
        missing->SetHeader("x-ms-error-code", "BlobNotFound");
        bodies.emplace_back();
        missing->SetBodyStream(std::make_unique<Azure::Core::IO::MemoryBodyStream>(bodies.back()));
        return missing;
      }
      contents = file->second;
    }
    int64_t start = 0, end = contents->size() - 1;
    // GetHeaders returns a map by value. Keep it alive while using iterators,
    // including the no-range HEAD request used to open positional-delete files.
    const auto headers = request.GetHeaders();
    auto range = headers.find("x-ms-range");
    if (range == headers.end())
      range = headers.find("Range");
    const bool partial = range != headers.end();
    if (partial) {
      const auto dash = range->second.find('-');
      start = std::stoll(range->second.substr(6, dash - 6));
      if (dash + 1 < range->second.size())
        end = std::min<int64_t>(end, std::stoll(range->second.substr(dash + 1)));
    }
    EXPECT_GE(start, 0);
    EXPECT_LT(end, contents->size());
    auto response = std::make_unique<Azure::Core::Http::RawResponse>(
        1, 1, partial ? Azure::Core::Http::HttpStatusCode::PartialContent : Azure::Core::Http::HttpStatusCode::Ok,
        "test");
    response->SetHeader("Content-Length", std::to_string(end - start + 1));
    if (partial)
      response->SetHeader("Content-Range", "bytes " + std::to_string(start) + "-" + std::to_string(end) + "/" +
                                               std::to_string(contents->size()));
    response->SetHeader("Last-Modified", "Tue, 01 Jan 2030 00:00:00 GMT");
    response->SetHeader("x-ms-creation-time", "Tue, 01 Jan 2030 00:00:00 GMT");
    response->SetHeader("ETag", "\"test\"");
    response->SetHeader("x-ms-blob-type", "BlockBlob");
    response->SetHeader("x-ms-server-encrypted", "true");
    if (request.GetMethod() == Azure::Core::Http::HttpMethod::Head)
      bodies.emplace_back();
    else
      bodies.emplace_back(contents->data() + start, contents->data() + end + 1);
    response->SetBodyStream(std::make_unique<Azure::Core::IO::MemoryBodyStream>(bodies.back()));
    return response;
  }
};

TEST(SourceFormatReaderTest, RealParquetRetainedReaderRenewsAndGuardsCachedStreamsAndAsyncReads) {
  arrow::Int64Builder builder;
  ASSERT_TRUE(builder.AppendValues({10, 11, 20, 21, 30, 31}).ok());
  auto values = builder.Finish();
  ASSERT_TRUE(values.ok());
  auto schema = arrow::schema({arrow::field("value", arrow::int64())});
  auto table = arrow::Table::Make(schema, {std::make_shared<arrow::ChunkedArray>(*values)});
  auto sink = arrow::io::BufferOutputStream::Create();
  ASSERT_TRUE(sink.ok());
  ASSERT_TRUE(::parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink, 2).ok());
  auto buffer = (*sink)->Finish();
  ASSERT_TRUE(buffer.ok());
  auto transport = std::make_shared<ParquetTransport>();
  transport->parquet = *buffer;
  auto provider = std::make_shared<Provider>();
  auto context = Context(provider);
  std::weak_ptr<SourceReadContext> retained = context;
  fs::AzureOptions options;
  options.account_name = "acct";
  ASSERT_TRUE(options.ConfigureSourceReadContext(context, transport).ok());
  auto filesystem = fs::AzureFileSystem::Make(options);
  ASSERT_TRUE(filesystem.ok());
  options = fs::AzureOptions{};
  auto raw = std::make_shared<parquet::ParquetFormatReader>(
      *filesystem, "tables/team/table/data.parquet", std::make_shared<api::Properties>(),
      std::vector<std::string>{"value"}, nullptr, (*buffer)->size(), 0);
  auto reader = GuardSourceFormatReader(raw, context);
  ASSERT_TRUE(reader->open().ok());
  auto clone = reader->clone_reader();
  ASSERT_TRUE(clone.ok());
  raw.reset();
  reader.reset();
  filesystem = std::shared_ptr<fs::AzureFileSystem>();
  for (int group = 0; group < 3; ++group) {
    auto batch = (*clone)->get_chunk(group);
    ASSERT_TRUE(batch.ok()) << batch.status();
    ASSERT_EQ((*batch)->num_rows(), 2);
    auto array = std::static_pointer_cast<arrow::Int64Array>((*batch)->column(0));
    EXPECT_EQ(array->Value(0), (group + 1) * 10);
    provider->now.fetch_add(121000);
  }
  EXPECT_GE(provider->calls.load(), 3);
  auto stream = (*clone)->read_with_range(0, 2);
  ASSERT_TRUE(stream.ok()) << stream.status();
  auto pending = (*clone)->take_async({0});
  context->Cancel();
  ExpectSource(std::move(pending).get().status(), LOON_SOURCE_CREDENTIAL_CANCELLED);
  ExpectSource((*clone)->get_row_group_infos().status(), LOON_SOURCE_CREDENTIAL_CANCELLED);
  clone = std::shared_ptr<FormatReader>();
  context.reset();
  EXPECT_FALSE(retained.expired());
  std::shared_ptr<arrow::RecordBatch> batch;
  ExpectSource((*stream)->ReadNext(&batch), LOON_SOURCE_CREDENTIAL_CANCELLED);
  EXPECT_FALSE(batch);
  ASSERT_TRUE((*stream)->Close().ok());
  stream = std::shared_ptr<arrow::RecordBatchReader>();
  EXPECT_TRUE(retained.expired());
}

TEST(SourceFormatReaderTest, NativeReaderReadsRealParquetAcrossRenewalRecoveryCacheAndAsyncLifetime) {
  arrow::Int64Builder builder;
  ASSERT_TRUE(builder.AppendValues({10, 11, 20, 21, 30, 31}).ok());
  auto values = builder.Finish();
  ASSERT_TRUE(values.ok());
  auto schema = arrow::schema({arrow::field("value", arrow::int64())});
  auto table = arrow::Table::Make(schema, {std::make_shared<arrow::ChunkedArray>(*values)});
  auto sink = arrow::io::BufferOutputStream::Create();
  ASSERT_TRUE(sink.ok());
  ASSERT_TRUE(::parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink, 2).ok());
  auto buffer = (*sink)->Finish();
  ASSERT_TRUE(buffer.ok());

  for (bool cached : {false, true}) {
    SCOPED_TRACE(cached);
    auto transport = std::make_shared<ParquetTransport>();
    transport->parquet = *buffer;
    auto provider = std::make_shared<Provider>();
    auto context = Context(provider);
    fs::AzureOptions options;
    options.account_name = "acct";
    ASSERT_TRUE(options.ConfigureSourceReadContext(context, transport).ok());
    auto native = fs::AzureFileSystem::Make(options);
    ASSERT_TRUE(native.ok());
    auto binding = AzureSourceStorageBinding::Create("acct", "core.windows.net", "tables");
    ASSERT_TRUE(binding);
    auto filesystem = std::make_shared<SourceFileSystemProxy>("tables", *native, context, *binding);
    auto properties = Properties();
    (*properties)[PROPERTY_READER_METADATA_CACHE_ENABLE] = cached;
    ASSERT_TRUE(SourceFilesystemTestAccess::Install(properties, kFile, context, filesystem).ok());
    auto groups = std::make_shared<api::ColumnGroups>();
    api::ColumnGroupFile file{kFile, 0, 6, {}};
    file.Set(api::kPropertyFileSize, static_cast<uint64_t>((*buffer)->size()));
    groups->push_back(std::make_shared<api::ColumnGroup>(api::ColumnGroup{{"value"}, LOON_FORMAT_PARQUET, {file}}));
    auto opened = api::Reader::create_with_context(groups, schema, nullptr, properties, context);
    ASSERT_TRUE(opened.ok()) << opened.status();
    auto reader = std::move(*opened);
    auto chunks = reader->get_chunk_reader(0);
    ASSERT_TRUE(chunks.ok()) << chunks.status();
    ASSERT_EQ((*chunks)->total_number_of_chunks(), 3);
    for (int group = 0; group < 3; ++group) {
      auto batch = (*chunks)->get_chunk(group);
      ASSERT_TRUE(batch.ok()) << batch.status();
      ASSERT_EQ((*batch)->num_rows(), 2);
      EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>((*batch)->column(0))->Value(0), (group + 1) * 10);
      if (group < 2)
        provider->now.fetch_add(121000);
    }
    EXPECT_GE(provider->calls.load(), 3);
    const auto requests_before_reopen = transport->signatures.size();
    auto reopened = std::move(reader->get_chunk_reader_async(0)).get();
    ASSERT_TRUE(reopened.ok()) << reopened.status();
    if (cached)
      EXPECT_EQ(transport->signatures.size(), requests_before_reopen);
    auto selected = std::move(reader->take_async({0, 5})).get();
    ASSERT_TRUE(selected.ok()) << selected.status();
    ASSERT_EQ((*selected)->num_rows(), 2);
    EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>((*selected)->column(0)->chunk(0))->Value(0), 10);
    auto selected_last = (*selected)->column(0)->GetScalar(1);
    ASSERT_TRUE(selected_last.ok());
    EXPECT_EQ((*selected_last)->ToString(), "31");

    provider->now.fetch_add(61000);
    provider->failure = LOON_SOURCE_CREDENTIAL_TRANSIENT;
    ASSERT_TRUE(reader->take({0}).ok());  // Still-valid cached SAS can serve fresh I/O.
    provider->now.fetch_add(60000);
    ExpectSource(reader->take({2}).status(), LOON_SOURCE_CREDENTIAL_TRANSIENT);
    provider->failure = LOON_SOURCE_CREDENTIAL_OK;
    auto recovered = reader->take({4});
    ASSERT_TRUE(recovered.ok()) << recovered.status();
    auto recovered_value = (*recovered)->column(0)->GetScalar(0);
    ASSERT_TRUE(recovered_value.ok());
    EXPECT_EQ((*recovered_value)->ToString(), "30");

    auto stream = reader->get_record_batch_reader();
    ASSERT_TRUE(stream.ok()) << stream.status();
    auto pending = reader->take_async({0});
    reader.reset();
    context->Cancel();
    ExpectSource(std::move(pending).get().status(), LOON_SOURCE_CREDENTIAL_CANCELLED);
    ExpectSource((*chunks)->get_chunk_rows().status(), LOON_SOURCE_CREDENTIAL_CANCELLED);
    std::shared_ptr<arrow::RecordBatch> batch;
    ExpectSource((*stream)->ReadNext(&batch), LOON_SOURCE_CREDENTIAL_CANCELLED);
    EXPECT_FALSE(batch);
  }
}

TEST(SourceFormatReaderTest, NativeLanceReaderRetainsContextAcrossRenewalAndIsolatesUsers) {
  auto directory = arrow::internal::TemporaryDir::Make("source-lance-");
  ASSERT_TRUE(directory.ok()) << directory.status();
  const auto local_root = (*directory)->path().ToString();
  auto local_properties = std::make_shared<api::Properties>(
      api::Properties{{PROPERTY_FS_STORAGE_TYPE, std::string("local")}, {PROPERTY_FS_ROOT_PATH, local_root}});
  arrow::Int64Builder builder;
  // Exceed the retained 64 KiB footer/block cache. Each round reads another
  // region, so a cache hit cannot be mistaken for successful credential renewal.
  constexpr int64_t rows_per_round = 65536;
  std::mt19937_64 random(42);
  std::vector<int64_t> expected(4 * rows_per_round);
  for (auto& value : expected) value = static_cast<int64_t>(random() & INT64_MAX);
  ASSERT_TRUE(builder.AppendValues(expected).ok());
  auto values = builder.Finish();
  ASSERT_TRUE(values.ok());
  auto schema = arrow::schema({arrow::field("value", arrow::int64())});
  lance::LanceTableWriter writer("fixture.lance", schema, local_properties);
  ASSERT_TRUE(writer.Write(arrow::RecordBatch::Make(schema, expected.size(), {*values})).ok());
  auto written = writer.Close();
  ASSERT_TRUE(written.ok()) << written.status();
  auto parsed = lance::ParseLanceUri(written->path);
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto transport = std::make_shared<ParquetTransport>();
  const auto local_dataset = std::filesystem::path(local_root) / "fixture.lance";
  for (const auto& entry : std::filesystem::recursive_directory_iterator(local_dataset)) {
    if (!entry.is_regular_file())
      continue;
    auto input = arrow::io::ReadableFile::Open(entry.path().string());
    ASSERT_TRUE(input.ok());
    auto contents = (*input)->Read(static_cast<int64_t>(entry.file_size()));
    ASSERT_TRUE(contents.ok());
    transport->files["tables/team/table/" + std::filesystem::relative(entry.path(), local_dataset).generic_string()] =
        *contents;
  }
  ASSERT_FALSE(transport->files.empty());
  // Lance probes the V2 manifest name before falling back to this fixture's V1 name.
  transport->missing_files.insert("tables/team/table/_versions/18446744073709551614.manifest");
  for (bool cached : {false, true}) {
    auto properties = Properties();
    (*properties)[PROPERTY_READER_METADATA_CACHE_ENABLE] = cached;
    (*properties)[PROPERTY_READER_LOGICAL_CHUNK_ROWS] = uint64_t{rows_per_round};
    auto provider = std::make_shared<Provider>();
    auto context = Context(provider);
    fs::AzureOptions options;
    options.account_name = "acct";
    ASSERT_TRUE(options.ConfigureSourceReadContext(context, transport).ok());
    auto native = fs::AzureFileSystem::Make(options);
    ASSERT_TRUE(native.ok());
    auto binding = AzureSourceStorageBinding::Create("acct", "core.windows.net", "tables");
    ASSERT_TRUE(binding);
    auto filesystem = std::make_shared<SourceFileSystemProxy>("tables", *native, context, *binding);
    ASSERT_TRUE(SourceFilesystemTestAccess::Install(properties, kRoot, context, filesystem).ok());
    auto file = *written;
    file.path = lance::MakeLanceUri(kRoot, parsed->second);
    file.Set(lance::kDatasetVersionProperty, uint64_t{1});
    auto groups = std::make_shared<api::ColumnGroups>();
    groups->push_back(std::make_shared<api::ColumnGroup>(api::ColumnGroup{{"value"}, LOON_FORMAT_LANCE_TABLE, {file}}));
    auto opened = api::Reader::create_with_context(groups, schema, nullptr, properties, context);
    ASSERT_TRUE(opened.ok()) << opened.status();
    auto reader = std::move(*opened);
    auto chunk = reader->get_chunk_reader(0);
    ASSERT_TRUE(chunk.ok()) << chunk.status();
    for (int round = 0; round < 3; ++round) {
      SCOPED_TRACE(round);
      if (round)
        provider->now.fetch_add(121000);
      auto batch = (*chunk)->get_chunk(round);
      ASSERT_TRUE(batch.ok()) << batch.status();
      ASSERT_EQ((*batch)->num_rows(), rows_per_round);
      auto actual = std::static_pointer_cast<arrow::Int64Array>((*batch)->column(0));
      EXPECT_EQ(actual->Value(0), expected[round * rows_per_round]);
      EXPECT_EQ(actual->Value(rows_per_round - 1), expected[(round + 1) * rows_per_round - 1]);
    }
    EXPECT_GE(provider->calls.load(), 3);
    provider->now.fetch_add(121000);
    provider->failure = LOON_SOURCE_CREDENTIAL_TRANSIENT;
    auto failed = (*chunk)->get_chunk(3);
    EXPECT_EQ(FFIErrorCodeFromExtendStatus(failed.status()), LOON_SOURCE_AUTH_TRANSIENT);
    provider->failure = LOON_SOURCE_CREDENTIAL_OK;
    auto recovered = (*chunk)->get_chunk(3);
    ASSERT_TRUE(recovered.ok()) << recovered.status();
    EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>((*recovered)->column(0))->Value(0),
              expected[3 * rows_per_round]);
    auto denied_provider = std::make_shared<Provider>();
    denied_provider->failure = LOON_SOURCE_CREDENTIAL_DENIED;
    auto denied_context = Context(denied_provider);
    auto denied = api::Reader::create_with_context(groups, schema, nullptr, properties, denied_context);
    ASSERT_TRUE(denied.ok());
    auto denied_result = std::move((*denied)->take_async({0})).get();
    EXPECT_EQ(FFIErrorCodeFromExtendStatus(denied_result.status()), LOON_SOURCE_AUTH_DENIED);
    EXPECT_EQ(denied_provider->calls.load(), 1);
    // Another user's denial must not poison this retained reader or its scheduler.
    ASSERT_TRUE(reader->take({0}).ok());
    auto pending = reader->take_async({0, 4 * rows_per_round - 1});
    reader.reset();
    context->Cancel();
    ExpectSource(std::move(pending).get().status(), LOON_SOURCE_CREDENTIAL_CANCELLED);
    ExpectSource((*chunk)->get_chunk(0).status(), LOON_SOURCE_CREDENTIAL_CANCELLED);
  }
}

TEST(SourceFormatReaderTest, NativeIcebergPlansFullMetadataTreeAndReadsThroughSourceContext) {
  auto directory = arrow::internal::TemporaryDir::Make("source-ice-");
  ASSERT_TRUE(directory.ok()) << directory.status();
  auto temporary = (*directory)->path().ToString();
  while (temporary.ends_with('/')) temporary.pop_back();
  const std::string remote_root = kRoot;
  ASSERT_LT(temporary.size() + 1, remote_root.size());
  const auto local_root = temporary + "/" + std::string(remote_root.size() - temporary.size() - 1, 'x');
  auto table = iceberg::CreateTestTable(local_root, 20, false, {});
  ASSERT_TRUE(table.ok()) << table.status();
  auto transport = std::make_shared<ParquetTransport>();
  size_t rewritten_files = 0;
  // The test writer emits uncompressed Avro metadata. An equal-byte-length
  // replacement preserves its string length varints and block lengths. Read
  // the resulting metadata through the actual Rust planner below, not a stub.
  for (const auto& entry : std::filesystem::recursive_directory_iterator(local_root)) {
    if (!entry.is_regular_file())
      continue;
    auto input = arrow::io::ReadableFile::Open(entry.path().string());
    ASSERT_TRUE(input.ok());
    auto bytes = (*input)->Read(static_cast<int64_t>(entry.file_size()));
    ASSERT_TRUE(bytes.ok());
    std::string contents = (*bytes)->ToString();
    if (entry.path().extension() == ".json" || entry.path().extension() == ".avro") {
      size_t count = 0;
      for (size_t pos = 0; (pos = contents.find(local_root, pos)) != std::string::npos; pos += remote_root.size()) {
        contents.replace(pos, local_root.size(), remote_root);
        ++count;
      }
      ASSERT_GT(count, 0) << entry.path().filename();
      ++rewritten_files;
    }
    transport->files["tables/team/table/" + std::filesystem::relative(entry.path(), local_root).generic_string()] =
        arrow::Buffer::FromString(std::move(contents));
  }
  ASSERT_GE(rewritten_files, 3);
  auto provider = std::make_shared<Provider>();
  auto context = Context(provider);
  fs::AzureOptions options;
  options.account_name = "acct";
  ASSERT_TRUE(options.ConfigureSourceReadContext(context, transport).ok());
  auto native = fs::AzureFileSystem::Make(options);
  ASSERT_TRUE(native.ok());
  auto binding = AzureSourceStorageBinding::Create("acct", "core.windows.net", "tables");
  ASSERT_TRUE(binding);
  auto filesystem = std::make_shared<SourceFileSystemProxy>("tables", *native, context, *binding);
  auto properties = Properties();
  const auto metadata_uri = remote_root + "/metadata/v1.metadata.json";
  ASSERT_TRUE(SourceFilesystemTestAccess::Install(properties, metadata_uri, context, filesystem).ok());
  auto format = Format::get(LOON_FORMAT_ICEBERG_TABLE);
  ASSERT_TRUE(format.ok());
  for (int round = 0; round < 3; ++round) {
    SCOPED_TRACE(round);
    if (round)
      provider->now.fetch_add(121000);
    auto files = (*format)->explore_with_context(metadata_uri, properties, context);
    ASSERT_TRUE(files.ok()) << files.status();
    ASSERT_EQ(files->size(), 1);
    EXPECT_EQ(files->front().path, remote_root + "/data/00000-0-data.parquet");
    EXPECT_EQ(files->front().end_index, 20);
    auto groups = std::make_shared<api::ColumnGroups>();
    groups->push_back(std::make_shared<api::ColumnGroup>(
        api::ColumnGroup{{"id", "name", "value"}, LOON_FORMAT_ICEBERG_TABLE, *files}));
    auto schema = arrow::schema({arrow::field("id", arrow::int64()), arrow::field("name", arrow::utf8()),
                                 arrow::field("value", arrow::float64())});
    auto reader = api::Reader::create_with_context(groups, schema, nullptr, properties, context);
    ASSERT_TRUE(reader.ok()) << reader.status();
    auto result = std::move((*reader)->take_async({0, 19})).get();
    ASSERT_TRUE(result.ok()) << result.status();
    ASSERT_EQ((*result)->num_rows(), 2);
    EXPECT_EQ((*result)->column(0)->GetScalar(1).ValueOrDie()->ToString(), "19");
  }
  EXPECT_GE(provider->calls.load(), 3);
  provider->now.fetch_add(121000);
  provider->failure = LOON_SOURCE_CREDENTIAL_TRANSIENT;
  auto failed = (*format)->explore_with_context(metadata_uri, properties, context);
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(failed.status()), LOON_SOURCE_AUTH_TRANSIENT);
  provider->failure = LOON_SOURCE_CREDENTIAL_OK;
  ASSERT_TRUE((*format)->explore_with_context(metadata_uri, properties, context).ok());

  // A manifest reference with the same container/key on another account must
  // be rejected while its original authority is still available to the planner.
  auto& manifest = transport->files.at("tables/team/table/metadata/manifest-data-0.avro");
  auto corrupted = manifest->ToString();
  const auto position = corrupted.find(remote_root);
  ASSERT_NE(position, std::string::npos);
  auto other_root = remote_root;
  other_root.replace(other_root.find("acct"), 4, "othr");
  corrupted.replace(position, remote_root.size(), other_root);
  manifest = arrow::Buffer::FromString(std::move(corrupted));
  auto denied = (*format)->explore_with_context(metadata_uri, properties, context);
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(denied.status()), LOON_SOURCE_AUTH_DENIED);
}

TEST(SourceFormatReaderTest, NativeIcebergReaderAppliesScopedDeletesAndRejectsWrongAccountRows) {
  auto encode = [](const std::shared_ptr<arrow::Table>& table) -> arrow::Result<std::shared_ptr<arrow::Buffer>> {
    ARROW_ASSIGN_OR_RAISE(auto sink, arrow::io::BufferOutputStream::Create());
    ARROW_RETURN_NOT_OK(::parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, 2));
    return sink->Finish();
  };
  arrow::Int64Builder value_builder;
  ASSERT_TRUE(value_builder.AppendValues({10, 11, 20, 21, 30, 31}).ok());
  auto values = value_builder.Finish();
  ASSERT_TRUE(values.ok());
  auto schema = arrow::schema({arrow::field("value", arrow::int64())});
  auto data = encode(arrow::Table::Make(schema, {std::make_shared<arrow::ChunkedArray>(*values)}));
  ASSERT_TRUE(data.ok());
  for (bool wrong_account : {false, true}) {
    for (bool cached : {false, true}) {
      SCOPED_TRACE(wrong_account);
      SCOPED_TRACE(cached);
      arrow::StringBuilder paths;
      ASSERT_TRUE(paths
                      .Append(wrong_account ? "abfss://tables@other.dfs.core.windows.net/team/table/data.parquet"
                                            : "abfss://tables@acct.dfs.core.windows.net/team/table/data.parquet")
                      .ok());
      ASSERT_TRUE(paths.Append(kFile).ok());
      auto path_values = paths.Finish();
      ASSERT_TRUE(path_values.ok());
      arrow::Int64Builder positions;
      ASSERT_TRUE(positions.AppendValues({1, 4}).ok());
      auto position_values = positions.Finish();
      ASSERT_TRUE(position_values.ok());
      auto delete_schema =
          arrow::schema({arrow::field("file_path", arrow::utf8()), arrow::field("pos", arrow::int64())});
      auto deletes =
          encode(arrow::Table::Make(delete_schema, {std::make_shared<arrow::ChunkedArray>(*path_values),
                                                    std::make_shared<arrow::ChunkedArray>(*position_values)}));
      ASSERT_TRUE(deletes.ok());
      auto transport = std::make_shared<ParquetTransport>();
      transport->files = {{"tables/team/table/data.parquet", *data}, {"tables/team/table/delete.parquet", *deletes}};
      auto provider = std::make_shared<Provider>();
      auto context = Context(provider);
      fs::AzureOptions options;
      options.account_name = "acct";
      ASSERT_TRUE(options.ConfigureSourceReadContext(context, transport).ok());
      auto native = fs::AzureFileSystem::Make(options);
      ASSERT_TRUE(native.ok());
      auto binding = AzureSourceStorageBinding::Create("acct", "core.windows.net", "tables");
      ASSERT_TRUE(binding);
      auto filesystem = std::make_shared<SourceFileSystemProxy>("tables", *native, context, *binding);
      auto properties = Properties();
      (*properties)[PROPERTY_READER_METADATA_CACHE_ENABLE] = cached;
      ASSERT_TRUE(SourceFilesystemTestAccess::Install(properties, kFile, context, filesystem).ok());
      api::ColumnGroupFile file{kFile, 0, 4, {}};
      file.Set(api::kPropertyFileSize, static_cast<uint64_t>((*data)->size()));
      file.Set(
          api::kPropertyMetadata,
          R"([{"file_type":"position","path":"abfss://tables@acct.dfs.core.windows.net/team/table/delete.parquet"}])");
      auto groups = std::make_shared<api::ColumnGroups>();
      groups->push_back(
          std::make_shared<api::ColumnGroup>(api::ColumnGroup{{"value"}, LOON_FORMAT_ICEBERG_TABLE, {file}}));
      auto opened = api::Reader::create_with_context(groups, schema, nullptr, properties, context);
      ASSERT_TRUE(opened.ok()) << opened.status();
      auto reader = std::move(*opened);
      auto result = reader->take({0, 1, 2, 3});
      if (wrong_account) {
        ExpectSource(result.status(), LOON_SOURCE_CREDENTIAL_DENIED);
        continue;
      }
      ASSERT_TRUE(result.ok()) << result.status();
      ASSERT_EQ((*result)->num_rows(), 4);
      for (int i = 0; i < 4; ++i) {
        auto value = (*result)->column(0)->GetScalar(i);
        ASSERT_TRUE(value.ok());
        EXPECT_EQ((*value)->ToString(), std::vector<std::string>({"10", "20", "21", "31"})[i]);
      }
      provider->now.fetch_add(121000);
      auto async_result = std::move(reader->take_async({0, 3})).get();
      ASSERT_TRUE(async_result.ok()) << async_result.status();
      ASSERT_EQ((*async_result)->num_rows(), 2);
      auto last = (*async_result)->column(0)->GetScalar(1);
      ASSERT_TRUE(last.ok());
      EXPECT_EQ((*last)->ToString(), "31");
      EXPECT_GE(provider->calls.load(), 2);
      context->Cancel();
      ExpectSource(reader->take({0}).status(), LOON_SOURCE_CREDENTIAL_CANCELLED);
    }
  }
}

TEST(SourceFormatReaderTest, ActualFactoryPreservesAllProviderFailuresAndRejectsLegacyFallback) {
  auto format = Format::get("parquet");
  ASSERT_TRUE(format.ok());
  for (auto failure : {LOON_SOURCE_CREDENTIAL_DENIED, LOON_SOURCE_CREDENTIAL_EXPIRED, LOON_SOURCE_CREDENTIAL_TRANSIENT,
                       LOON_SOURCE_CREDENTIAL_CANCELLED, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE}) {
    auto provider = std::make_shared<Provider>();
    provider->failure = failure;
    auto context = Context(provider);
    api::ColumnGroupFile file{kFile, 0, 0, {}};
    ExpectSource((*format)->create_reader_with_context(nullptr, file, Properties(), {}, nullptr, context).status(),
                 failure);
    EXPECT_EQ(provider->calls, failure == LOON_SOURCE_CREDENTIAL_TRANSIENT ? 4 : 1);
    provider->calls = 0;
    context = Context(provider);
    ExpectSource((*format)->explore_with_context(kRoot, Properties(), context).status(), failure);
    EXPECT_EQ(provider->calls, failure == LOON_SOURCE_CREDENTIAL_TRANSIENT ? 4 : 1);
  }
  auto provider = std::make_shared<Provider>();
  auto context = Context(provider);
  for (const auto* path : {"/tmp/file.parquet", "https://acct.blob.core.windows.net/tables/team/other/file"}) {
    api::ColumnGroupFile file{path, 0, 0, {}};
    ExpectSource((*format)->create_reader_with_context(nullptr, file, Properties(), {}, nullptr, context).status(),
                 LOON_SOURCE_CREDENTIAL_DENIED);
  }
  for (const auto* name : {LOON_FORMAT_VORTEX, LOON_FORMAT_PAIMON_TABLE}) {
    auto unsupported = Format::get(name);
    ASSERT_TRUE(unsupported.ok());
    EXPECT_TRUE((*unsupported)->explore_with_context(kRoot, Properties(), context).status().IsNotImplemented());
  }
  EXPECT_EQ(provider->calls, 0);
}

TEST(SourceFormatReaderTest, FooterOptimizationPreservesSourceFailureWithoutAnotherAcquisitionBatch) {
  auto format = Format::get(LOON_FORMAT_PARQUET);
  ASSERT_TRUE(format.ok());
  for (auto failure : {LOON_SOURCE_CREDENTIAL_DENIED, LOON_SOURCE_CREDENTIAL_EXPIRED, LOON_SOURCE_CREDENTIAL_TRANSIENT,
                       LOON_SOURCE_CREDENTIAL_CANCELLED, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE}) {
    SCOPED_TRACE(failure);
    auto provider = std::make_shared<Provider>();
    provider->failure = failure;
    api::ColumnGroupFile file{kFile, 0, 0, {}};
    file.properties[api::kPropertyFileSize] = "4096";
    file.properties[api::kPropertyFooterSize] = "64";
    auto reader = (*format)->create_reader_with_context(nullptr, file, Properties(), {}, nullptr, Context(provider));
    ExpectSource(reader.status(), failure);
    EXPECT_EQ(provider->calls, failure == LOON_SOURCE_CREDENTIAL_TRANSIENT ? 4 : 1);
  }
}

TEST(SourceFormatReaderTest, CAbiMetadataAndExplorePreserveDenialWithoutWritingInternalManifest) {
  for (auto failure : {LOON_SOURCE_CREDENTIAL_DENIED, LOON_SOURCE_CREDENTIAL_EXPIRED, LOON_SOURCE_CREDENTIAL_TRANSIENT,
                       LOON_SOURCE_CREDENTIAL_CANCELLED, LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE}) {
    const char* roots[] = {kRoot};
    LoonSourceReadContextOptions options{1, sizeof(LoonSourceReadContextOptions), "acct", roots, 1, 0, 3000};
    LoonSourceCredentialCallbacks callbacks{
        1,
        sizeof(LoonSourceCredentialCallbacks),
        &failure,
        [](void* data, const LoonSourceCredentialRequest*, LoonSourceCredentialResult* result) {
          result->status = *static_cast<LoonSourceCredentialStatus*>(data);
        },
        [](void*, LoonSourceCredentialResult*) {},
        [](void*) {}};
    LoonSourceReadContextHandle context = 0;
    auto result = loon_source_read_context_create(&options, &callbacks, &context);
    ASSERT_EQ(result.err_code, LOON_SUCCESS);
    loon_ffi_free_result(&result);
    const char* keys[] = {"extfs.catalog.cloud_provider", "extfs.catalog.access_key_id", "fs.storage_type"};
    const char* vals[] = {"azure", "acct", "local"};
    LoonProperties props{};
    result = loon_properties_create(keys, vals, 3, &props);
    ASSERT_EQ(result.err_code, LOON_SUCCESS);
    loon_ffi_free_result(&result);
    uint64_t rows = 999;
    result = loon_exttable_get_file_info_with_context("parquet", kFile, &props, context, &rows);
    EXPECT_EQ(result.err_code, 116 + static_cast<int>(failure));
    EXPECT_EQ(rows, 0);
    loon_ffi_free_result(&result);
    const auto destination =
        std::filesystem::temp_directory_path() / ("milvus-source-failed-explore-" + std::to_string(context));
    ASSERT_FALSE(std::filesystem::exists(destination));
    const char* columns[] = {"value"};
    uint64_t count = 999;
    char* path = nullptr;
    result = loon_exttable_explore_with_context(columns, 1, "parquet", destination.c_str(), kRoot, &props, context,
                                                &count, &path);
    EXPECT_EQ(result.err_code, 116 + static_cast<int>(failure));
    EXPECT_EQ(count, 0);
    EXPECT_EQ(path, nullptr);
    EXPECT_FALSE(std::filesystem::exists(destination));
    loon_ffi_free_result(&result);
    loon_properties_free(&props);
    loon_source_read_context_release(context);
  }
}
}  // namespace
}  // namespace milvus_storage
