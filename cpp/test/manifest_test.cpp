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

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <avro/Compiler.hh>
#include <avro/DataFile.hh>
#include <avro/Stream.hh>
#include <parquet/arrow/writer.h>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/filesystem/upload_conditional.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/manifest.h"
#include "test_env.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

namespace {

class ManifestTestOutputStream final : public arrow::io::OutputStream {
  public:
  ManifestTestOutputStream(arrow::Status write_status, arrow::Status close_status, arrow::Status abort_status)
      : write_status_(std::move(write_status)),
        close_status_(std::move(close_status)),
        abort_status_(std::move(abort_status)) {}

  arrow::Status Close() override {
    ++close_count_;
    if (close_status_.ok()) {
      closed_ = true;
    }
    return close_status_;
  }

  arrow::Status Abort() override {
    ++abort_count_;
    return abort_status_;
  }

  arrow::Result<int64_t> Tell() const override { return position_; }

  bool closed() const override { return closed_; }

  arrow::Status Write(const void*, int64_t nbytes) override {
    ++write_count_;
    if (write_status_.ok()) {
      position_ += nbytes;
    }
    return write_status_;
  }

  int abort_count() const { return abort_count_; }
  int close_count() const { return close_count_; }
  int write_count() const { return write_count_; }

  private:
  arrow::Status write_status_;
  arrow::Status close_status_;
  arrow::Status abort_status_;
  int64_t position_ = 0;
  int abort_count_ = 0;
  int close_count_ = 0;
  int write_count_ = 0;
  bool closed_ = false;
};

class ManifestTestOutputFileSystem final : public arrow::fs::SubTreeFileSystem {
  public:
  ManifestTestOutputFileSystem(std::shared_ptr<arrow::fs::FileSystem> base_fs,
                               std::shared_ptr<arrow::io::OutputStream> stream)
      : arrow::fs::SubTreeFileSystem("", std::move(base_fs)), stream_(std::move(stream)) {}

  std::string type_name() const override { return "manifest-test-output"; }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string&, const std::shared_ptr<const arrow::KeyValueMetadata>&) override {
    return stream_;
  }

  private:
  std::shared_ptr<arrow::io::OutputStream> stream_;
};

class ManifestTestConditionalFileSystem final : public arrow::fs::SubTreeFileSystem, public UploadConditional {
  public:
  ManifestTestConditionalFileSystem(std::shared_ptr<arrow::fs::FileSystem> base_fs,
                                    std::shared_ptr<arrow::io::OutputStream> stream)
      : arrow::fs::SubTreeFileSystem("", std::move(base_fs)), stream_(std::move(stream)) {}

  std::string type_name() const override { return "manifest-test-conditional"; }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenConditionalOutputStream(
      const std::string&, std::shared_ptr<arrow::KeyValueMetadata>) override {
    return stream_;
  }

  private:
  std::shared_ptr<arrow::io::OutputStream> stream_;
};

}  // namespace

class ManifestTest : public ::testing::Test {
  protected:
  void SetUp() override {
    Manifest::CleanCache();
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("manifest-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  // Helper: write manifest and read it back via WriteTo/ReadFrom
  std::shared_ptr<Manifest> RoundTrip(const Manifest& manifest, size_t version = 1) {
    std::string path = get_manifest_filepath(base_path_, version);
    auto status = Manifest::WriteTo(fs_, path, manifest);
    EXPECT_TRUE(status.ok()) << status.ToString();
    Manifest::CleanCache();
    auto result = Manifest::ReadFrom(fs_, path);
    EXPECT_TRUE(result.ok()) << result.status().ToString();
    return std::move(result).ValueOrDie();
  }

  // Helper: make a column group with specified columns, format, and files
  static std::shared_ptr<ColumnGroup> MakeCG(std::vector<std::string> columns,
                                             const std::string& format,
                                             std::vector<ColumnGroupFile> files) {
    auto cg = std::make_shared<ColumnGroup>();
    cg->columns = std::move(columns);
    cg->format = format;
    cg->files = std::move(files);
    return cg;
  }

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  api::Properties properties_;
  std::string base_path_;
};

// ---------- Read-Write Roundtrip Tests ----------

TEST_F(ManifestTest, EmptyManifestRoundTrip) {
  Manifest manifest;
  auto read_back = RoundTrip(manifest);
  ASSERT_NE(read_back, nullptr);
  EXPECT_TRUE(read_back->columnGroups().empty());
  EXPECT_TRUE(read_back->deltaLogs().empty());
  EXPECT_TRUE(read_back->stats().empty());
  EXPECT_TRUE(read_back->indexes().empty());
  EXPECT_TRUE(read_back->lobFiles().empty());
}

TEST_F(ManifestTest, ColumnGroupsRoundTrip) {
  auto cg = MakeCG({"id", "name"}, LOON_FORMAT_PARQUET,
                   {{.path = get_data_filepath(base_path_, "file1.parquet"), .start_index = 0, .end_index = 100}});

  Manifest manifest({cg});
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->columnGroups().size(), 1);
  auto& rcg = read_back->columnGroups()[0];
  EXPECT_EQ(rcg->columns.size(), 2);
  EXPECT_EQ(rcg->columns[0], "id");
  EXPECT_EQ(rcg->columns[1], "name");
  EXPECT_EQ(rcg->format, LOON_FORMAT_PARQUET);
  ASSERT_EQ(rcg->files.size(), 1);
  EXPECT_EQ(rcg->files[0].start_index, 0);
  EXPECT_EQ(rcg->files[0].end_index, 100);
}

// Absolute URIs survive the round trip -- in the form this library actually
// consumes, and only that form.
//
// The URI convention here is scheme://ENDPOINT/bucket/key, not the AWS-console
// s3://bucket/key. StorageUri::Parse rejects the latter, and so therefore do
// FilesystemCache::resolve_config and PlainFormat::create_reader, which parse
// the stored path again on the way to opening it.
//
// An earlier version of this test asserted that s3://bucket/key.parquet
// round-trips, which it does -- textually. That made it look like support
// existed when nothing downstream can open such a path, which is a worse
// failure than no test: it documents a capability that is not there. The
// unsupported form is pinned separately below, by its classification.
TEST_F(ManifestTest, AbsoluteUriPathsRoundTrip) {
  const std::vector<std::string> absolute = {
      "s3://s3.us-east-1.amazonaws.com/my-bucket/data/file1.parquet",
      "s3://minio.internal:9000/my-bucket/deeply/nested/file2.parquet",
      "local:///local/dir/_data/f.parquet",
  };

  // What a deployment that can actually open these locations looks like: one
  // extfs.* entry per remote URI, and nothing at all for the local one, which
  // needs no external filesystem.
  api::Properties uri_properties;
  ASSERT_STATUS_OK(InitTestProperties(uri_properties));
  uri_properties["extfs.east.address"] = std::string("s3.us-east-1.amazonaws.com");
  uri_properties["extfs.east.bucket_name"] = std::string("my-bucket");
  uri_properties["extfs.east.storage_type"] = std::string("remote");
  uri_properties["extfs.east.cloud_provider"] = std::string(kCloudProviderAWS);
  uri_properties["extfs.east.access_key_id"] = std::string("east_key");
  uri_properties["extfs.east.access_key_value"] = std::string("east_secret");
  uri_properties["extfs.minio.address"] = std::string("minio.internal:9000");
  uri_properties["extfs.minio.bucket_name"] = std::string("my-bucket");
  uri_properties["extfs.minio.storage_type"] = std::string("remote");
  uri_properties["extfs.minio.cloud_provider"] = std::string(kCloudProviderAWS);
  uri_properties["extfs.minio.access_key_id"] = std::string("minio_key");
  uri_properties["extfs.minio.access_key_value"] = std::string("minio_secret");

  ColumnGroups cgs;
  for (const auto& path : absolute) {
    cgs.push_back(MakeCG({"id"}, LOON_FORMAT_PARQUET, {{.path = path, .start_index = 0, .end_index = 1}}));
  }

  Manifest manifest(cgs);
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->columnGroups().size(), absolute.size());
  for (size_t i = 0; i < absolute.size(); ++i) {
    ASSERT_EQ(read_back->columnGroups()[i]->files.size(), 1u) << absolute[i];
    // Unchanged: an absolute path is already absolute, so ToAbsolute must
    // return it as it was rather than gluing base_path onto it.
    EXPECT_EQ(read_back->columnGroups()[i]->files[0].path, absolute[i]) << absolute[i];

    // And the consumption chain accepts what came back -- the check the string
    // comparison above does not make.
    //
    // This calls FilesystemCache::resolve_config, not StorageUri::Parse. An
    // earlier version of this test called Parse and claimed in its comment to
    // be covering the resolver, which is a different function with a stricter
    // answer: "local:///local/dir/_data/f.parquet" parses perfectly well and
    // was still rejected by resolve_config, because the resolver demanded an
    // extfs.* entry for every scheme'd path -- including the local scheme
    // Format::explore() itself stamps onto the files it lists. The round trip
    // this test exists to protect was broken at the step the test did not
    // exercise.
    auto resolved = FilesystemCache::resolve_config(uri_properties, read_back->columnGroups()[i]->files[0].path);
    ASSERT_TRUE(resolved.ok()) << absolute[i] << " -> " << resolved.status().ToString();
  }
}

// The local scheme resolves to the local filesystem rather than demanding an
// extfs.* entry, and does so through the caller-facing entry point.
//
// Pinned separately from the round-trip test because the two failures look the
// same from a string comparison and are not the same bug: this one is about
// what Format::explore() produces being openable at all.
TEST_F(ManifestTest, LocalSchemeUriResolvesToLocalFilesystem) {
  api::Properties props;
  ASSERT_STATUS_OK(InitTestProperties(props));

  const char* local_path = "local:///local/dir/_data/f.parquet";
  auto config = FilesystemCache::resolve_config(props, local_path);
  ASSERT_TRUE(config.ok()) << local_path << " -> " << config.status().ToString();
  EXPECT_EQ(config->storage_type, LOON_FS_TYPE_LOCAL);

  // Do not advertise the standard file:// form until its absolute path can be
  // consumed through the rooted SubTreeFileSystem. Accepting it in the resolver
  // while the reader opens a different path would repeat the exact false-green
  // this regression test exists to prevent.
  auto file_config = FilesystemCache::resolve_config(props, "file:///local/dir/_data/f.parquet");
  ASSERT_FALSE(file_config.ok());
  auto detail = ExtendStatusDetail::UnwrapStatus(file_config.status());
  ASSERT_NE(detail, nullptr) << file_config.status().ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageConfigInvalid);
}

// Exercise the whole producer/consumer chain, not just the URI parser or
// resolver: PlainFormat::explore emits local:///..., the manifest persists and
// restores it, and PlainFormat::create_reader opens that exact restored path.
// This is the regression that originally escaped because the test stopped at
// StorageUri::Parse while the failure lived one layer later in filesystem
// resolution.
TEST_F(ManifestTest, LocalExploreUriSurvivesManifestAndOpensInPlainFormat) {
  if (IsCloudEnv()) {
    GTEST_SKIP() << "Test requires the local filesystem; it is already covered by the local CI pass.";
  }

  const auto schema = arrow::schema({arrow::field("id", arrow::int64())});
  arrow::Int64Builder builder;
  ASSERT_STATUS_OK(builder.AppendValues({1, 2, 3}));
  ASSERT_AND_ASSIGN(auto values, builder.Finish());
  const auto table = arrow::Table::Make(schema, {values});

  const std::string source_dir = base_path_ + "/local-source";
  const std::string source_path = source_dir + "/data.parquet";
  ASSERT_STATUS_OK(fs_->CreateDir(source_dir, /*recursive=*/true));
  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(source_path));
  ASSERT_STATUS_OK(::parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/3));
  ASSERT_STATUS_OK(sink->Close());

  ASSERT_AND_ASSIGN(auto format, Format::get(LOON_FORMAT_PARQUET));
  ASSERT_AND_ASSIGN(auto explored, format->explore(source_dir, properties_));
  ASSERT_EQ(explored.size(), 1u);
  EXPECT_EQ(explored[0].path.rfind("local:///", 0), 0u) << explored[0].path;

  auto column_group = MakeCG({"id"}, LOON_FORMAT_PARQUET, explored);
  auto read_back = RoundTrip(Manifest({column_group}), /*version=*/42);
  ASSERT_EQ(read_back->columnGroups().size(), 1u);
  ASSERT_EQ(read_back->columnGroups()[0]->files.size(), 1u);
  const auto& restored = read_back->columnGroups()[0]->files[0];
  EXPECT_EQ(restored.path, explored[0].path);

  ASSERT_AND_ASSIGN(auto reader,
                    format->create_reader(schema, restored, properties_, std::vector<std::string>{"id"}, nullptr));
  ASSERT_AND_ASSIGN(auto row_groups, reader->get_row_group_infos());
  ASSERT_EQ(row_groups.size(), 1u);
  EXPECT_EQ(row_groups[0].start_offset, 0);
  EXPECT_EQ(row_groups[0].end_offset, 3);
}

// A stored path that is not a URI is bad manifest content, and is reported as
// that at the point the manifest is turned into paths.
//
// It used to be accepted here and cached, then fail much later from the
// filesystem layer as a configuration error -- sending an operator to check
// endpoints and credentials for a file whose bytes were the problem. The
// distinction the fix has to keep is in the second case below: a well-formed
// URI this build does not support is NOT corruption, and must keep its
// configuration classification.
TEST_F(ManifestTest, MalformedStoredUriIsCorruptedNotConfig) {
  auto cgs = ColumnGroups{
      MakeCG({"id"}, LOON_FORMAT_PARQUET, {{.path = "s3://bucket/%ZZ", .start_index = 0, .end_index = 1}})};
  Manifest manifest(cgs);

  std::string path = get_manifest_filepath(base_path_, 7);
  ASSERT_STATUS_OK(Manifest::WriteTo(fs_, path, manifest));
  Manifest::CleanCache();

  auto read_back = Manifest::ReadFrom(fs_, path);
  ASSERT_FALSE(read_back.ok()) << "a location that names no object was accepted";
  auto detail = ExtendStatusDetail::UnwrapStatus(read_back.status());
  ASSERT_NE(detail, nullptr) << read_back.status().ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::DataCorrupted);
  EXPECT_EQ(CategoryForExtendStatusCode(detail->code()), ErrorCategory::DataFormat);

  // The console-style form is well-formed and merely unsupported: it survives
  // the load, and is answered by the filesystem layer as configuration.
  Manifest::CleanCache();
  auto supported = ColumnGroups{
      MakeCG({"id"}, LOON_FORMAT_PARQUET, {{.path = "s3://bucket/key.parquet", .start_index = 0, .end_index = 1}})};
  std::string path2 = get_manifest_filepath(base_path_, 8);
  ASSERT_STATUS_OK(Manifest::WriteTo(fs_, path2, Manifest(supported)));
  Manifest::CleanCache();
  auto ok_load = Manifest::ReadFrom(fs_, path2);
  ASSERT_TRUE(ok_load.ok()) << "an intact manifest was called corrupted: " << ok_load.status().ToString();

  api::Properties props;
  ASSERT_STATUS_OK(InitTestProperties(props));
  auto resolved = FilesystemCache::resolve_config(props, ok_load.ValueOrDie()->columnGroups()[0]->files[0].path);
  ASSERT_FALSE(resolved.ok());
  auto config_detail = ExtendStatusDetail::UnwrapStatus(resolved.status());
  ASSERT_NE(config_detail, nullptr) << resolved.status().ToString();
  EXPECT_EQ(CategoryForExtendStatusCode(config_detail->code()), ErrorCategory::System);
}

// The console form is not supported, and says so in a way an operator can act
// on rather than failing later as a missing object.
TEST_F(ManifestTest, ConsoleStyleUriIsRejectedAsConfigNotSilentlyMangled) {
  for (const char* path : {"s3://my-bucket/key.parquet", "gs://my-bucket/key.parquet"}) {
    auto parsed = StorageUri::Parse(path);
    ASSERT_FALSE(parsed.ok()) << path << ": if this now parses, the convention changed and"
                              << " AbsoluteUriPathsRoundTrip should cover this form too";
    auto detail = ExtendStatusDetail::UnwrapStatus(parsed.status());
    ASSERT_NE(detail, nullptr) << path << " arrived unclassified: " << parsed.status().ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageConfigInvalid) << path;
    EXPECT_EQ(CategoryForExtendStatusCode(detail->code()), ErrorCategory::System) << path;
  }
}

TEST_F(ManifestTest, DeltaLogsRoundTrip) {
  DeltaLog d1{
      .path = get_delta_filepath(base_path_, "del1.parquet"), .type = DeltaLogType::PRIMARY_KEY, .num_entries = 50};
  DeltaLog d2{
      .path = get_delta_filepath(base_path_, "del2.parquet"), .type = DeltaLogType::POSITIONAL, .num_entries = 30};
  DeltaLog d3{
      .path = get_delta_filepath(base_path_, "del3.parquet"), .type = DeltaLogType::EQUALITY, .num_entries = 10};

  Manifest manifest({}, {d1, d2, d3});
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->deltaLogs().size(), 3);
  EXPECT_EQ(read_back->deltaLogs()[0].type, DeltaLogType::PRIMARY_KEY);
  EXPECT_EQ(read_back->deltaLogs()[0].num_entries, 50);
  EXPECT_EQ(read_back->deltaLogs()[1].type, DeltaLogType::POSITIONAL);
  EXPECT_EQ(read_back->deltaLogs()[1].num_entries, 30);
  EXPECT_EQ(read_back->deltaLogs()[2].type, DeltaLogType::EQUALITY);
  EXPECT_EQ(read_back->deltaLogs()[2].num_entries, 10);
}

TEST_F(ManifestTest, StatsRoundTrip) {
  Statistics stat1;
  stat1.paths = {get_stats_filepath(base_path_, "bloom_100.bin"), get_stats_filepath(base_path_, "bloom_101.bin")};
  stat1.metadata = {{"type", "bloom_filter"}, {"fpp", "0.01"}};

  Statistics stat2;
  stat2.paths = {get_stats_filepath(base_path_, "bm25_101.bin")};
  stat2.metadata = {};

  std::map<std::string, Statistics> stats = {{"bloom_filter.100", stat1}, {"bm25.101", stat2}};

  Manifest manifest({}, {}, stats);
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->stats().size(), 2);

  auto& rs1 = read_back->stats().at("bloom_filter.100");
  EXPECT_EQ(rs1.paths.size(), 2);
  EXPECT_EQ(rs1.metadata.at("type"), "bloom_filter");
  EXPECT_EQ(rs1.metadata.at("fpp"), "0.01");

  auto& rs2 = read_back->stats().at("bm25.101");
  EXPECT_EQ(rs2.paths.size(), 1);
  EXPECT_TRUE(rs2.metadata.empty());
}

TEST_F(ManifestTest, IndexesRoundTrip) {
  Index idx1{.column_name = "vector",
             .index_name = "vector_hnsw",
             .index_type = "hnsw",
             .path = get_index_filepath(base_path_, "vec_hnsw"),
             .field_id = 100,
             .index_id = 200,
             .build_id = 300,
             .index_version = 4,
             .num_rows = 1000,
             .serialized_size = 1024,
             .mem_size = 2048,
             .current_index_version = 15,
             .current_scalar_index_version = 7,
             .index_store_path_version = 1,
             .index_file_keys = {"index.bin", "raw_data.bin"},
             .properties = {{"M", "16"}, {"ef_construction", "128"}}};

  Index idx2{.column_name = "id",
             .index_type = "inverted",
             .path = get_index_filepath(base_path_, "id_inverted.idx"),
             .properties = {}};

  Manifest manifest({}, {}, {}, {idx1, idx2});
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->indexes().size(), 2);
  const Index* found_hnsw = read_back->getIndex("vector", "hnsw");
  ASSERT_NE(found_hnsw, nullptr);
  EXPECT_EQ(found_hnsw->index_name, idx1.index_name);
  EXPECT_EQ(found_hnsw->path, idx1.path);
  EXPECT_EQ(found_hnsw->field_id, idx1.field_id);
  EXPECT_EQ(found_hnsw->index_id, idx1.index_id);
  EXPECT_EQ(found_hnsw->build_id, idx1.build_id);
  EXPECT_EQ(found_hnsw->index_version, idx1.index_version);
  EXPECT_EQ(found_hnsw->num_rows, idx1.num_rows);
  EXPECT_EQ(found_hnsw->serialized_size, idx1.serialized_size);
  EXPECT_EQ(found_hnsw->mem_size, idx1.mem_size);
  EXPECT_EQ(found_hnsw->current_index_version, idx1.current_index_version);
  EXPECT_EQ(found_hnsw->current_scalar_index_version, idx1.current_scalar_index_version);
  EXPECT_EQ(found_hnsw->index_store_path_version, idx1.index_store_path_version);
  EXPECT_EQ(found_hnsw->index_file_keys, idx1.index_file_keys);
  EXPECT_EQ(found_hnsw->properties.at("M"), "16");
  EXPECT_EQ(found_hnsw->properties.at("ef_construction"), "128");

  const Index* found_inv = read_back->getIndex("id", "inverted");
  ASSERT_NE(found_inv, nullptr);
  EXPECT_EQ(found_inv->field_id, 0);
  EXPECT_TRUE(found_inv->index_file_keys.empty());
  EXPECT_TRUE(found_inv->properties.empty());
}

TEST_F(ManifestTest, LobFilesRoundTrip) {
  // LOB files live at partition level: base_path/../lobs/{field_id}/_data/
  // After normalization, the absolute path starts with "lobs/" for base_path_ = "manifest-test"
  std::string lob_prefix = std::filesystem::path(base_path_).parent_path().string();
  if (!lob_prefix.empty())
    lob_prefix += "/";
  lob_prefix += "lobs/";

  LobFileInfo lob1{lob_prefix + "101/_data/lob_001.vortex", 101, 1000, 900, 1048576};
  LobFileInfo lob2{lob_prefix + "101/_data/lob_002.vortex", 101, 2000, 1800, 2097152};
  LobFileInfo lob3{lob_prefix + "102/_data/lob_001.vortex", 102, 500, 450, 524288};

  Manifest manifest({}, {}, {}, {}, {lob1, lob2, lob3});
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->lobFiles().size(), 3);

  EXPECT_EQ(read_back->lobFiles()[0].path, lob1.path);
  EXPECT_EQ(read_back->lobFiles()[0].field_id, 101);
  EXPECT_EQ(read_back->lobFiles()[0].total_rows, 1000);
  EXPECT_EQ(read_back->lobFiles()[0].valid_rows, 900);
  EXPECT_EQ(read_back->lobFiles()[0].file_size_bytes, 1048576);

  EXPECT_EQ(read_back->lobFiles()[1].path, lob2.path);
  EXPECT_EQ(read_back->lobFiles()[1].total_rows, 2000);

  EXPECT_EQ(read_back->lobFiles()[2].field_id, 102);
  EXPECT_EQ(read_back->lobFiles()[2].file_size_bytes, 524288);

  // getLobFilesForField filtering
  auto field101 = read_back->getLobFilesForField(101);
  ASSERT_EQ(field101.size(), 2);

  auto field102 = read_back->getLobFilesForField(102);
  ASSERT_EQ(field102.size(), 1);

  EXPECT_TRUE(read_back->getLobFilesForField(999).empty());
}

TEST_F(ManifestTest, FullManifestRoundTrip) {
  // Populate all fields
  auto cg1 =
      MakeCG({"id", "name"}, LOON_FORMAT_PARQUET,
             {{.path = get_data_filepath(base_path_, "cg1_part0.parquet"), .start_index = 0, .end_index = 500},
              {.path = get_data_filepath(base_path_, "cg1_part1.parquet"), .start_index = 500, .end_index = 1000}});
  auto cg2 = MakeCG({"value", "vector"}, LOON_FORMAT_PARQUET,
                    {{.path = get_data_filepath(base_path_, "cg2.parquet"), .start_index = 0, .end_index = 1000}});

  std::vector<DeltaLog> deltas = {
      {.path = get_delta_filepath(base_path_, "del.parquet"), .type = DeltaLogType::PRIMARY_KEY, .num_entries = 20}};

  Statistics stat;
  stat.paths = {get_stats_filepath(base_path_, "bloom.bin")};
  stat.metadata = {{"version", "1"}};
  std::map<std::string, Statistics> stats = {{"bloom_filter.100", stat}};

  std::vector<Index> indexes = {{.column_name = "vector",
                                 .index_type = "hnsw",
                                 .path = get_index_filepath(base_path_, "vec.idx"),
                                 .properties = {{"M", "16"}}}};

  std::string lob_prefix = std::filesystem::path(base_path_).parent_path().string();
  if (!lob_prefix.empty())
    lob_prefix += "/";
  lob_prefix += "lobs/";
  std::vector<LobFileInfo> lob_files = {{lob_prefix + "100/_data/lob_001.vortex", 100, 500, 480, 65536}};

  Manifest manifest({cg1, cg2}, deltas, stats, indexes, lob_files);
  auto read_back = RoundTrip(manifest);

  EXPECT_EQ(read_back->columnGroups().size(), 2);
  EXPECT_EQ(read_back->deltaLogs().size(), 1);
  EXPECT_EQ(read_back->stats().size(), 1);
  EXPECT_EQ(read_back->indexes().size(), 1);
  EXPECT_EQ(read_back->lobFiles().size(), 1);

  // Verify multi-file column group
  EXPECT_EQ(read_back->columnGroups()[0]->files.size(), 2);
  EXPECT_EQ(read_back->columnGroups()[0]->files[0].end_index, 500);
  EXPECT_EQ(read_back->columnGroups()[0]->files[1].start_index, 500);

  // Verify LOB file
  EXPECT_EQ(read_back->lobFiles()[0].field_id, 100);
  EXPECT_EQ(read_back->lobFiles()[0].total_rows, 500);
}

// ---------- Column Group Policy Tests ----------

TEST_F(ManifestTest, SingleColumnGroupPolicy) {
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema));

  auto groups = policy->get_column_groups();
  ASSERT_EQ(groups.size(), 1);
  EXPECT_EQ(groups[0]->columns.size(), 4);
  EXPECT_EQ(groups[0]->format, LOON_FORMAT_PARQUET);

  // RoundTrip with policy-generated column groups
  Manifest manifest(groups);
  auto read_back = RoundTrip(manifest);
  ASSERT_EQ(read_back->columnGroups().size(), 1);
  EXPECT_EQ(read_back->columnGroups()[0]->columns.size(), 4);
}

TEST_F(ManifestTest, SchemaBasedColumnGroupPolicy) {
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  // "id|value" in group 1, "name" in group 2, "vector" in group 3
  ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy("id|value,name,vector", LOON_FORMAT_PARQUET, schema));

  auto groups = policy->get_column_groups();
  ASSERT_EQ(groups.size(), 3);

  // Verify each group has the expected columns
  EXPECT_EQ(groups[0]->columns.size(), 2);  // id, value
  EXPECT_EQ(groups[1]->columns.size(), 1);  // name
  EXPECT_EQ(groups[2]->columns.size(), 1);  // vector

  // RoundTrip
  Manifest manifest(groups);
  auto read_back = RoundTrip(manifest);
  ASSERT_EQ(read_back->columnGroups().size(), 3);

  // Verify column names survived roundtrip
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(read_back->columnGroups()[i]->columns.size(), groups[i]->columns.size());
    for (size_t j = 0; j < groups[i]->columns.size(); ++j) {
      EXPECT_EQ(read_back->columnGroups()[i]->columns[j], groups[i]->columns[j]);
    }
  }
}

TEST_F(ManifestTest, SizeBasedColumnGroupPolicy) {
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto test_batch, CreateTestData(schema));
  // max_avg_column_size = 1, max_columns_in_group = 2 -> should split into multiple groups
  ASSERT_AND_ASSIGN(auto policy, CreateSizeBasePolicy(1, 2, LOON_FORMAT_PARQUET, schema));
  // SizeBasedPolicy requires a sample before get_column_groups()
  ASSERT_STATUS_OK(policy->sample(test_batch));

  auto groups = policy->get_column_groups();
  EXPECT_GT(groups.size(), 1);

  // Every group should have at most 2 columns
  for (const auto& g : groups) {
    EXPECT_LE(g->columns.size(), 2);
    EXPECT_EQ(g->format, LOON_FORMAT_PARQUET);
  }

  // RoundTrip
  Manifest manifest(groups);
  auto read_back = RoundTrip(manifest);
  ASSERT_EQ(read_back->columnGroups().size(), groups.size());
}

// ---------- Hybrid Format Tests ----------

TEST_F(ManifestTest, HybridFormatsInSingleManifest) {
  auto cg_parquet =
      MakeCG({"id", "name"}, LOON_FORMAT_PARQUET,
             {{.path = get_data_filepath(base_path_, "cg_parquet.parquet"), .start_index = 0, .end_index = 100}});

  auto cg_iceberg = MakeCG({"value"}, LOON_FORMAT_ICEBERG_TABLE,
                           {{.path = "s3://bucket/warehouse/table/data/file1.parquet",
                             .start_index = 0,
                             .end_index = 100,
                             .properties = {{api::kPropertyMetadata, std::string({'\x01', '\x02', '\x03'})}}}});

  auto cg_lance = MakeCG({"vector"}, LOON_FORMAT_LANCE_TABLE,
                         {{.path = "s3://bucket/lance/table.lance", .start_index = 0, .end_index = 100}});

  Manifest manifest({cg_parquet, cg_iceberg, cg_lance});
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->columnGroups().size(), 3);
  EXPECT_EQ(read_back->columnGroups()[0]->format, LOON_FORMAT_PARQUET);
  EXPECT_EQ(read_back->columnGroups()[1]->format, LOON_FORMAT_ICEBERG_TABLE);
  EXPECT_EQ(read_back->columnGroups()[2]->format, LOON_FORMAT_LANCE_TABLE);

  // Verify iceberg column group metadata bytes survived roundtrip
  EXPECT_EQ(read_back->columnGroups()[1]->files[0].properties.at(api::kPropertyMetadata),
            std::string({'\x01', '\x02', '\x03'}));

  // Verify external table paths are preserved as-is (absolute URIs)
  EXPECT_EQ(read_back->columnGroups()[1]->files[0].path, "s3://bucket/warehouse/table/data/file1.parquet");
  EXPECT_EQ(read_back->columnGroups()[2]->files[0].path, "s3://bucket/lance/table.lance");
}

TEST_F(ManifestTest, HybridFormatsWithGetColumnGroup) {
  auto cg_parquet = MakeCG({"id"}, LOON_FORMAT_PARQUET,
                           {{.path = get_data_filepath(base_path_, "p.parquet"), .start_index = 0, .end_index = 50}});
  auto cg_iceberg = MakeCG({"name", "value"}, LOON_FORMAT_ICEBERG_TABLE,
                           {{.path = "s3://bucket/iceberg/data.parquet",
                             .start_index = 0,
                             .end_index = 50,
                             .properties = {{api::kPropertyMetadata, std::string(1, '\xAB')}}}});

  Manifest manifest({cg_parquet, cg_iceberg});
  auto read_back = RoundTrip(manifest);

  // getColumnGroup should locate each column in the correct group
  auto id_cg = read_back->getColumnGroup("id");
  ASSERT_NE(id_cg, nullptr);
  EXPECT_EQ(id_cg->format, LOON_FORMAT_PARQUET);

  auto name_cg = read_back->getColumnGroup("name");
  ASSERT_NE(name_cg, nullptr);
  EXPECT_EQ(name_cg->format, LOON_FORMAT_ICEBERG_TABLE);

  auto value_cg = read_back->getColumnGroup("value");
  ASSERT_NE(value_cg, nullptr);
  EXPECT_EQ(value_cg.get(), name_cg.get());  // same column group

  EXPECT_EQ(read_back->getColumnGroup("nonexistent"), nullptr);
}

// ---------- Multiple Files in One Column Group ----------

TEST_F(ManifestTest, MultipleFilesInOneColumnGroup) {
  std::vector<ColumnGroupFile> files;
  files.reserve(5);
  for (int i = 0; i < 5; ++i) {
    files.push_back({.path = get_data_filepath(base_path_, "part_" + std::to_string(i) + ".parquet"),
                     .start_index = i * 1000,
                     .end_index = (i + 1) * 1000});
  }

  auto cg = MakeCG({"id", "name", "value", "vector"}, LOON_FORMAT_PARQUET, files);
  Manifest manifest({cg});
  auto read_back = RoundTrip(manifest);

  ASSERT_EQ(read_back->columnGroups().size(), 1);
  auto& rcg = read_back->columnGroups()[0];
  ASSERT_EQ(rcg->files.size(), 5);

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(rcg->files[i].start_index, i * 1000);
    EXPECT_EQ(rcg->files[i].end_index, (i + 1) * 1000);
  }
}

TEST_F(ManifestTest, MultipleFilesWithMetadata) {
  std::string meta1 = {'\x10', '\x20', '\x30'};
  std::string meta2 = {'\xAA', '\xBB'};
  std::string meta3 = {};

  auto cg = MakeCG({"id"}, LOON_FORMAT_ICEBERG_TABLE,
                   {{.path = "s3://bucket/data/part0.parquet",
                     .start_index = 0,
                     .end_index = 500,
                     .properties = {{api::kPropertyMetadata, meta1}}},
                    {.path = "s3://bucket/data/part1.parquet",
                     .start_index = 500,
                     .end_index = 1000,
                     .properties = {{api::kPropertyMetadata, meta2}}},
                    {.path = "s3://bucket/data/part2.parquet", .start_index = 1000, .end_index = 1500}});

  Manifest manifest({cg});
  auto read_back = RoundTrip(manifest);

  auto& rcg = read_back->columnGroups()[0];
  ASSERT_EQ(rcg->files.size(), 3);
  EXPECT_EQ(rcg->files[0].properties.at(api::kPropertyMetadata), meta1);
  EXPECT_EQ(rcg->files[1].properties.at(api::kPropertyMetadata), meta2);
  EXPECT_TRUE(rcg->files[2].properties.find(api::kPropertyMetadata) == rcg->files[2].properties.end());
}

// ---------- Edge Cases ----------

TEST_F(ManifestTest, ConditionalWriteFailureAbortsAndPreservesPrimaryStatus) {
  auto write_failure =
      MakeExtendError(ExtendStatusCode::StorageTransientNetwork, "conditional write failed", "write detail");
  auto stream = std::make_shared<ManifestTestOutputStream>(write_failure, arrow::Status::OK(),
                                                           arrow::Status::IOError("abort cleanup failed"));
  auto fs = std::make_shared<ManifestTestConditionalFileSystem>(fs_, stream);

  auto status = Manifest::WriteTo(fs, base_path_ + "/conditional-write-failure.manifest", Manifest{});

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.message(), write_failure.message());
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientNetwork);
  EXPECT_EQ(stream->write_count(), 1);
  EXPECT_EQ(stream->close_count(), 0);
  EXPECT_EQ(stream->abort_count(), 1);
}

TEST_F(ManifestTest, ConditionalCloseConflictAbortsBeforeMappingToAlreadyExists) {
  auto close_failure = MakeExtendError(ExtendStatusCode::StorageConflict, "conditional close conflict");
  auto stream = std::make_shared<ManifestTestOutputStream>(arrow::Status::OK(), close_failure,
                                                           arrow::Status::IOError("abort cleanup failed"));
  auto fs = std::make_shared<ManifestTestConditionalFileSystem>(fs_, stream);
  const auto path = base_path_ + "/conditional-close-conflict.manifest";

  auto status = Manifest::WriteTo(fs, path, Manifest{});

  ASSERT_TRUE(status.IsAlreadyExists()) << status.ToString();
  EXPECT_NE(status.message().find(path), std::string::npos);
  EXPECT_EQ(stream->write_count(), 1);
  EXPECT_EQ(stream->close_count(), 1);
  EXPECT_EQ(stream->abort_count(), 1);
}

TEST_F(ManifestTest, PlainWriteFailureAbortsAndPreservesPrimaryStatus) {
  auto write_failure = MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "plain write failed", "write detail");
  auto stream = std::make_shared<ManifestTestOutputStream>(write_failure, arrow::Status::OK(),
                                                           arrow::Status::IOError("abort cleanup failed"));
  auto fs = std::make_shared<ManifestTestOutputFileSystem>(fs_, stream);

  auto status = Manifest::WriteTo(fs, base_path_ + "/plain-write-failure.manifest", Manifest{});

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.message(), write_failure.message());
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientTimeout);
  EXPECT_EQ(stream->write_count(), 1);
  EXPECT_EQ(stream->close_count(), 0);
  EXPECT_EQ(stream->abort_count(), 1);
}

TEST_F(ManifestTest, PlainCloseFailureAbortsAndPreservesPrimaryStatus) {
  auto close_failure = MakeExtendError(ExtendStatusCode::StorageTransientService, "plain close failed", "close detail");
  auto stream = std::make_shared<ManifestTestOutputStream>(arrow::Status::OK(), close_failure,
                                                           arrow::Status::IOError("abort cleanup failed"));
  auto fs = std::make_shared<ManifestTestOutputFileSystem>(fs_, stream);

  auto status = Manifest::WriteTo(fs, base_path_ + "/plain-close-failure.manifest", Manifest{});

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.message(), close_failure.message());
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientService);
  EXPECT_EQ(stream->write_count(), 1);
  EXPECT_EQ(stream->close_count(), 1);
  EXPECT_EQ(stream->abort_count(), 1);
}

TEST_F(ManifestTest, WriteToExistingPathFails) {
  Manifest manifest;
  std::string path = get_manifest_filepath(base_path_, 1);
  ASSERT_STATUS_OK(Manifest::WriteTo(fs_, path, manifest));

  // Second write to the same path should fail with AlreadyExists
  auto status = Manifest::WriteTo(fs_, path, manifest);
  EXPECT_TRUE(status.IsAlreadyExists()) << status.ToString();
}

TEST_F(ManifestTest, ReadFromCachesResult) {
  auto cg = MakeCG({"id"}, LOON_FORMAT_PARQUET,
                   {{.path = get_data_filepath(base_path_, "cached.parquet"), .start_index = 0, .end_index = 10}});

  Manifest manifest({cg});
  std::string path = get_manifest_filepath(base_path_, 1);
  ASSERT_STATUS_OK(Manifest::WriteTo(fs_, path, manifest));

  Manifest::CleanCache();
  ASSERT_AND_ASSIGN(auto m1, Manifest::ReadFrom(fs_, path));
  ASSERT_AND_ASSIGN(auto m2, Manifest::ReadFrom(fs_, path));
  EXPECT_EQ(m1.get(), m2.get());  // same pointer from cache
}

TEST_F(ManifestTest, ManifestVersion) {
  Manifest manifest;
  EXPECT_EQ(manifest.version(), MANIFEST_VERSION);

  auto read_back = RoundTrip(manifest);
  EXPECT_EQ(read_back->version(), MANIFEST_VERSION);
}

TEST_F(ManifestTest, ColumnGroupsXFormatsXFiles) {
  // A realistic manifest with all 4 supported formats, multiple files per group,
  // plus delta logs, stats, and indexes.
  auto cg_parquet =
      MakeCG({"id"}, LOON_FORMAT_PARQUET,
             {{.path = get_data_filepath(base_path_, "p0.parquet"), .start_index = 0, .end_index = 1000},
              {.path = get_data_filepath(base_path_, "p1.parquet"), .start_index = 1000, .end_index = 2000},
              {.path = get_data_filepath(base_path_, "p2.parquet"), .start_index = 2000, .end_index = 3000}});

  auto cg_vortex =
      MakeCG({"name"}, LOON_FORMAT_VORTEX,
             {{.path = get_data_filepath(base_path_, "v0.vortex"), .start_index = 0, .end_index = 1500},
              {.path = get_data_filepath(base_path_, "v1.vortex"), .start_index = 1500, .end_index = 3000}});

  auto cg_lance = MakeCG({"value"}, LOON_FORMAT_LANCE_TABLE,
                         {{.path = "s3://bucket/lance/table.lance", .start_index = 0, .end_index = 3000}});

  auto cg_iceberg = MakeCG({"vector"}, LOON_FORMAT_ICEBERG_TABLE,
                           {{.path = "s3://bucket/iceberg/data/i0.parquet",
                             .start_index = 0,
                             .end_index = 1500,
                             .properties = {{api::kPropertyMetadata, std::string(1, '\x01')}}},
                            {.path = "s3://bucket/iceberg/data/i1.parquet",
                             .start_index = 1500,
                             .end_index = 3000,
                             .properties = {{api::kPropertyMetadata, std::string(1, '\x02')}}}});

  DeltaLog delta{
      .path = get_delta_filepath(base_path_, "del.parquet"), .type = DeltaLogType::POSITIONAL, .num_entries = 100};

  Statistics stat;
  stat.paths = {get_stats_filepath(base_path_, "bloom.bin")};
  stat.metadata = {{"fpp", "0.001"}};

  Index idx{.column_name = "vector",
            .index_type = "hnsw",
            .path = get_index_filepath(base_path_, "vec_hnsw.idx"),
            .properties = {{"M", "32"}, {"ef_construction", "256"}}};

  Manifest manifest({cg_parquet, cg_vortex, cg_lance, cg_iceberg}, {delta}, {{"bloom_filter.100", stat}}, {idx});
  auto read_back = RoundTrip(manifest);

  // All 4 column groups present with correct formats
  ASSERT_EQ(read_back->columnGroups().size(), 4);
  EXPECT_EQ(read_back->columnGroups()[0]->format, LOON_FORMAT_PARQUET);
  EXPECT_EQ(read_back->columnGroups()[0]->files.size(), 3);
  EXPECT_EQ(read_back->columnGroups()[1]->format, LOON_FORMAT_VORTEX);
  EXPECT_EQ(read_back->columnGroups()[1]->files.size(), 2);
  EXPECT_EQ(read_back->columnGroups()[2]->format, LOON_FORMAT_LANCE_TABLE);
  EXPECT_EQ(read_back->columnGroups()[2]->files.size(), 1);
  EXPECT_EQ(read_back->columnGroups()[3]->format, LOON_FORMAT_ICEBERG_TABLE);
  EXPECT_EQ(read_back->columnGroups()[3]->files.size(), 2);
  EXPECT_EQ(read_back->columnGroups()[3]->files[0].properties.at(api::kPropertyMetadata), std::string(1, '\x01'));

  // External table paths preserved as absolute URIs
  EXPECT_EQ(read_back->columnGroups()[2]->files[0].path, "s3://bucket/lance/table.lance");
  EXPECT_EQ(read_back->columnGroups()[3]->files[0].path, "s3://bucket/iceberg/data/i0.parquet");

  // Delta logs
  ASSERT_EQ(read_back->deltaLogs().size(), 1);
  EXPECT_EQ(read_back->deltaLogs()[0].type, DeltaLogType::POSITIONAL);

  // Stats
  ASSERT_EQ(read_back->stats().size(), 1);
  EXPECT_EQ(read_back->stats().at("bloom_filter.100").metadata.at("fpp"), "0.001");

  // Indexes
  ASSERT_EQ(read_back->indexes().size(), 1);
  EXPECT_EQ(read_back->indexes()[0].properties.at("M"), "32");
}

// DataCorrupted(117) exists because the coarse arrow-status fallback no
// longer guesses DataFormatBroken for a plain Status::Invalid. Without an
// explicit code here, a manifest that does not parse would silently downgrade
// from "your data is corrupt" to a generic storage error.
//
// These drive the real Manifest::deserialize rather than synthesizing the code,
// because the table entry was pinned while the code path that justifies it had
// no coverage at all -- exactly the "dead code that looks alive" shape the
// producer gate exists to catch, one level down.
TEST_F(ManifestTest, CorruptManifestIsClassifiedCorrupted) {
  struct Case {
    const char* bytes;
    const char* what;
  };
  const Case cases[] = {
      {"", "empty file"},
      {"ab", "shorter than the 4-byte format header"},
      {"NOPE----not-a-manifest", "readable length, but neither avro nor MILV magic"},
  };

  for (const auto& c : cases) {
    // Through the public ReadFrom, not deserialize -- that one is private, and
    // reaching past it would test a path no caller can take.
    std::string path = base_path_ + "/corrupt.manifest";
    ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path));
    ASSERT_STATUS_OK(out->Write(c.bytes, static_cast<int64_t>(std::string(c.bytes).size())));
    ASSERT_STATUS_OK(out->Close());
    Manifest::CleanCache();

    auto result = Manifest::ReadFrom(fs_, path);
    ASSERT_FALSE(result.ok()) << c.what;
    auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
    ASSERT_NE(detail, nullptr) << c.what << ": arrived unclassified, so it reaches segcore as a generic"
                               << " storage failure rather than as corrupt data: " << result.status().ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::DataCorrupted) << c.what;
    EXPECT_EQ(CategoryForExtendStatusCode(detail->code()), ErrorCategory::DataFormat) << c.what;
    EXPECT_FALSE(RetryableForExtendStatusCode(detail->code())) << c.what;
    EXPECT_EQ(ToSegcoreError(result.status()).get_error_code(), milvus::DataFormatBroken) << c.what;
  }
}

TEST_F(ManifestTest, ValidAvroWithIncompatibleSchemaIsDataCorrupted) {
  // This is a healthy Avro object-container file, but it is not a Manifest.
  // The distinction matters: coverage with random/truncated bytes alone does
  // not exercise Avro's writer/reader schema-resolution failure path.
  std::ostringstream bytes;
  auto avro_output = avro::ostreamOutputStream(bytes);
  const auto incompatible_schema = avro::compileJsonSchemaFromString(R"("int")");
  avro::DataFileWriter<int32_t> writer(std::move(avro_output), incompatible_schema);
  writer.write(42);
  writer.close();

  const std::string path = base_path_ + "/wrong-schema.manifest";
  ASSERT_AND_ASSIGN(auto out, fs_->OpenOutputStream(path));
  const auto contents = bytes.str();
  ASSERT_STATUS_OK(out->Write(contents.data(), static_cast<int64_t>(contents.size())));
  ASSERT_STATUS_OK(out->Close());
  Manifest::CleanCache();

  auto result = Manifest::ReadFrom(fs_, path);
  ASSERT_FALSE(result.ok());
  auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
  ASSERT_NE(detail, nullptr) << result.status().ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::DataCorrupted);
  EXPECT_EQ(CategoryForExtendStatusCode(detail->code()), ErrorCategory::DataFormat);
  EXPECT_FALSE(RetryableForExtendStatusCode(detail->code()));
  EXPECT_EQ(ToSegcoreError(result.status()).get_error_code(), milvus::DataFormatBroken);
}

}  // namespace milvus_storage::test
