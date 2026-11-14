// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/file.h>
#include <arrow/util/io_util.h>
#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/properties.h"
#include "deltalake/deltalake_bridge.h"
#include "test_env.h"

namespace milvus_storage {
namespace {

using Json = nlohmann::json;

class FailingDeltaFileSystem final : public FileSystemProxy {
  public:
  FailingDeltaFileSystem(const std::string& root, ExtendStatusCode code)
      : FileSystemProxy(root, std::make_shared<arrow::fs::LocalFileSystem>()), code_(code) {}

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string&) override {
    return MakeExtendError(code_, "injected Delta read failure");
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo&) override {
    return MakeExtendError(code_, "injected Delta read failure");
  }

  private:
  ExtendStatusCode code_;
};

// Checkpoints contain typed maps and structs, so generate them directly rather
// than inferring a Parquet schema from JSON (which would turn maps into structs).
arrow::Status AppendJson(const Json& value, arrow::ArrayBuilder* builder) {
  if (value.is_null()) {
    return builder->AppendNull();
  }
  switch (builder->type()->id()) {
    case arrow::Type::STRUCT: {
      auto* target = static_cast<arrow::StructBuilder*>(builder);
      // StructBuilder can reconstruct the type on demand. Keep its shared
      // owner alive while walking the child fields.
      const auto type = std::static_pointer_cast<arrow::StructType>(builder->type());
      ARROW_RETURN_NOT_OK(target->Append());
      for (int i = 0; i < type->num_fields(); ++i) {
        ARROW_RETURN_NOT_OK(AppendJson(value.value(type->field(i)->name(), Json()), target->field_builder(i)));
      }
      return arrow::Status::OK();
    }
    case arrow::Type::MAP: {
      auto* target = static_cast<arrow::MapBuilder*>(builder);
      ARROW_RETURN_NOT_OK(target->Append());
      for (auto item = value.begin(); item != value.end(); ++item) {
        ARROW_RETURN_NOT_OK(AppendJson(item.key(), target->key_builder()));
        ARROW_RETURN_NOT_OK(AppendJson(item.value(), target->item_builder()));
      }
      return arrow::Status::OK();
    }
    case arrow::Type::LIST: {
      auto* target = static_cast<arrow::ListBuilder*>(builder);
      ARROW_RETURN_NOT_OK(target->Append());
      for (const auto& item : value) {
        ARROW_RETURN_NOT_OK(AppendJson(item, target->value_builder()));
      }
      return arrow::Status::OK();
    }
    case arrow::Type::STRING:
      return static_cast<arrow::StringBuilder*>(builder)->Append(value.get<std::string>());
    case arrow::Type::INT32:
      return static_cast<arrow::Int32Builder*>(builder)->Append(value.get<int32_t>());
    case arrow::Type::INT64:
      return static_cast<arrow::Int64Builder*>(builder)->Append(value.get<int64_t>());
    case arrow::Type::BOOL:
      return static_cast<arrow::BooleanBuilder*>(builder)->Append(value.get<bool>());
    default:
      return arrow::Status::NotImplemented("Unsupported checkpoint fixture type");
  }
}

// Generate tiny Parquet files and Delta JSON histories at runtime. The same
// source files can exercise fresh snapshots, removals and historical schemas.
class DeltaExplorationTest : public testing::Test {
  protected:
  void SetUp() override {
    ASSERT_AND_ASSIGN(auto temp, arrow::internal::TemporaryDir::Make("delta-explore-"));
    temporary_ = std::move(temp);
    root_ = temporary_->path().ToString();
    properties_ = std::make_shared<api::Properties>(
        api::Properties{{PROPERTY_FS_STORAGE_TYPE, std::string("local")}, {PROPERTY_FS_ROOT_PATH, root_}});
  }

  std::string Table(const std::string& name = "table") {
    auto location = root_ + "/" + name;
    std::filesystem::create_directories(location + "/_delta_log");
    return location;
  }

  Json Schema(const std::string& name = "id", const std::string& type = "long", Json metadata = Json::object()) {
    return Json{
        {"type", "struct"},
        {"fields", Json::array({Json{{"name", name}, {"type", type}, {"nullable", true}, {"metadata", metadata}}})}};
  }

  Json Metadata(const Json& schema, Json partitions = Json::array(), Json configuration = Json::object()) {
    return Json{{"metaData", Json{{"id", "bf692c52-1384-41b8-a8d0-99413a3400bc"},
                                  {"format", Json{{"provider", "parquet"}, {"options", Json::object()}}},
                                  {"schemaString", schema.dump()},
                                  {"partitionColumns", partitions},
                                  {"configuration", configuration},
                                  {"createdTime", 0}}}};
  }

  Json Add(const std::string& table, const std::string& name, int64_t rows, bool with_stats = true) {
    Json add{{"path", name},
             {"partitionValues", Json::object()},
             {"size", std::filesystem::file_size(table + "/" + name)},
             {"modificationTime", 0},
             {"dataChange", true}};
    if (with_stats) {
      add["stats"] = Json{{"numRecords", rows}}.dump();
    }
    return Json{{"add", add}};
  }

  void Commit(const std::string& table, int version, const std::vector<Json>& actions) {
    std::ostringstream name;
    name << table << "/_delta_log/" << std::setw(20) << std::setfill('0') << version << ".json";
    std::ofstream output(name.str());
    ASSERT_TRUE(output.good());
    for (const auto& action : actions) {
      output << action.dump() << '\n';
    }
    output.close();
    ASSERT_TRUE(output.good());
  }

  void WriteData(const std::string& path,
                 int64_t rows = 4,
                 const std::string& name = "id",
                 const std::shared_ptr<arrow::DataType>& type = arrow::int64(),
                 const std::shared_ptr<const arrow::KeyValueMetadata>& metadata = nullptr) {
    arrow::Int64Builder builder;
    for (int64_t i = 0; i < rows; ++i) {
      ASSERT_STATUS_OK(builder.Append(i));
    }
    std::shared_ptr<arrow::Array> data;
    ASSERT_STATUS_OK(builder.Finish(&data));
    if (type->id() == arrow::Type::INT32) {
      arrow::Int32Builder narrowed;
      for (int64_t i = 0; i < rows; ++i) {
        ASSERT_STATUS_OK(narrowed.Append(static_cast<int32_t>(i)));
      }
      ASSERT_STATUS_OK(narrowed.Finish(&data));
    }
    auto schema = arrow::schema({arrow::field(name, type, true, metadata)});
    auto batch = arrow::RecordBatch::Make(schema, rows, {data});
    ASSERT_AND_ASSIGN(auto sink, arrow::io::FileOutputStream::Open(path));
    ASSERT_AND_ASSIGN(auto writer, ::parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), sink));
    ASSERT_STATUS_OK(writer->WriteRecordBatch(*batch));
    ASSERT_STATUS_OK(writer->Close());
    ASSERT_STATUS_OK(sink->Close());
  }

  void Checkpoint(const std::string& table, const std::vector<Json>& actions, bool parsed_stats) {
    const auto strings = arrow::map(arrow::utf8(), arrow::utf8());
    const auto protocol = arrow::struct_({arrow::field("minReaderVersion", arrow::int32()),
                                          arrow::field("minWriterVersion", arrow::int32()),
                                          arrow::field("readerFeatures", arrow::list(arrow::utf8())),
                                          arrow::field("writerFeatures", arrow::list(arrow::utf8()))});
    const auto metadata = arrow::struct_(
        {arrow::field("id", arrow::utf8()), arrow::field("name", arrow::utf8()),
         arrow::field("description", arrow::utf8()),
         arrow::field("format",
                      arrow::struct_({arrow::field("provider", arrow::utf8()), arrow::field("options", strings)})),
         arrow::field("schemaString", arrow::utf8()), arrow::field("partitionColumns", arrow::list(arrow::utf8())),
         arrow::field("configuration", strings), arrow::field("createdTime", arrow::int64())});
    arrow::FieldVector add_fields{arrow::field("path", arrow::utf8()), arrow::field("partitionValues", strings),
                                  arrow::field("size", arrow::int64()),
                                  arrow::field("modificationTime", arrow::int64()),
                                  arrow::field("dataChange", arrow::boolean())};
    // Omit the unused representation physically, as checkpoint writers do.
    // This also exercises the default reader's missing-nullable-field handling.
    add_fields.push_back(
        parsed_stats ? arrow::field("stats_parsed", arrow::struct_({arrow::field("numRecords", arrow::int64())}))
                     : arrow::field("stats", arrow::utf8()));
    const auto add = arrow::struct_(std::move(add_fields));
    auto schema = arrow::schema(
        {arrow::field("protocol", protocol), arrow::field("metaData", metadata), arrow::field("add", add)});
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (const auto& field : schema->fields()) {
      ASSERT_AND_ASSIGN(auto builder, arrow::MakeBuilder(field->type()));
      for (const auto& action : actions) {
        ASSERT_STATUS_OK(AppendJson(action.value(field->name(), Json()), builder.get()));
      }
      std::shared_ptr<arrow::Array> column;
      ASSERT_STATUS_OK(builder->Finish(&column));
      columns.push_back(std::move(column));
    }
    auto batch = arrow::RecordBatch::Make(schema, actions.size(), std::move(columns));
    ASSERT_AND_ASSIGN(auto sink,
                      arrow::io::FileOutputStream::Open(table + "/_delta_log/00000000000000000000.checkpoint.parquet"));
    ASSERT_AND_ASSIGN(auto writer, ::parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), sink));
    ASSERT_STATUS_OK(writer->WriteRecordBatch(*batch));
    ASSERT_STATUS_OK(writer->Close());
    ASSERT_STATUS_OK(sink->Close());
    std::ofstream marker(table + "/_delta_log/_last_checkpoint");
    marker << Json{{"version", 0}, {"size", actions.size()}}.dump();
  }

  arrow::Result<std::vector<api::ColumnGroupFile>> Explore(const std::string& table) {
    ARROW_ASSIGN_OR_RAISE(auto* format, Format::get(LOON_FORMAT_DELTALAKE_TABLE));
    return format->explore(table, properties_);
  }

  void SelectVersion(int64_t version) {
    auto updated = std::make_shared<api::Properties>(*properties_);
    (*updated)[PROPERTY_READER_EXTTABLE_SNAPSHOT_ID] = version;
    properties_ = std::move(updated);
  }

  Json Protocol() { return Json{{"protocol", Json{{"minReaderVersion", 1}, {"minWriterVersion", 2}}}}; }

  Json WideningProtocol() {
    const auto features = Json::array({"typeWidening", "timestampNtz"});
    return Json{{"protocol", Json{{"minReaderVersion", 3},
                                  {"minWriterVersion", 7},
                                  {"readerFeatures", features},
                                  {"writerFeatures", features}}}};
  }

  Json TypeChange(const std::string& from, const std::string& to) {
    return Json{{"delta.typeChanges", Json::array({Json{{"fromType", from}, {"toType", to}}})}};
  }

  std::unique_ptr<arrow::internal::TemporaryDir> temporary_;
  std::string root_;
  std::shared_ptr<api::Properties> properties_;
};

TEST_F(DeltaExplorationTest, MinimalMetadataPinsLatestAndVersionZero) {
  auto table = Table();
  WriteData(table + "/old.parquet");
  WriteData(table + "/orphan.parquet");
  Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "old.parquet", 4)});
  ASSERT_AND_ASSIGN(auto old_files, Explore(table));
  ASSERT_EQ(old_files.size(), 1);
  const auto original = Json::parse(old_files[0].Get<std::string>(api::kPropertyMetadata));
  EXPECT_EQ(original.size(), 3);
  EXPECT_EQ(original["version"], 1);
  EXPECT_EQ(original["snapshot_version"], 0);
  EXPECT_EQ(old_files[0].end_index, 4);
  EXPECT_GT(old_files[0].Get<uint64_t>(api::kPropertyFileSize), 0);

  WriteData(table + "/new.parquet", 2);
  Commit(table, 1,
         {Json{{"remove", Json{{"path", "old.parquet"}, {"dataChange", true}, {"deletionTimestamp", 1}}}},
          Add(table, "new.parquet", 2)});
  ASSERT_AND_ASSIGN(auto latest, Explore(table));
  ASSERT_EQ(latest.size(), 1);
  EXPECT_NE(latest[0].path.find("new.parquet"), std::string::npos);
  EXPECT_EQ(Json::parse(latest[0].Get<std::string>(api::kPropertyMetadata))["snapshot_version"], 1);
  EXPECT_EQ(Json::parse(old_files[0].Get<std::string>(api::kPropertyMetadata)), original);

  SelectVersion(0);
  ASSERT_AND_ASSIGN(auto historical, Explore(table));
  ASSERT_EQ(historical.size(), 1);
  EXPECT_EQ(historical[0].path, old_files[0].path);
  SelectVersion(99);
  EXPECT_FALSE(Explore(table).ok());
}

TEST_F(DeltaExplorationTest, FooterFallbackAndLiteralPathCharacters) {
  auto table = Table("literal%2F space \xE9\x9B\xAA");
  WriteData(table + "/part.parquet", 3);
  Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "part.parquet", 3, false)});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_EQ(files.size(), 1);
  EXPECT_EQ(files[0].end_index, 3);
  EXPECT_NE(files[0].path.find("literal%252F%20space%20"), std::string::npos);
  ASSERT_TRUE(std::filesystem::remove(table + "/part.parquet"));
  EXPECT_FALSE(Explore(table).ok());  // Without a count, exploration needs the footer.
}

TEST_F(DeltaExplorationTest, SnapshotCountsAvoidDataFileAccess) {
  auto table = Table();
  WriteData(table + "/part.parquet");
  auto add = Add(table, "part.parquet", 4);
  ASSERT_TRUE(std::filesystem::remove(table + "/part.parquet"));
  Commit(table, 0, {Protocol(), Metadata(Schema()), add});
  // A valid snapshot count must not trigger a data-file HEAD or footer read.
  // Source retention/existence is checked when the Stage 2 reader opens.
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_EQ(files.size(), 1);
  EXPECT_EQ(files[0].end_index, 4);
}

TEST_F(DeltaExplorationTest, EncodedDataPathsAndLocalRootLocations) {
  auto table = Table();
  const std::string name = "part \xE9\x9B\xAA %2F.parquet";
  WriteData(table + "/" + name, 3);
  auto add = Add(table, name, 3, false);
  add["add"]["path"] = "part%20%E9%9B%AA%20%252F.parquet";
  Commit(table, 0, {Protocol(), Metadata(Schema()), add});
  ASSERT_AND_ASSIGN(auto relative, Explore("table"));
  ASSERT_AND_ASSIGN(auto absolute, Explore(table));
  ASSERT_AND_ASSIGN(auto uri, Explore("file://" + table));
  ASSERT_EQ(relative.size(), 1);
  ASSERT_EQ(absolute.size(), 1);
  ASSERT_EQ(uri.size(), 1);
  EXPECT_EQ(relative[0].path, absolute[0].path);
  EXPECT_EQ(uri[0].path, absolute[0].path);
  EXPECT_NE(relative[0].path.find("part%20%E9%9B%AA%20%252F.parquet"), std::string::npos);
  EXPECT_EQ(relative[0].end_index, 3);
}

TEST_F(DeltaExplorationTest, EmptyLocalRootUsesCurrentDirectory) {
  // Keep the source under cwd without changing the process-wide directory.
  auto temporary_path = std::filesystem::path(root_);
  if (temporary_path.filename().empty()) {
    temporary_path = temporary_path.parent_path();
  }
  auto table = std::filesystem::current_path() / temporary_path.filename();
  ASSERT_TRUE(std::filesystem::create_directory(table));
  auto cleanup = folly::makeGuard([&] {
    std::error_code error;
    std::filesystem::remove_all(table, error);
  });
  ASSERT_TRUE(std::filesystem::create_directory(table / "_delta_log"));
  WriteData((table / "part.parquet").string());
  Commit(table.string(), 0, {Protocol(), Metadata(Schema()), Add(table.string(), "part.parquet", 4)});
  auto properties = std::make_shared<api::Properties>(*properties_);
  (*properties)[PROPERTY_FS_ROOT_PATH] = std::string();
  properties_ = std::move(properties);
  ASSERT_AND_ASSIGN(auto relative, Explore(table.filename().string()));
  ASSERT_AND_ASSIGN(auto absolute, Explore(table.string()));
  ASSERT_AND_ASSIGN(auto uri, Explore("file://" + table.string()));
  ASSERT_EQ(relative.size(), 1);
  ASSERT_EQ(absolute.size(), 1);
  ASSERT_EQ(uri.size(), 1);
  EXPECT_EQ(relative[0].path, absolute[0].path);
  EXPECT_EQ(uri[0].path, absolute[0].path);
  EXPECT_EQ(relative[0].end_index, 4);
}

TEST_F(DeltaExplorationTest, RemoteKeysUseTheBoundFilesystemAndRetainItsLease) {
  auto table = Table();
  WriteData(table + "/part.parquet");
  Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "part.parquet", 4, false)});
  // Use a local subtree as a bucket to exercise remote object keys without a
  // provider SDK or credentials. The fallback count also reads through it.
  std::shared_ptr<arrow::fs::FileSystem> filesystem =
      std::make_shared<FileSystemProxy>(root_, std::make_shared<arrow::fs::LocalFileSystem>());
  std::weak_ptr<arrow::fs::FileSystem> lease = filesystem;
  ASSERT_AND_ASSIGN(auto snapshot, deltalake::BlockingDeltaSnapshot::Open("s3://bucket/table/", -1, filesystem,
                                                                          {false, "s3", "bucket", ""}));
  filesystem.reset();
  EXPECT_FALSE(lease.expired());
  EXPECT_EQ(snapshot->Version(), 0);
  const auto files = snapshot->Files();
  ASSERT_EQ(files.size(), 1);
  EXPECT_EQ(files[0].path, "s3://bucket/table/part.parquet");
  EXPECT_EQ(files[0].row_count, 4);
  snapshot.reset();
  EXPECT_TRUE(lease.expired());
}

TEST_F(DeltaExplorationTest, FilesystemErrorsSurviveSnapshotResolution) {
  auto table = Table();
  Commit(table, 0, {Protocol(), Metadata(Schema())});
  for (auto code : {ExtendStatusCode::AwsErrorAccessDenied, ExtendStatusCode::StorageTransientThrottling}) {
    auto filesystem = std::make_shared<FailingDeltaFileSystem>(root_, code);
    auto result =
        deltalake::BlockingDeltaSnapshot::Open("file://" + table + "/", -1, filesystem, {true, "file", "", root_});
    ASSERT_FALSE(result.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
    ASSERT_NE(detail, nullptr) << result.status();
    EXPECT_EQ(detail->code(), code);
  }
}

TEST_F(DeltaExplorationTest, OpensCheckpointWithoutOriginalCommit) {
  for (int checkpoint_kind : {0, 1, 2}) {
    const bool parsed_stats = checkpoint_kind != 0;
    const bool partition_only = checkpoint_kind == 2;
    SCOPED_TRACE(checkpoint_kind);
    auto table = Table("checkpoint-" + std::to_string(checkpoint_kind));
    WriteData(table + "/part.parquet");
    auto add = Add(table, "part.parquet", 4, !parsed_stats);
    auto configuration = Json::object();
    if (parsed_stats) {
      add["add"]["stats_parsed"] = Json{{"numRecords", 4}};
      configuration =
          Json{{"delta.checkpoint.writeStatsAsJson", "false"}, {"delta.checkpoint.writeStatsAsStruct", "true"}};
    }
    auto partitions = Json::array();
    if (partition_only) {
      partitions = Json::array({"id"});
      add["add"]["partitionValues"] = Json{{"id", "7"}};
    }
    Checkpoint(table, {Protocol(), Metadata(Schema(), partitions, configuration), add}, parsed_stats);
    ASSERT_TRUE(std::filesystem::remove(table + "/part.parquet"));
    ASSERT_AND_ASSIGN(auto files, Explore(table));
    ASSERT_EQ(files.size(), 1);
    EXPECT_EQ(files[0].end_index, 4);
    EXPECT_EQ(Json::parse(files[0].Get<std::string>(api::kPropertyMetadata))["snapshot_version"], 0);
  }
}

TEST_F(DeltaExplorationTest, MappedPartitionsAndAddedNullableFields) {
  const auto mapping = Json{{"delta.columnMapping.id", 1}, {"delta.columnMapping.physicalName", "physical_id"}};
  auto schema = Schema("renamed_id", "long", mapping);
  schema["fields"].push_back(Json{
      {"name", "country"},
      {"type", "string"},
      {"nullable", true},
      {"metadata", Json{{"delta.columnMapping.id", 2}, {"delta.columnMapping.physicalName", "physical_country"}}}});
  schema["fields"].push_back(
      Json{{"name", "new_field"},
           {"type", "long"},
           {"nullable", true},
           {"metadata", Json{{"delta.columnMapping.id", 3}, {"delta.columnMapping.physicalName", "physical_new"}}}});
  const auto protocol = Json{{"protocol", Json{{"minReaderVersion", 2}, {"minWriterVersion", 5}}}};
  for (const auto& mode : {"name", "id"}) {
    auto table = Table(mode);
    WriteData(table + "/part.parquet", 4, "physical_id", arrow::int64(),
              arrow::key_value_metadata({"PARQUET:field_id"}, {"1"}));
    auto add = Add(table, "part.parquet", 4);
    add["add"]["partitionValues"] = Json{{"physical_country", "apac"}};
    Commit(table, 0,
           {protocol,
            Metadata(schema, Json::array({"country"}),
                     Json{{"delta.columnMapping.mode", mode}, {"delta.columnMapping.maxColumnId", "3"}}),
            add});
    ASSERT_AND_ASSIGN(auto files, Explore(table));
    ASSERT_EQ(files.size(), 1);
    EXPECT_EQ(files[0].end_index, 4);
  }
}

TEST_F(DeltaExplorationTest, DeletionCardinalityAndAllDeletedFiles) {
  auto table = Table();
  WriteData(table + "/part.parquet");
  auto add = Add(table, "part.parquet", 4);
  // Stage 1 uses the descriptor, not the bitmap payload. An absolute DV
  // reference also exercises filesystem identity validation without decoding.
  add["add"]["deletionVector"] = Json{{"storageType", "p"},
                                      {"pathOrInlineDv", "file://" + table + "/deletions.bin"},
                                      {"offset", 0},
                                      {"sizeInBytes", 32},
                                      {"cardinality", 2}};
  ASSERT_TRUE(std::filesystem::remove(table + "/part.parquet"));
  const auto protocol = Json{{"protocol", Json{{"minReaderVersion", 3},
                                               {"minWriterVersion", 7},
                                               {"readerFeatures", Json::array({"deletionVectors"})},
                                               {"writerFeatures", Json::array({"deletionVectors"})}}}};
  Commit(table, 0, {protocol, Metadata(Schema()), add});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_EQ(files.size(), 1);
  EXPECT_EQ(files[0].end_index, 2);
  add["add"]["deletionVector"]["cardinality"] = 4;
  Commit(table, 0, {protocol, Metadata(Schema()), add});
  ASSERT_AND_ASSIGN(auto deleted, Explore(table));
  EXPECT_TRUE(deleted.empty());
  add["add"]["deletionVector"]["cardinality"] = 5;
  Commit(table, 0, {protocol, Metadata(Schema()), add});
  EXPECT_FALSE(Explore(table).ok());
  add["add"]["deletionVector"]["cardinality"] = -1;
  Commit(table, 0, {protocol, Metadata(Schema()), add});
  EXPECT_TRUE(Explore(table).status().IsInvalid());
  add["add"]["deletionVector"]["cardinality"] = 2;
  add["add"]["deletionVector"]["pathOrInlineDv"] = "s3://other/deletions.bin";
  Commit(table, 0, {protocol, Metadata(Schema()), add});
  EXPECT_TRUE(Explore(table).status().IsNotImplemented());
}

TEST_F(DeltaExplorationTest, AcceptsRecordedDataWideningWithOlderPhysicalFiles) {
  for (bool with_stats : {false, true}) {
    SCOPED_TRACE(with_stats ? "snapshot count" : "footer count");
    SelectVersion(-1);
    auto table = Table(with_stats ? "widened-with-stats" : "widened-without-stats");
    WriteData(table + "/part.parquet", 4, "id", arrow::int32());
    Commit(table, 0,
           {WideningProtocol(), Metadata(Schema("id", "integer")), Add(table, "part.parquet", 4, with_stats)});
    // Disabling future widening does not remove the history needed to read old files.
    Commit(table, 1,
           {Metadata(Schema("id", "long", TypeChange("integer", "long")), Json::array(),
                     Json{{"delta.enableTypeWidening", "false"}})});
    ASSERT_AND_ASSIGN(auto files, Explore(table));
    ASSERT_EQ(files.size(), 1);
    EXPECT_EQ(files[0].end_index, 4);
    EXPECT_EQ(Json::parse(files[0].Get<std::string>(api::kPropertyMetadata))["snapshot_version"], 1);
    SelectVersion(0);
    EXPECT_TRUE(Explore(table).ok());
  }
}

TEST_F(DeltaExplorationTest, RejectsUnsupportedAndMalformedTypeChangeHistory) {
  auto table = Table();
  Commit(table, 0, {WideningProtocol(), Metadata(Schema("id", "integer", TypeChange("long", "integer")))});
  EXPECT_TRUE(Explore(table).status().IsNotImplemented());
  Commit(table, 0, {WideningProtocol(), Metadata(Schema("id", "long", Json{{"delta.typeChanges", "invalid"}}))});
  EXPECT_TRUE(Explore(table).status().IsInvalid());
}

TEST_F(DeltaExplorationTest, RejectsUnsupportedMetadataEvenWithoutFiles) {
  auto table = Table();
  auto metadata = Metadata(Schema());
  metadata["metaData"]["format"]["provider"] = "orc";
  Commit(table, 0, {Protocol(), metadata});
  EXPECT_TRUE(Explore(table).status().IsNotImplemented());

  const auto unknown = Json::array({"unsupportedReaderFeature"});
  const auto protocol = Json{{"protocol", Json{{"minReaderVersion", 3},
                                               {"minWriterVersion", 7},
                                               {"readerFeatures", unknown},
                                               {"writerFeatures", unknown}}}};
  Commit(table, 0, {protocol, Metadata(Schema())});
  EXPECT_FALSE(Explore(table).ok());

  // Validate the reader subset even when there are no active data files.
  auto schema = Schema();
  schema["fields"][0]["type"] = Json{{"type", "array"}, {"elementType", "long"}, {"containsNull", true}};
  Commit(table, 0, {Protocol(), Metadata(schema, Json::array({"id"}))});
  EXPECT_TRUE(Explore(table).status().IsNotImplemented());
  Commit(table, 0, {Protocol(), Metadata(Schema("id", "variant"))});
  EXPECT_FALSE(Explore(table).ok());
}

TEST_F(DeltaExplorationTest, NestedTypeChangeHistoryIsValidatedWithoutDataReads) {
  auto table = Table();
  WriteData(table + "/part.parquet");
  auto add = Add(table, "part.parquet", 4);
  ASSERT_TRUE(std::filesystem::remove(table + "/part.parquet"));
  auto schema = Schema();
  const auto map = Json{{"type", "map"}, {"keyType", "string"}, {"valueType", "long"}, {"valueContainsNull", true}};
  const auto array = Json{{"type", "array"}, {"elementType", map}, {"containsNull", true}};
  auto history = Json::array({Json{{"fromType", "short"}, {"toType", "integer"}, {"fieldPath", "element.value"}},
                              Json{{"fromType", "integer"}, {"toType", "long"}, {"fieldPath", "element.value"}}});
  schema["fields"].push_back(Json{
      {"name", "nested"}, {"type", array}, {"nullable", true}, {"metadata", Json{{"delta.typeChanges", history}}}});
  Commit(table, 0, {WideningProtocol(), Metadata(schema), add});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  EXPECT_EQ(files.size(), 1);

  schema["fields"][1]["metadata"]["delta.typeChanges"][0]["fieldPath"] = "element.key";
  Commit(table, 0, {WideningProtocol(), Metadata(schema), add});
  EXPECT_TRUE(Explore(table).status().IsInvalid());

  auto bad = Schema("unprojected", "integer", TypeChange("long", "integer"));
  schema["fields"][1]["type"] = Json{{"type", "array"}, {"elementType", bad}, {"containsNull", true}};
  schema["fields"][1]["metadata"] = Json::object();
  Commit(table, 0, {WideningProtocol(), Metadata(schema), add});
  EXPECT_TRUE(Explore(table).status().IsNotImplemented());
}

TEST_F(DeltaExplorationTest, PartitionWideningRequiresNoAdditionalStringConversion) {
  struct PartitionChange {
    std::string from;
    std::string to;
    std::string value;
    bool supported;
  };
  const std::vector<PartitionChange> changes{{"integer", "long", "42", true},
                                             {"integer", "double", "42", true},
                                             {"float", "double", "1.5", true},
                                             {"decimal(10,2)", "decimal(12,2)", "12.30", true},
                                             {"integer", "decimal(10,0)", "42", true},
                                             {"decimal(10,2)", "decimal(12,3)", "12.300", false},
                                             {"integer", "decimal(12,2)", "42.00", false},
                                             {"date", "timestamp_ntz", "2026-11-14 00:00:00", false}};
  for (size_t i = 0; i < changes.size(); ++i) {
    const auto& change = changes[i];
    SCOPED_TRACE(change.from + " -> " + change.to);
    auto table = Table("partition-" + std::to_string(i));
    WriteData(table + "/part.parquet");
    auto schema = Schema();
    schema["fields"].push_back(
        Json{{"name", "p"}, {"type", change.to}, {"nullable", true}, {"metadata", TypeChange(change.from, change.to)}});
    auto add = Add(table, "part.parquet", 4);
    add["add"]["partitionValues"] = Json{{"p", change.value}};
    Commit(table, 0, {WideningProtocol(), Metadata(schema, Json::array({"p"})), add});
    auto result = Explore(table);
    if (change.supported) {
      ASSERT_TRUE(result.ok()) << result.status();
      ASSERT_EQ(result->size(), 1);
      EXPECT_EQ((*result)[0].end_index, 4);
    } else {
      // Even strings already in the new representation cannot bypass the
      // metadata-level restriction on unsupported partition type changes.
      EXPECT_TRUE(result.status().IsNotImplemented()) << result.status();
    }
  }
}

TEST_F(DeltaExplorationTest, RejectsAmbiguousTimestampPartitions) {
  auto table = Table();
  WriteData(table + "/part.parquet");
  auto schema = Schema();
  schema["fields"].push_back(
      Json{{"name", "ts"}, {"type", "timestamp"}, {"nullable", true}, {"metadata", Json::object()}});
  auto add = Add(table, "part.parquet", 4);
  add["add"]["partitionValues"] = Json{{"ts", "2026-11-14 12:00:00"}};
  Commit(table, 0, {Protocol(), Metadata(schema, Json::array({"ts"})), add});
  EXPECT_TRUE(Explore(table).status().IsNotImplemented());
  add["add"]["partitionValues"]["ts"] = "2026-11-14T12:00:00Z";
  Commit(table, 0, {Protocol(), Metadata(schema, Json::array({"ts"})), add});
  EXPECT_TRUE(Explore(table).ok());
}

TEST_F(DeltaExplorationTest, PartitionValuesUseStrictSnapshotConversions) {
  struct PartitionValue {
    std::string type;
    Json value;
    bool supported;
  };
  const std::vector<PartitionValue> cases{{"byte", "127", true},
                                          {"byte", "128", false},
                                          {"decimal(10,2)", "12.30", true},
                                          {"decimal(10,2)", "12.3", false},
                                          {"decimal(10,2)", "", true},
                                          {"integer", nullptr, true},
                                          {"string", "", true},
                                          {"binary", "AA==", true},
                                          {"date", "2026-11-15", true},
                                          {"date", "invalid", false}};
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& item = cases[i];
    SCOPED_TRACE(item.type + ": " + item.value.dump());
    auto table = Table("partition-value-" + std::to_string(i));
    WriteData(table + "/part.parquet");
    auto schema = Schema();
    schema["fields"].push_back(
        Json{{"name", "p"}, {"type", item.type}, {"nullable", true}, {"metadata", Json::object()}});
    auto add = Add(table, "part.parquet", 4);
    add["add"]["partitionValues"] = Json{{"p", item.value}};
    Commit(table, 0, {Protocol(), Metadata(schema, Json::array({"p"})), add});
    auto result = Explore(table);
    if (item.supported) {
      ASSERT_TRUE(result.ok()) << result.status();
      EXPECT_EQ(result->size(), 1);
    } else {
      EXPECT_TRUE(result.status().IsInvalid()) << result.status();
    }
  }
}

TEST_F(DeltaExplorationTest, RejectsForeignPathsAndInvalidSelectors) {
  auto table = Table();
  WriteData(table + "/part.parquet");
  auto add = Add(table, "part.parquet", 4);
  add["add"]["path"] = "s3://other/table/part.parquet";
  Commit(table, 0, {Protocol(), Metadata(Schema()), add});
  EXPECT_TRUE(Explore(table).status().IsNotImplemented());
  EXPECT_TRUE(Explore("file://" + table + "?version=0").status().IsInvalid());
  SelectVersion(-2);
  EXPECT_TRUE(Explore(table).status().IsInvalid());
}

TEST_F(DeltaExplorationTest, ExplicitFileUrlCannotBeRebasedToRemoteStorage) {
  auto remote = std::make_shared<api::Properties>(*properties_);
  (*remote)[PROPERTY_FS_STORAGE_TYPE] = std::string("remote");
  properties_ = std::move(remote);
  EXPECT_TRUE(Explore("file://" + root_ + "/table").status().IsInvalid());
}

TEST_F(DeltaExplorationTest, EmptyAndMissingTablesAndStageBoundaries) {
  auto table = Table();
  Commit(table, 0, {Protocol(), Metadata(Schema())});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  EXPECT_TRUE(files.empty());
  EXPECT_FALSE(Explore(root_ + "/missing").ok());
  ASSERT_AND_ASSIGN(auto* format, Format::get(LOON_FORMAT_DELTALAKE_TABLE));
  EXPECT_TRUE(format->create_reader(nullptr, {}, properties_, {}, nullptr).status().IsNotImplemented());
  EXPECT_TRUE(format->create_writer(nullptr, nullptr, "", "", properties_).status().IsNotImplemented());
}

}  // namespace
}  // namespace milvus_storage
