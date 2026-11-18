// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <future>

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/file.h>
#include <arrow/util/io_util.h>
#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>
#include <parquet/encryption/encryption.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/async_tasks.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/reader.h"
#include "deltalake/deltalake_bridge.h"
#include "format/deltalake/deltalake_common.h"
#include "milvus-storage/format/deltalake/deltalake_format_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/column_group_reader.h"
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
    case arrow::Type::BINARY:
      return static_cast<arrow::BinaryBuilder*>(builder)->Append(value.get<std::string>());
    case arrow::Type::INT8:
      return static_cast<arrow::Int8Builder*>(builder)->Append(value.get<int8_t>());
    case arrow::Type::INT16:
      return static_cast<arrow::Int16Builder*>(builder)->Append(value.get<int16_t>());
    case arrow::Type::FLOAT:
      return static_cast<arrow::FloatBuilder*>(builder)->Append(value.get<float>());
    case arrow::Type::DOUBLE:
      return static_cast<arrow::DoubleBuilder*>(builder)->Append(value.get<double>());
    case arrow::Type::DATE32:
      return static_cast<arrow::Date32Builder*>(builder)->Append(value.get<int32_t>());
    case arrow::Type::TIMESTAMP:
      return static_cast<arrow::TimestampBuilder*>(builder)->Append(value.get<int64_t>());
    case arrow::Type::DECIMAL128: {
      ARROW_ASSIGN_OR_RAISE(auto decimal, arrow::Decimal128::FromString(value.get<std::string>()));
      return static_cast<arrow::Decimal128Builder*>(builder)->Append(decimal);
    }
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
  ASSERT_AND_ASSIGN(auto location, deltalake::ResolveLocation(table, properties_));
  for (auto code : {ExtendStatusCode::AwsErrorAccessDenied, ExtendStatusCode::StorageTransientThrottling}) {
    auto filesystem = std::make_shared<FailingDeltaFileSystem>(location.config.root_path, code);
    auto result = deltalake::BlockingDeltaSnapshot::Open(location.url, -1, filesystem, location.binding);
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
                                             {"float", "double", "0.1", false},
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
  EXPECT_TRUE(format->create_reader(nullptr, {}, properties_, {}, nullptr).status().IsInvalid());
  EXPECT_TRUE(format->create_writer(nullptr, nullptr, "", "", properties_).status().IsNotImplemented());
}

// Exercise generated Delta histories through the C++ reader and Rust/Kernel
// bridge. Path units below do not open Delta logs or data files.
class DeltaReaderTest : public DeltaExplorationTest {
  protected:
  arrow::Result<std::shared_ptr<FormatReader>> Open(const api::ColumnGroupFile& file,
                                                    std::vector<std::string> columns = {},
                                                    std::shared_ptr<arrow::Schema> schema = nullptr) {
    ARROW_ASSIGN_OR_RAISE(auto* format, Format::get(LOON_FORMAT_DELTALAKE_TABLE));
    return format->create_reader(schema, file, properties_, columns, nullptr);
  }

  void WriteRows(const std::string& path,
                 const std::shared_ptr<arrow::Schema>& schema,
                 const std::vector<Json>& rows,
                 int64_t group_size = 2,
                 std::shared_ptr<::parquet::FileEncryptionProperties> encryption = nullptr) {
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    for (const auto& field : schema->fields()) {
      ASSERT_AND_ASSIGN(auto builder, arrow::MakeBuilder(field->type()));
      for (const auto& row : rows) ASSERT_STATUS_OK(AppendJson(row.value(field->name(), Json()), builder.get()));
      ASSERT_AND_ASSIGN(auto array, builder->Finish());
      arrays.push_back(std::move(array));
    }
    auto table = arrow::Table::Make(schema, arrays, rows.size());
    ASSERT_AND_ASSIGN(auto sink, arrow::io::FileOutputStream::Open(path));
    ::parquet::WriterProperties::Builder properties;
    if (encryption)
      properties.encryption(std::move(encryption));
    ASSERT_STATUS_OK(
        ::parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, group_size, properties.build()));
    ASSERT_STATUS_OK(sink->Close());
  }

  Json Deletions(const std::string& table, const std::vector<uint64_t>& rows) {
    auto dv = deltalake::ffi::delta_test_deletion_vector(rust::Slice<const uint64_t>(rows.data(), rows.size()));
    WriteDvFile(table + "/deletions.bin", {dv.bytes.data(), dv.bytes.size()});
    return Json{{"storageType", "p"},
                {"pathOrInlineDv", "file://" + table + "/deletions.bin"},
                {"offset", dv.offset},
                {"sizeInBytes", dv.size},
                {"cardinality", dv.cardinality}};
  }

  Json DvProtocol() {
    const auto features = Json::array({"deletionVectors"});
    return Json{{"protocol", Json{{"minReaderVersion", 3},
                                  {"minWriterVersion", 7},
                                  {"readerFeatures", features},
                                  {"writerFeatures", features}}}};
  }

  deltalake::ffi::GeneratedDeltaDvContainer GenerateDvs(const std::vector<std::vector<uint64_t>>& rows) {
    rust::Vec<deltalake::ffi::DeltaDvPositions> vectors;
    for (const auto& positions : rows) {
      deltalake::ffi::DeltaDvPositions vector;
      for (auto row : positions) vector.rows.push_back(row);
      vectors.push_back(std::move(vector));
    }
    return deltalake::ffi::delta_test_deletion_vectors(std::move(vectors));
  }

  void WriteDvFile(const std::string& path, std::span<const uint8_t> bytes) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    output.close();
    ASSERT_TRUE(output.good());
  }

  void WriteDvRows(const std::string& table, const std::string& name, int64_t start = 0) {
    std::vector<Json> rows;
    for (int64_t i = 0; i < 6; ++i) rows.push_back(Json{{"id", start + i}});
    WriteRows(table + "/" + name, arrow::schema({arrow::field("id", arrow::int64())}), rows);
  }

  Json DvAdd(const std::string& table, const std::string& name, const Json& descriptor) {
    auto add = Add(table, name, 6);
    add["add"]["deletionVector"] = descriptor;
    return add;
  }

  arrow::Result<std::shared_ptr<FormatReader>> OpenCached(const api::ColumnGroupFile& file,
                                                          const MetadataCache& owner) {
    using Reader = deltalake::DeltaLakeFormatReader;
    using Trait = Reader::MetaTrait;
    auto cache = owner.get<Reader>();
    ARROW_ASSIGN_OR_RAISE(auto metadata, cache->get_or_open(Trait::cache_key(file), [&] {
      return Trait::load_metadata(file, properties_, nullptr);
    }));
    ARROW_ASSIGN_OR_RAISE(auto reader, Trait::create_from_metadata(metadata, file, nullptr, {}, ""));
    return std::shared_ptr<FormatReader>(std::move(reader));
  }

  void CheckDvIds(const std::shared_ptr<arrow::Table>& table, const std::vector<int64_t>& expected) {
    ASSERT_EQ(table->num_columns(), 1);
    ASSERT_EQ(table->num_rows(), expected.size());
    ASSERT_EQ(table->column(0)->type()->id(), arrow::Type::INT64);
    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_AND_ASSIGN(auto value, table->column(0)->GetScalar(i));
      ASSERT_EQ(std::static_pointer_cast<arrow::Int64Scalar>(value)->value, expected[i]);
    }
  }

  void CheckDvReads(const std::shared_ptr<FormatReader>& reader, const std::vector<int64_t>& expected) {
    ASSERT_GE(expected.size(), 2);
    ASSERT_AND_ASSIGN(auto groups, reader->get_row_group_infos());
    std::vector<int> indices;
    for (size_t i = 0; i < groups.size(); ++i) indices.push_back(i);
    ASSERT_AND_ASSIGN(auto chunks, reader->get_chunks(indices));
    ASSERT_AND_ASSIGN(auto chunked, arrow::Table::FromRecordBatches(chunks));
    CheckDvIds(chunked, expected);
    ASSERT_AND_ASSIGN(auto taken, reader->take({0, static_cast<int64_t>(expected.size() - 1)}));
    CheckDvIds(taken, {expected.front(), expected.back()});
    ASSERT_AND_ASSIGN(auto range, reader->read_with_range(1, expected.size()));
    ASSERT_AND_ASSIGN(auto ranged, range->ToTable());
    CheckDvIds(ranged, std::vector<int64_t>(expected.begin() + 1, expected.end()));
  }
};

enum class DeltaDvStorage { Inline, Relative, PrefixedRelative, Absolute };

class DeltaDvStorageTest : public DeltaReaderTest, public ::testing::WithParamInterface<DeltaDvStorage> {
  protected:
  std::string ContainerPath(const std::string& table) {
    if (GetParam() == DeltaDvStorage::Relative || GetParam() == DeltaDvStorage::PrefixedRelative) {
      return table + "/" + (GetParam() == DeltaDvStorage::PrefixedRelative ? "dv/prefix/" : "") +
             "deletion_vector_00000000-0000-0000-0000-000000000000.bin";
    }
    return table + "/deletions.bin";
  }

  Json Descriptor(const std::string& table, const deltalake::ffi::GeneratedDeltaDvDescriptor& dv) {
    Json descriptor{{"sizeInBytes", dv.size}, {"cardinality", dv.cardinality}};
    if (GetParam() == DeltaDvStorage::Inline) {
      descriptor["storageType"] = "i";
      descriptor["pathOrInlineDv"] = std::string(dv.inline_data);
    } else {
      descriptor["offset"] = dv.offset;
      if (GetParam() == DeltaDvStorage::Absolute) {
        descriptor["storageType"] = "p";
        descriptor["pathOrInlineDv"] = "file://" + ContainerPath(table);
      } else {
        // The all-zero UUID encodes to twenty Z85 zeroes; the prefix has no delimiter.
        descriptor["storageType"] = "u";
        descriptor["pathOrInlineDv"] =
            (GetParam() == DeltaDvStorage::PrefixedRelative ? "dv/prefix" : "") + std::string(20, '0');
      }
    }
    return descriptor;
  }

  void WriteContainer(const std::string& table, std::span<const uint8_t> bytes) {
    // Inline cases deliberately have no external DV object to read.
    if (GetParam() != DeltaDvStorage::Inline)
      WriteDvFile(ContainerPath(table), bytes);
  }

  void WriteDvTable(const std::string& table, const Json& descriptor, std::span<const uint8_t> bytes) {
    WriteDvRows(table, "part.parquet");
    WriteContainer(table, bytes);
    Commit(table, 0, {DvProtocol(), Metadata(Schema()), DvAdd(table, "part.parquet", descriptor)});
  }
};

TEST(DeltaPathTest, NormalizesLocalIdentityAndChecksContainmentBeforeOpening) {
  const deltalake::FilesystemBinding binding{true, "file", "", "/bound"};
  const auto resolve = [&](const std::string& path) { return deltalake::ResolveDataFile(path, binding, "", nullptr); };
  for (const auto& url : {"file:///bound/%70art%20a%252F.parquet", "file:///bound/dir/../part%20a%252F.parquet"}) {
    ASSERT_AND_ASSIGN(auto file, resolve(url));
    EXPECT_EQ(file.url, "file:///bound/part%20a%252F.parquet");
    EXPECT_EQ(file.path, "part a%2F.parquet");
  }
  for (const auto& url : {"file:///bound/dir/../../outside.parquet", "file:///bound/%2E%2E/outside.parquet",
                          "file:///bound-other/part.parquet"}) {
    EXPECT_TRUE(resolve(url).status().IsInvalid()) << url;
  }
}

TEST(DeltaRowMappingTest, EnumeratesLivePositionsAcrossDeletionPatterns) {
  for (unsigned mask = 0; mask < (1u << 10); ++mask) {
    std::vector<uint64_t> deleted, live;
    for (uint64_t row = 0; row < 10; ++row) ((mask >> row) & 1 ? deleted : live).push_back(row);
    for (size_t logical = 0; logical < live.size(); ++logical)
      ASSERT_EQ(deltalake::LogicalToPhysical(logical, deleted), live[logical]) << mask;
  }
}

TEST_F(DeltaReaderTest, ReadsEveryMethodWithDeletionVectorsAcrossGroups) {
  const auto table = Table();
  WriteRows(table + "/part.parquet", arrow::schema({arrow::field("id", arrow::int64())}),
            {Json{{"id", 0}}, Json{{"id", 1}}, Json{{"id", 2}}, Json{{"id", 3}}, Json{{"id", 4}}, Json{{"id", 5}}});
  auto add = Add(table, "part.parquet", 6);
  add["add"]["deletionVector"] = Deletions(table, {1, 2, 3});
  Commit(table, 0, {DvProtocol(), Metadata(Schema()), add});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_EQ(files.size(), 1);
  ASSERT_AND_ASSIGN(auto reader, Open(files[0]));
  ASSERT_AND_ASSIGN(auto groups, reader->get_row_group_infos());
  ASSERT_EQ(groups.size(), 3);
  EXPECT_EQ(groups[1].start_offset, groups[1].end_offset);
  ASSERT_AND_ASSIGN(auto chunks, reader->get_chunks({0, 1, 2}));
  EXPECT_EQ(chunks[0]->num_rows(), 1);
  EXPECT_EQ(chunks[1]->num_rows(), 0);
  EXPECT_EQ(chunks[2]->num_rows(), 2);
  ASSERT_AND_ASSIGN(auto selected, reader->take({0, 2}));
  ASSERT_EQ(selected->num_rows(), 2);
  ASSERT_AND_ASSIGN(auto last, selected->column(0)->GetScalar(1));
  EXPECT_EQ(last->ToString(), "5");
  ASSERT_AND_ASSIGN(auto range, reader->read_with_range(1, 3));
  ASSERT_AND_ASSIGN(auto ranged, range->ToTable());
  EXPECT_EQ(ranged->num_rows(), 2);
  ASSERT_AND_ASSIGN(auto first, ranged->column(0)->GetScalar(0));
  EXPECT_EQ(first->ToString(), "4");
  ASSERT_AND_ASSIGN(auto clone, reader->clone_reader());
  ASSERT_AND_ASSIGN(auto clone_chunk, clone->get_chunk(2));
  EXPECT_TRUE(clone_chunk->Equals(*chunks[2]));
  EXPECT_FALSE(reader->take({2, 1}).ok());
  EXPECT_FALSE(reader->take({0, 0}).ok());
  EXPECT_FALSE(reader->take({3}).ok());
  EXPECT_FALSE(reader->get_chunk(-1).ok());
  EXPECT_FALSE(reader->read_with_range(0, 4).ok());
}

TEST_P(DeltaDvStorageTest, ReadsDistinctVectorsWithNormalClonedAndCachedReaders) {
  const auto table = Table();
  auto generated = GenerateDvs({{1, 2, 3}, {0, 4}});
  ASSERT_EQ(generated.descriptors.size(), 2);
  ASSERT_LT(generated.descriptors[0].offset, generated.descriptors[1].offset);
  WriteContainer(table, {generated.bytes.data(), generated.bytes.size()});
  WriteDvRows(table, "a.parquet");
  WriteDvRows(table, "b.parquet", 100);
  Commit(table, 0,
         {DvProtocol(), Metadata(Schema()), DvAdd(table, "a.parquet", Descriptor(table, generated.descriptors[0])),
          DvAdd(table, "b.parquet", Descriptor(table, generated.descriptors[1]))});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_EQ(files.size(), 2);
  MetadataCache cache(true);
  for (const auto& file : files) {
    SCOPED_TRACE(file.path);
    const bool first = file.path.ends_with("/a.parquet");
    const std::vector<int64_t> expected =
        first ? std::vector<int64_t>{0, 4, 5} : std::vector<int64_t>{101, 102, 103, 105};
    EXPECT_EQ(file.end_index, expected.size());
    ASSERT_AND_ASSIGN(auto reader, Open(file));
    CheckDvReads(reader, expected);
    ASSERT_AND_ASSIGN(auto clone, reader->clone_reader());
    CheckDvReads(clone, expected);
    ASSERT_AND_ASSIGN(auto cached, OpenCached(file, cache));
    CheckDvReads(cached, expected);
    ASSERT_AND_ASSIGN(auto reopened, OpenCached(file, cache));
    CheckDvReads(reopened, expected);
  }
}

TEST_P(DeltaDvStorageTest, EmptyVectorKeepsRowsAndFullyDeletedFileIsOmitted) {
  const auto table = Table();
  auto generated = GenerateDvs({{}, {0, 1, 2, 3, 4, 5}});
  WriteContainer(table, {generated.bytes.data(), generated.bytes.size()});
  WriteDvRows(table, "keep.parquet");
  WriteDvRows(table, "deleted.parquet", 100);
  Commit(table, 0,
         {DvProtocol(), Metadata(Schema()), DvAdd(table, "keep.parquet", Descriptor(table, generated.descriptors[0])),
          DvAdd(table, "deleted.parquet", Descriptor(table, generated.descriptors[1]))});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_EQ(files.size(), 1);
  EXPECT_TRUE(files[0].path.ends_with("/keep.parquet"));
  ASSERT_AND_ASSIGN(auto reader, Open(files[0]));
  CheckDvReads(reader, {0, 1, 2, 3, 4, 5});
}

TEST_P(DeltaDvStorageTest, RejectsBitmapCardinalityMismatchAndOutOfBoundsPositions) {
  struct InvalidVector {
    std::vector<uint64_t> rows;
    int64_t cardinality;
    std::string error;
  };
  const std::vector<InvalidVector> cases = {{{1, 2, 3}, 2, "cardinality differs"},
                                            {{1, 2, 3}, 4, "cardinality differs"},
                                            {{6}, 1, "position exceeds"},
                                            {{uint64_t{1} << 32}, 1, "position exceeds"}};
  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    const auto table = Table("invalid-" + std::to_string(i));
    auto generated = GenerateDvs({cases[i].rows});
    auto descriptor = Descriptor(table, generated.descriptors[0]);
    descriptor["cardinality"] = cases[i].cardinality;
    WriteDvTable(table, descriptor, {generated.bytes.data(), generated.bytes.size()});
    ASSERT_AND_ASSIGN(auto files, Explore(table));
    ASSERT_EQ(files.size(), 1);
    EXPECT_EQ(files[0].end_index, 6 - cases[i].cardinality);
    auto opened = Open(files[0]);
    ASSERT_FALSE(opened.ok());
    EXPECT_NE(opened.status().ToString().find(cases[i].error), std::string::npos) << opened.status();
    MetadataCache cache(true);
    EXPECT_FALSE(OpenCached(files[0], cache).ok());
    using Reader = deltalake::DeltaLakeFormatReader;
    EXPECT_FALSE(cache.get<Reader>()->get(Reader::MetaTrait::cache_key(files[0])).has_value());
  }
}

TEST_P(DeltaDvStorageTest, RejectsMalformedAndTruncatedPayloads) {
  enum class Damage { Encoding, InlineBitmap, InlineMagic, TruncatedFile, Checksum, Size, Offset };
  // Persisted frames also carry a size prefix and CRC; inline frames carry neither.
  const auto cases = GetParam() == DeltaDvStorage::Inline
                         ? std::vector<Damage>{Damage::Encoding, Damage::InlineBitmap, Damage::InlineMagic}
                         : std::vector<Damage>{Damage::TruncatedFile, Damage::Checksum, Damage::Size, Damage::Offset};
  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    const auto table = Table("corrupt-" + std::to_string(i));
    auto generated = GenerateDvs({{1, 2, 3}});
    auto descriptor = Descriptor(table, generated.descriptors[0]);
    std::vector<uint8_t> bytes(generated.bytes.begin(), generated.bytes.end());
    switch (cases[i]) {
      case Damage::Encoding:
        descriptor["pathOrInlineDv"] = "~~~";
        break;
      case Damage::InlineBitmap:
        // One Z85 block retains the valid four-byte magic but drops the bitmap.
        descriptor["pathOrInlineDv"] = std::string(generated.descriptors[0].inline_data).substr(0, 5);
        break;
      case Damage::InlineMagic:
        descriptor["pathOrInlineDv"] = "00000";
        break;
      case Damage::TruncatedFile:
        bytes.pop_back();
        break;
      case Damage::Checksum:
        bytes.back() ^= 1;
        break;
      case Damage::Size:
        descriptor["sizeInBytes"] = generated.descriptors[0].size + 1;
        break;
      case Damage::Offset:
        descriptor["offset"] = bytes.size() + 100;
        break;
    }
    WriteDvTable(table, descriptor, bytes);
    // Valid Add counts do not require decoding DVs during exploration.
    ASSERT_AND_ASSIGN(auto files, Explore(table));
    ASSERT_EQ(files.size(), 1);
    EXPECT_EQ(files[0].end_index, 3);
    EXPECT_FALSE(Open(files[0]).ok());
    MetadataCache cache(true);
    EXPECT_FALSE(OpenCached(files[0], cache).ok());
  }
}

INSTANTIATE_TEST_SUITE_P(StorageForms,
                         DeltaDvStorageTest,
                         ::testing::Values(DeltaDvStorage::Inline,
                                           DeltaDvStorage::Relative,
                                           DeltaDvStorage::PrefixedRelative,
                                           DeltaDvStorage::Absolute),
                         [](const ::testing::TestParamInfo<DeltaDvStorage>& info) {
                           switch (info.param) {
                             case DeltaDvStorage::Inline:
                               return "Inline";
                             case DeltaDvStorage::Relative:
                               return "Relative";
                             case DeltaDvStorage::PrefixedRelative:
                               return "PrefixedRelative";
                             case DeltaDvStorage::Absolute:
                               return "Absolute";
                           }
                           return "Unknown";
                         });

TEST_F(DeltaReaderTest, FailedDvReadsRetryWithinAnExistingSnapshotCache) {
  const auto table = Table();
  WriteDvRows(table, "a.parquet");
  WriteDvRows(table, "b.parquet", 100);
  auto generated = GenerateDvs({{1, 2, 3}, {0, 4}});
  const auto healthy = table + "/healthy.bin";
  const auto failing = table + "/recover.bin";
  const std::span<const uint8_t> bytes(generated.bytes.data(), generated.bytes.size());
  WriteDvFile(healthy, bytes);
  auto descriptor = [&](size_t i, const std::string& path) {
    const auto& dv = generated.descriptors[i];
    return Json{{"storageType", "p"},
                {"pathOrInlineDv", "file://" + path},
                {"offset", dv.offset},
                {"sizeInBytes", dv.size},
                {"cardinality", dv.cardinality}};
  };
  Commit(table, 0,
         {DvProtocol(), Metadata(Schema()), DvAdd(table, "a.parquet", descriptor(0, healthy)),
          DvAdd(table, "b.parquet", descriptor(1, failing))});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_EQ(files.size(), 2);
  std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.path < b.path; });
  MetadataCache cache(true);
  ASSERT_AND_ASSIGN(auto first, OpenCached(files[0], cache));
  // The snapshot remains owned by the healthy reader/cache while the other file fails.
  EXPECT_FALSE(OpenCached(files[1], cache).ok());
  auto corrupt = std::vector<uint8_t>(bytes.begin(), bytes.end());
  corrupt.back() ^= 1;
  WriteDvFile(failing, corrupt);
  EXPECT_FALSE(OpenCached(files[1], cache).ok());
  // Restore precisely the planned bytes; both failed loads must remain retryable.
  WriteDvFile(failing, bytes);
  ASSERT_AND_ASSIGN(auto recovered, OpenCached(files[1], cache));
  CheckDvReads(recovered, {101, 102, 103, 105});
  CheckDvReads(first, {0, 4, 5});
  ASSERT_AND_ASSIGN(auto clone, recovered->clone_reader());
  CheckDvReads(clone, {101, 102, 103, 105});
}

TEST_F(DeltaReaderTest, MappedWidenedMissingAndPartitionColumns) {
  for (const auto& mode : {"name", "id"}) {
    const auto table = Table(mode);
    WriteData(table + "/part.parquet", 4, "physical_id", arrow::int32(),
              arrow::key_value_metadata({"PARQUET:field_id"}, {"1"}));
    auto schema = Schema("id", "long", TypeChange("integer", "long"));
    schema["fields"][0]["metadata"]["delta.columnMapping.id"] = 1;
    schema["fields"][0]["metadata"]["delta.columnMapping.physicalName"] = "physical_id";
    schema["fields"].push_back(Json{
        {"name", "region"},
        {"type", "string"},
        {"nullable", true},
        {"metadata", Json{{"delta.columnMapping.id", 2}, {"delta.columnMapping.physicalName", "physical_region"}}}});
    schema["fields"].push_back(
        Json{{"name", "note"},
             {"type", "string"},
             {"nullable", true},
             {"metadata", Json{{"delta.columnMapping.id", 3}, {"delta.columnMapping.physicalName", "physical_note"}}}});
    auto add = Add(table, "part.parquet", 4);
    add["add"]["partitionValues"] = Json{{"physical_region", "apac"}};
    auto protocol = WideningProtocol();
    protocol["protocol"]["readerFeatures"].push_back("columnMapping");
    protocol["protocol"]["writerFeatures"].push_back("columnMapping");
    Commit(table, 0,
           {protocol,
            Metadata(schema, Json::array({"region"}),
                     Json{{"delta.columnMapping.mode", mode}, {"delta.columnMapping.maxColumnId", "3"}}),
            add});
    ASSERT_AND_ASSIGN(auto files, Explore(table));
    ASSERT_AND_ASSIGN(auto reader, Open(files[0], {"note", "id", "region"}));
    EXPECT_EQ(reader->get_schema()->field_names(), (std::vector<std::string>{"id", "region", "note"}));
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
    EXPECT_EQ(batch->schema()->field_names(), (std::vector<std::string>{"note", "id", "region"}));
    EXPECT_EQ(batch->column(0)->null_count(), 4);
    EXPECT_EQ(batch->column(1)->type_id(), arrow::Type::INT64);
    ASSERT_AND_ASSIGN(auto region, batch->column(2)->GetScalar(0));
    EXPECT_EQ(region->ToString(), "apac");
    ASSERT_AND_ASSIGN(auto constants, Open(files[0], {"region", "note"}));
    ASSERT_AND_ASSIGN(auto constant_rows, constants->take({1, 3}));
    EXPECT_EQ(constant_rows->num_rows(), 2);
    ASSERT_AND_ASSIGN(auto zero, Open(files[0], {}, arrow::schema({})));
    ASSERT_AND_ASSIGN(auto zero_batch, zero->get_chunk(0));
    EXPECT_EQ(zero_batch->num_columns(), 0);
    EXPECT_EQ(zero_batch->num_rows(), 4);
    EXPECT_FALSE(Open(files[0], {}, arrow::schema({arrow::field("id", arrow::int32())})).ok());
    EXPECT_FALSE(Open(files[0], {"unknown"}).ok());
    EXPECT_FALSE(
        Open(files[0], {}, arrow::schema({arrow::field("id", arrow::int64()), arrow::field("id", arrow::int64())}))
            .ok());
  }
}

TEST_F(DeltaReaderTest, ConstantOnlyReadsPreserveLiveRangesAndValidation) {
  const auto table = Table();
  std::vector<Json> rows;
  for (int64_t id = 0; id < 8; ++id) rows.push_back(Json{{"id", id}});
  WriteRows(table + "/part.parquet", arrow::schema({arrow::field("id", arrow::int64())}), rows);
  auto schema = Schema();
  schema["fields"].push_back(Schema("region", "string")["fields"][0]);
  schema["fields"].push_back(Schema("note", "string")["fields"][0]);
  auto add = Add(table, "part.parquet", 8);
  add["add"]["partitionValues"] = Json{{"region", "apac"}};
  add["add"]["deletionVector"] = Deletions(table, {0, 1, 3, 4, 5});
  Commit(table, 0, {DvProtocol(), Metadata(schema, Json::array({"region"})), add});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_EQ(files.size(), 1);
  const std::vector<int64_t> live_rows{0, 1, 0, 2};
  struct RangeCase {
    uint64_t start, end;
    std::vector<int64_t> batch_rows;
  };
  for (const auto& names : std::vector<std::vector<std::string>>{{"region"}, {"note"}, {"region", "note"}, {}}) {
    SCOPED_TRACE(::testing::PrintToString(names));
    ASSERT_AND_ASSIGN(auto reader, Open(files[0], names, names.empty() ? arrow::schema({}) : nullptr));
    const auto check = [&](const std::shared_ptr<arrow::RecordBatch>& batch, int64_t expected_rows) {
      ASSERT_NE(batch, nullptr);
      ASSERT_EQ(batch->num_rows(), expected_rows);
      ASSERT_EQ(batch->schema()->field_names(), names);
      for (size_t column = 0; column < names.size(); ++column) {
        if (names[column] == "note") {
          EXPECT_EQ(batch->column(column)->null_count(), expected_rows);
        } else {
          const auto values = std::static_pointer_cast<arrow::StringArray>(batch->column(column));
          for (int64_t row = 0; row < expected_rows; ++row) EXPECT_EQ(values->GetString(row), "apac");
        }
      }
    };
    ASSERT_AND_ASSIGN(auto groups, reader->get_row_group_infos());
    ASSERT_EQ(groups.size(), live_rows.size());
    for (size_t group = 0; group < groups.size(); ++group) {
      EXPECT_EQ(groups[group].end_offset - groups[group].start_offset, live_rows[group]);
      EXPECT_EQ(groups[group].read_size, 0);
      EXPECT_FALSE(groups[group].memory_size_available);
      ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(group));
      check(batch, live_rows[group]);
    }
    // Leading and middle deleted groups repeat offsets. Ranges must retain the
    // original live-group batch boundaries while materializing only their slice.
    for (const auto& test : {RangeCase{0, 3, {1, 2}}, RangeCase{0, 2, {1, 1}}, RangeCase{1, 2, {1}},
                             RangeCase{1, 3, {2}}, RangeCase{0, 0, {}}, RangeCase{1, 1, {}}, RangeCase{3, 3, {}}}) {
      SCOPED_TRACE(::testing::Message() << test.start << ':' << test.end);
      ASSERT_AND_ASSIGN(auto range, reader->read_with_range(test.start, test.end));
      std::shared_ptr<arrow::RecordBatch> batch;
      for (auto expected_rows : test.batch_rows) {
        ASSERT_STATUS_OK(range->ReadNext(&batch));
        check(batch, expected_rows);
      }
      ASSERT_STATUS_OK(range->ReadNext(&batch));
      EXPECT_EQ(batch, nullptr);
    }
    ASSERT_AND_ASSIGN(auto selected, reader->take({0, 2}));
    ASSERT_AND_ASSIGN(auto selected_batch, selected->CombineChunksToBatch());
    check(selected_batch, 2);
    ASSERT_AND_ASSIGN(auto empty, reader->take({}));
    EXPECT_EQ(empty->num_rows(), 0);
    EXPECT_EQ(empty->schema()->field_names(), names);
    EXPECT_TRUE(reader->take({-1}).status().IsInvalid());
    EXPECT_TRUE(reader->take({1, 1}).status().IsInvalid());
    EXPECT_TRUE(reader->take({2, 0}).status().IsInvalid());
    EXPECT_TRUE(reader->take({3}).status().IsInvalid());
    EXPECT_TRUE(reader->read_with_range(2, 1).status().IsInvalid());
    EXPECT_TRUE(reader->read_with_range(0, 4).status().IsInvalid());
  }
}

TEST_F(DeltaReaderTest, ProjectedReadBytesDriveSchedulingWithoutScalingForDeletions) {
  const auto table = Table();
  std::vector<Json> rows;
  for (int64_t id = 0; id < 8; ++id) rows.push_back(Json{{"id", id}, {"payload", "payload-" + std::to_string(id)}});
  const auto physical = arrow::schema({arrow::field("id", arrow::int64()), arrow::field("payload", arrow::utf8())});
  WriteRows(table + "/part.parquet", physical, rows);
  auto schema = Schema();
  schema["fields"].push_back(Schema("payload", "string")["fields"][0]);
  auto add = Add(table, "part.parquet", 8);
  add["add"]["deletionVector"] = Deletions(table, {0, 1, 3, 4, 5});
  Commit(table, 0, {DvProtocol(), Metadata(schema), add});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_EQ(files.size(), 1);
  ASSERT_AND_ASSIGN(auto input, arrow::io::ReadableFile::Open(table + "/part.parquet"));
  const auto footer = ::parquet::ParquetFileReader::Open(input)->metadata();
  ASSERT_AND_ASSIGN(auto narrow, Open(files[0], {"id"}));
  ASSERT_AND_ASSIGN(auto wide, Open(files[0], {"id", "payload"}));
  ASSERT_AND_ASSIGN(auto narrow_groups, narrow->get_row_group_infos());
  ASSERT_AND_ASSIGN(auto wide_groups, wide->get_row_group_infos());
  ASSERT_EQ(narrow_groups.size(), 4);
  ASSERT_EQ(wide_groups.size(), narrow_groups.size());
  for (size_t group = 0; group < narrow_groups.size(); ++group) {
    const bool deleted = group == 0 || group == 2;
    const auto row_group = footer->RowGroup(group);
    const auto id_bytes = row_group->ColumnChunk(0)->total_compressed_size();
    const auto payload_bytes = row_group->ColumnChunk(1)->total_compressed_size();
    EXPECT_EQ(narrow_groups[group].read_size, deleted ? 0 : id_bytes);
    EXPECT_EQ(wide_groups[group].read_size, deleted ? 0 : id_bytes + payload_bytes);
    EXPECT_TRUE(narrow_groups[group].memory_size_available);
    EXPECT_TRUE(wide_groups[group].memory_size_available);
  }
  ASSERT_AND_ASSIGN(auto clone, narrow->clone_reader());
  ASSERT_AND_ASSIGN(auto cloned_groups, clone->get_row_group_infos());
  ASSERT_EQ(cloned_groups.size(), narrow_groups.size());
  for (size_t group = 0; group < cloned_groups.size(); ++group)
    EXPECT_EQ(cloned_groups[group].read_size, narrow_groups[group].read_size);

  std::vector<api::ChunkInfo> infos;
  for (const auto& group : narrow_groups) {
    infos.push_back({.file_index = 0,
                     .row_offset_in_file = group.start_offset,
                     .number_of_rows = group.end_offset - group.start_offset,
                     .read_size = group.read_size});
  }
  const auto lookup = [&infos](int64_t index) -> const api::ChunkInfo& { return infos[index]; };
  auto tasks = api::ChunkTask::Build({0, 1, 2, 3}, lookup);
  ASSERT_EQ(tasks.size(), 1);
  ASSERT_TRUE(narrow_groups[1].read_size.has_value());
  ASSERT_TRUE(narrow_groups[3].read_size.has_value());
  ASSERT_GT(*narrow_groups[1].read_size, 0);
  ASSERT_GT(*narrow_groups[3].read_size, 0);
  const auto target_bytes = std::max(*narrow_groups[1].read_size, *narrow_groups[3].read_size);
  api::SplitAsyncTasks(tasks, 1, api::ChunkTask::SplitTraits{lookup}, api::AsyncTaskSplitStrategy::kSplitByBytes,
                       target_bytes);
  ASSERT_EQ(tasks.size(), 2);  // Unavailable estimates would fall back to one task.
  std::vector<std::shared_ptr<arrow::Table>> scheduled;
  for (const auto& task : tasks) {
    ASSERT_AND_ASSIGN(auto range, narrow->read_with_range(task.range_start, task.range_end));
    ASSERT_AND_ASSIGN(auto chunk, range->ToTable());
    scheduled.push_back(std::move(chunk));
  }
  ASSERT_AND_ASSIGN(auto actual, arrow::ConcatenateTables(scheduled));
  ASSERT_AND_ASSIGN(auto whole, narrow->read_with_range(0, 3));
  ASSERT_AND_ASSIGN(auto expected, whole->ToTable());
  EXPECT_TRUE(actual->Equals(*expected));
  ASSERT_EQ(actual->num_rows(), 3);
  for (int64_t row = 0; row < 3; ++row) {
    ASSERT_AND_ASSIGN(auto value, actual->column(0)->GetScalar(row));
    EXPECT_EQ(value->ToString(), row == 0 ? "2" : std::to_string(row + 5));
  }
}

TEST_F(DeltaReaderTest, NestedDroppedFieldsAndNullableChildren) {
  const auto table = Table();
  auto physical = arrow::schema({arrow::field(
      "profile", arrow::struct_({arrow::field("keep", arrow::int64()), arrow::field("dropped", arrow::int64())}))});
  WriteRows(table + "/part.parquet", physical,
            {Json{{"profile", Json{{"keep", 7}, {"dropped", 8}}}}, Json{{"profile", nullptr}}});
  auto schema = Schema("profile");
  auto child = Schema("keep");
  child["fields"].push_back(
      Json{{"name", "added"}, {"type", "string"}, {"nullable", true}, {"metadata", Json::object()}});
  schema["fields"][0]["type"] = child;
  Commit(table, 0, {Protocol(), Metadata(schema), Add(table, "part.parquet", 2)});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_AND_ASSIGN(auto reader, Open(files[0]));
  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
  auto profile = std::static_pointer_cast<arrow::StructArray>(batch->column(0));
  EXPECT_TRUE(profile->IsNull(1));
  EXPECT_EQ(profile->field(1)->null_count(), 2);
  EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>(profile->field(0))->Value(0), 7);

  // The pinned Kernel treats an existing parent as missing when all requested
  // children are new. Retain this upstream behavior until Kernel fixes it.
  child["fields"].erase(child["fields"].begin());
  schema["fields"][0]["type"] = child;
  Commit(table, 1, {Metadata(schema)});
  SelectVersion(1);
  ASSERT_AND_ASSIGN(auto replaced, Explore(table));
  ASSERT_AND_ASSIGN(auto replacement_reader, Open(replaced[0]));
  ASSERT_AND_ASSIGN(auto replacement_batch, replacement_reader->get_chunk(0));
  auto replacement = std::static_pointer_cast<arrow::StructArray>(replacement_batch->column(0));
  EXPECT_EQ(replacement->null_count(), 2);
  ASSERT_EQ(replacement->num_fields(), 1);
  EXPECT_EQ(replacement->field(0)->null_count(), 2);
}

TEST_F(DeltaReaderTest, ProjectsNestedBatchesThroughOrdinaryParquetReader) {
  const auto table = Table();
  const auto child_type =
      arrow::struct_({arrow::field("keep", arrow::int64()), arrow::field("dropped", arrow::int64())});
  const auto physical = arrow::schema(
      {arrow::field("profile", child_type), arrow::field("events", arrow::list(child_type)),
       arrow::field("lookup", arrow::map(arrow::utf8(), child_type)), arrow::field("unused", arrow::utf8())});
  const auto child = [](int64_t value) { return Json{{"keep", value}, {"dropped", value + 100}}; };
  WriteRows(table + "/part.parquet", physical,
            {Json{{"profile", child(7)},
                  {"events", Json::array({child(1), child(2)})},
                  {"lookup", Json{{"a", child(4)}}},
                  {"unused", "unrequested"}},
             Json{{"profile", nullptr}, {"events", nullptr}, {"lookup", nullptr}},
             Json{{"profile", child(9)}, {"events", Json::array()}, {"lookup", Json::object()}},
             Json{{"profile", child(11)}, {"events", Json::array({child(3)})}, {"lookup", Json{{"b", nullptr}}}}});
  auto logical_child = Schema("keep");
  logical_child["fields"].push_back(Schema("added", "string")["fields"][0]);
  auto schema = Schema("profile");
  schema["fields"][0]["type"] = logical_child;
  auto events = Schema("events")["fields"][0];
  events["type"] = Json{{"type", "array"}, {"elementType", logical_child}, {"containsNull", true}};
  auto lookup = Schema("lookup")["fields"][0];
  lookup["type"] =
      Json{{"type", "map"}, {"keyType", "string"}, {"valueType", logical_child}, {"valueContainsNull", true}};
  schema["fields"].push_back(events);
  schema["fields"].push_back(lookup);
  Commit(table, 0, {Protocol(), Metadata(schema), Add(table, "part.parquet", 4)});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_AND_ASSIGN(auto reader, Open(files[0], {"lookup", "events", "profile"}));
  ASSERT_AND_ASSIGN(auto chunks, reader->get_chunks({0, 1}));
  ASSERT_AND_ASSIGN(auto all, arrow::ConcatenateRecordBatches(chunks));
  ASSERT_EQ(all->schema()->field_names(), (std::vector<std::string>{"lookup", "events", "profile"}));
  const auto profile = std::static_pointer_cast<arrow::StructArray>(all->column(2));
  ASSERT_EQ(profile->num_fields(), 2);
  EXPECT_EQ(profile->type()->field(0)->name(), "keep");
  EXPECT_TRUE(profile->IsNull(1));
  EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>(profile->field(0))->Value(0), 7);
  EXPECT_EQ(profile->field(1)->null_count(), 4);
  const auto list = std::static_pointer_cast<arrow::ListArray>(all->column(1));
  EXPECT_EQ(list->value_length(0), 2);
  EXPECT_TRUE(list->IsNull(1));
  EXPECT_EQ(list->value_length(2), 0);
  EXPECT_EQ(list->value_length(3), 1);
  EXPECT_EQ(std::static_pointer_cast<arrow::StructArray>(list->values())->num_fields(), 2);
  const auto map = std::static_pointer_cast<arrow::MapArray>(all->column(0));
  EXPECT_EQ(map->value_length(0), 1);
  EXPECT_TRUE(map->IsNull(1));
  EXPECT_EQ(map->value_length(2), 0);
  EXPECT_TRUE(map->items()->IsNull(1));

  ASSERT_AND_ASSIGN(auto clone, reader->clone_reader());
  ASSERT_AND_ASSIGN(auto cloned, clone->get_chunk(0));
  EXPECT_TRUE(cloned->Equals(*chunks[0]));
  ASSERT_AND_ASSIGN(auto range, reader->read_with_range(1, 3));
  ASSERT_AND_ASSIGN(auto range_batches, range->ToRecordBatches());
  ASSERT_AND_ASSIGN(auto ranged, arrow::ConcatenateRecordBatches(range_batches));
  EXPECT_TRUE(ranged->Equals(*all->Slice(1, 2)));
  ASSERT_AND_ASSIGN(auto taken, reader->take({0, 3}));
  ASSERT_AND_ASSIGN(auto expected_take, arrow::Table::FromRecordBatches({all->Slice(0, 1), all->Slice(3, 1)}));
  EXPECT_TRUE(taken->Equals(*expected_take, false));

  // The ordinary reader reads whole selected top-level fields. Charge every
  // nested leaf, including dropped children, but exclude the unused root field.
  ASSERT_AND_ASSIGN(auto input, arrow::io::ReadableFile::Open(table + "/part.parquet"));
  const auto footer = ::parquet::ParquetFileReader::Open(input)->metadata();
  ASSERT_AND_ASSIGN(auto groups, reader->get_row_group_infos());
  ASSERT_EQ(groups.size(), 2);
  for (size_t group = 0; group < groups.size(); ++group) {
    const auto physical_group = footer->RowGroup(group);
    uint64_t expected = 0;
    for (int leaf = 0; leaf < physical_group->num_columns() - 1; ++leaf)
      expected += physical_group->ColumnChunk(leaf)->total_compressed_size();
    EXPECT_EQ(groups[group].read_size, expected);
  }
}

TEST_F(DeltaReaderTest, ValidatesUnprojectedFieldsAndPhysicalCounts) {
  const auto table = Table();
  WriteRows(table + "/part.parquet",
            arrow::schema({arrow::field("id", arrow::int64()), arrow::field("bad", arrow::utf8())}),
            {Json{{"id", 1}, {"bad", "text"}}});
  auto schema = Schema();
  schema["fields"].push_back(Schema("bad")["fields"][0]);
  Commit(table, 0, {Protocol(), Metadata(schema), Add(table, "part.parquet", 1)});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  EXPECT_FALSE(Open(files[0], {"id"}).ok());
  Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "part.parquet", 2)});
  ASSERT_AND_ASSIGN(auto wrong_count, Explore(table));
  EXPECT_FALSE(Open(wrong_count[0]).ok());
}

TEST_F(DeltaReaderTest, SharesSnapshotAndReprojectsWithoutLogIoThenReleases) {
  const auto table = Table();
  WriteData(table + "/part.parquet");
  auto schema = Schema();
  schema["fields"].push_back(Schema("added", "string")["fields"][0]);
  Commit(table, 0, {Protocol(), Metadata(schema), Add(table, "part.parquet", 4)});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  using Trait = deltalake::DeltaLakeFormatReader::MetaTrait;
  ASSERT_AND_ASSIGN(auto metadata, Trait::load_metadata(files[0], properties_, nullptr));
  std::weak_ptr<void> context = metadata->payload.table;
  std::filesystem::remove_all(table + "/_delta_log");
  ASSERT_AND_ASSIGN(auto reader, Open(files[0], {"added"}));
  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
  EXPECT_EQ(batch->column(0)->null_count(), 4);
  ASSERT_AND_ASSIGN(auto reused, Trait::load_metadata(files[0], properties_, nullptr));
  EXPECT_EQ(metadata->payload.table, reused->payload.table);
  reader.reset();
  reused.reset();
  metadata.reset();
  EXPECT_TRUE(context.expired());
  EXPECT_FALSE(Open(files[0]).ok());
  std::filesystem::create_directories(table + "/_delta_log");
  Commit(table, 0, {Protocol(), Metadata(schema), Add(table, "part.parquet", 4)});
  EXPECT_TRUE(Open(files[0]).ok());  // Failed cold opens must not poison the registry.
}

TEST_F(DeltaReaderTest, FileMetadataFollowsItsOwnerWhileSnapshotIsShared) {
  const auto table = Table();
  WriteData(table + "/a.parquet");
  WriteData(table + "/b.parquet");
  const auto dv = Deletions(table, {1});
  auto first_add = Add(table, "a.parquet", 4);
  auto second_add = Add(table, "b.parquet", 4);
  first_add["add"]["deletionVector"] = dv;
  second_add["add"]["deletionVector"] = dv;
  Commit(table, 0, {DvProtocol(), Metadata(Schema()), first_add, second_add});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_EQ(files.size(), 2);

  using Trait = deltalake::DeltaLakeFormatReader::MetaTrait;
  MetadataCache first_owner(true), second_owner(true);
  int loads = 0;
  auto load = [&](const MetadataCache& owner, const api::ColumnGroupFile& file) {
    return owner.get<deltalake::DeltaLakeFormatReader>()->get_or_open(Trait::cache_key(file), [&] {
      ++loads;
      return Trait::load_metadata(file, properties_, nullptr);
    });
  };
  ASSERT_AND_ASSIGN(auto first, load(first_owner, files[0]));
  std::weak_ptr<const deltalake::DeltaFileMetadata> first_file = first->payload.file;
  std::weak_ptr<void> snapshot = first->payload.table;
  ASSERT_AND_ASSIGN(auto reader, Trait::create_from_metadata(first, files[0], nullptr, {}, ""));
  ASSERT_AND_ASSIGN(auto other, load(first_owner, files[1]));
  std::weak_ptr<const deltalake::DeltaFileMetadata> other_file = other->payload.file;
  EXPECT_EQ(first->payload.table, other->payload.table);
  EXPECT_NE(first->payload.file, other->payload.file);
  EXPECT_EQ(loads, 2);
  ASSERT_AND_ASSIGN(auto reused, load(first_owner, files[0]));
  EXPECT_EQ(first, reused);
  EXPECT_EQ(loads, 2);
  reused.reset();
  EXPECT_TRUE(Trait::create_from_metadata(first, files[1], nullptr, {}, "").status().IsInvalid());
  auto wrong_size = files[0];
  wrong_size.Set(api::kPropertyFileSize, files[0].Get<uint64_t>(api::kPropertyFileSize) + 1);
  EXPECT_TRUE(Trait::create_from_metadata(first, wrong_size, nullptr, {}, "").status().IsInvalid());
  auto wrong_rows = files[0];
  ++wrong_rows.end_index;
  EXPECT_TRUE(Trait::create_from_metadata(first, wrong_rows, nullptr, {}, "").status().IsInvalid());
  ASSERT_AND_ASSIGN(auto clone, reader->clone_reader());
  ASSERT_AND_ASSIGN(auto range, reader->read_with_range(0, 3));

  ASSERT_AND_ASSIGN(auto second, load(second_owner, files[0]));
  EXPECT_EQ(loads, 3);
  EXPECT_EQ(first->payload.table, second->payload.table);
  EXPECT_NE(first->payload.file, second->payload.file);
  // Per-file estimates do not grow with other files or charge the shared snapshot.
  EXPECT_EQ(first->cache_size, second->cache_size);

  first_owner = MetadataCache(false);
  first.reset();
  reader.reset();
  other.reset();
  EXPECT_TRUE(other_file.expired());
  EXPECT_FALSE(first_file.expired());
  CheckDvReads(clone, {0, 2, 3});
  clone.reset();
  EXPECT_FALSE(first_file.expired());  // The range still retains its reader.
  ASSERT_AND_ASSIGN(auto ranged, range->ToTable());
  CheckDvIds(ranged, {0, 2, 3});
  range.reset();
  EXPECT_TRUE(first_file.expired());
  EXPECT_FALSE(snapshot.expired());  // The independent owner retains the snapshot.

  ASSERT_AND_ASSIGN(auto survivor, Trait::create_from_metadata(second, files[0], nullptr, {}, ""));
  CheckDvReads(survivor, {0, 2, 3});
  survivor.reset();
  second.reset();
  second_owner = MetadataCache(false);
  EXPECT_TRUE(snapshot.expired());
}

TEST_F(DeltaReaderTest, ColdContextDoesNotCountUnrelatedFiles) {
  const auto table = Table();
  WriteData(table + "/a.parquet");
  WriteData(table + "/b.parquet");
  Commit(table, 0,
         {Protocol(), Metadata(Schema()), Add(table, "a.parquet", 4, false), Add(table, "b.parquet", 4, false)});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_EQ(files.size(), 2);
  std::filesystem::remove(table + "/b.parquet");
  ASSERT_AND_ASSIGN(auto reader, Open(files[0]));
  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
  EXPECT_EQ(batch->num_rows(), 4);
}

TEST_F(DeltaReaderTest, ReadsRelativeLocalRoot) {
  const auto table = Table("relative/table");
  WriteData(table + "/part.parquet", 2);
  Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "part.parquet", 2)});
  auto properties = std::make_shared<api::Properties>(*properties_);
  (*properties)[PROPERTY_FS_ROOT_PATH] = std::string(".");
  properties_ = std::move(properties);
  const auto previous = std::filesystem::current_path();
  auto restore = folly::makeGuard([&] {
    std::filesystem::current_path(previous);
    FilesystemCache::getInstance().clean();
  });

  FilesystemCache::getInstance().clean();
  std::filesystem::current_path(std::filesystem::path(table).parent_path());
  ASSERT_AND_ASSIGN(auto files, Explore("table"));
  ASSERT_EQ(files.size(), 1);
  EXPECT_EQ(files[0].end_index, 2);
  EXPECT_EQ(files[0].path, "file://" + table + "/part.parquet");
  ASSERT_AND_ASSIGN(auto reader, Open(files[0]));
  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
  EXPECT_EQ(batch->num_rows(), 2);
}

TEST_F(DeltaReaderTest, ReadsThroughStableLocalAliases) {
  const auto table = Table("physical/table");
  const auto root = std::filesystem::path(root_);
  const auto alias = (root / "alias").generic_string();
  std::filesystem::create_directory_symlink(root / "physical", alias);
  std::filesystem::create_directory_symlink(table, root / "physical/table-alias");
  (*properties_)[PROPERTY_FS_ROOT_PATH] = alias;
  WriteData(table + "/relative.parquet");
  WriteData(table + "/absolute.parquet");
  const auto logical_table = alias + "/table-alias";
  auto absolute = Add(table, "absolute.parquet", 4);
  absolute["add"]["path"] = "file://" + logical_table + "/absolute.parquet";
  Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "relative.parquet", 4), absolute});
  ASSERT_AND_ASSIGN(auto files, Explore("table-alias"));
  ASSERT_EQ(files.size(), 2);
  for (size_t i = 0; i < files.size(); ++i) {
    const auto filename = i == 0 ? "absolute.parquet" : "relative.parquet";
    EXPECT_EQ(files[i].path, "file://" + logical_table + "/" + filename);
    EXPECT_EQ(Json::parse(files[i].Get<std::string>(api::kPropertyMetadata))["table_uri"],
              "file://" + logical_table + "/");
    ASSERT_AND_ASSIGN(auto reader, Open(files[i]));
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
    EXPECT_EQ(batch->num_rows(), 4);
  }
}

TEST_F(DeltaReaderTest, MaterializedPartitionsKeepMemoryEstimatesUnavailable) {
  const auto table = Table();
  const auto physical = arrow::schema({arrow::field("id", arrow::int64()), arrow::field("region", arrow::utf8())});
  WriteRows(table + "/part.parquet", physical,
            {Json{{"id", 1}, {"region", "west"}}, Json{{"id", 2}, {"region", "west"}}});
  auto schema = Schema();
  schema["fields"].push_back(Schema("region", "string")["fields"][0]);
  auto add = Add(table, "part.parquet", 2);
  add["add"]["partitionValues"] = Json{{"region", "west"}};
  Commit(table, 0, {Protocol(), Metadata(schema, Json::array({"region"})), add});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_AND_ASSIGN(auto reader, Open(files[0]));
  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
  ASSERT_EQ(batch->num_rows(), 2);
  EXPECT_EQ(std::static_pointer_cast<arrow::StringArray>(batch->column(1))->GetString(0), "west");
  ASSERT_AND_ASSIGN(auto groups, reader->get_row_group_infos());
  ASSERT_EQ(groups.size(), 1);
  EXPECT_FALSE(groups[0].memory_size_available);
  ASSERT_TRUE(groups[0].read_size.has_value());
  EXPECT_GT(*groups[0].read_size, 0);
  EXPECT_TRUE(reader->get_rg_column_memsz(0).status().IsNotImplemented());
}

TEST_F(DeltaReaderTest, DescriptorValidationAndSnapshotIsolation) {
  const auto table = Table();
  WriteData(table + "/part.parquet");
  Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "part.parquet", 4)});
  ASSERT_AND_ASSIGN(auto old, Explore(table));
  auto schema = Schema();
  schema["fields"].push_back(Schema("added", "string")["fields"][0]);
  Commit(table, 1, {Metadata(schema)});
  ASSERT_AND_ASSIGN(auto latest, Explore(table));
  ASSERT_AND_ASSIGN(auto reader0, Open(old[0]));
  ASSERT_AND_ASSIGN(auto reader1, Open(latest[0]));
  EXPECT_EQ(reader0->get_schema()->num_fields(), 1);
  EXPECT_EQ(reader1->get_schema()->num_fields(), 2);
  for (auto metadata : {Json{{"version", 2}, {"table_uri", "file:///"}, {"snapshot_version", 0}},
                        Json{{"version", 1}, {"table_uri", "file:///"}, {"snapshot_version", -1}}, Json::object()}) {
    auto bad = old[0];
    bad.Set(api::kPropertyMetadata, metadata.dump());
    EXPECT_FALSE(Open(bad).ok());
  }
  auto bad = old[0];
  ++bad.end_index;
  EXPECT_FALSE(Open(bad).ok());
  bad = old[0];
  bad.Set(api::kPropertyFileSize, uint64_t{1});
  EXPECT_FALSE(Open(bad).ok());
  bad = old[0];
  bad.path += ".not-active";
  EXPECT_FALSE(Open(bad).ok());
}

TEST_F(DeltaReaderTest, RejectsOutOfBoundsDeletionPositions) {
  const auto table = Table();
  WriteData(table + "/part.parquet");
  auto add = Add(table, "part.parquet", 4);
  add["add"]["deletionVector"] = Deletions(table, {9});
  Commit(table, 0, {DvProtocol(), Metadata(Schema()), add});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  EXPECT_FALSE(Open(files[0]).ok());
}

TEST_F(DeltaReaderTest, SupportedScalarLayoutsCrossArrowVersions) {
  struct ScalarCase {
    std::string delta;
    std::shared_ptr<arrow::DataType> arrow;
    Json value;
  };
  const std::vector<ScalarCase> cases{{"boolean", arrow::boolean(), true},
                                      {"byte", arrow::int8(), 3},
                                      {"short", arrow::int16(), 3},
                                      {"integer", arrow::int32(), 3},
                                      {"long", arrow::int64(), 3},
                                      {"float", arrow::float32(), 1.5},
                                      {"double", arrow::float64(), 1.5},
                                      {"string", arrow::utf8(), "value"},
                                      {"binary", arrow::binary(), "bytes"},
                                      {"decimal(10,2)", arrow::decimal128(10, 2), "1230"},
                                      {"date", arrow::date32(), 1},
                                      {"timestamp", arrow::timestamp(arrow::TimeUnit::MICRO, "UTC"), 123456},
                                      {"timestamp_ntz", arrow::timestamp(arrow::TimeUnit::MICRO), 123456}};
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& item = cases[i];
    SCOPED_TRACE(item.delta);
    auto table = Table("scalar-" + std::to_string(i));
    WriteRows(table + "/part.parquet", arrow::schema({arrow::field("id", item.arrow)}),
              {Json{{"id", item.value}}, Json{{"id", nullptr}}});
    Commit(table, 0, {WideningProtocol(), Metadata(Schema("id", item.delta)), Add(table, "part.parquet", 2)});
    ASSERT_AND_ASSIGN(auto files, Explore(table));
    ASSERT_AND_ASSIGN(auto reader, Open(files[0]));
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
    EXPECT_TRUE(batch->column(0)->type()->Equals(item.arrow));
    EXPECT_FALSE(batch->column(0)->IsNull(0));
    EXPECT_TRUE(batch->column(0)->IsNull(1));
  }
}

TEST_F(DeltaReaderTest, AppliesSupportedWideningFamilies) {
  struct Change {
    std::string from, to;
    std::shared_ptr<arrow::DataType> physical, logical;
    Json value;
  };
  const std::vector<Change> changes{
      {"byte", "short", arrow::int8(), arrow::int16(), 3},
      {"short", "integer", arrow::int16(), arrow::int32(), 3},
      {"integer", "long", arrow::int32(), arrow::int64(), 3},
      {"integer", "double", arrow::int32(), arrow::float64(), 3},
      {"float", "double", arrow::float32(), arrow::float64(), 0.1},
      {"decimal(10,2)", "decimal(12,4)", arrow::decimal128(10, 2), arrow::decimal128(12, 4), "1230"},
      {"integer", "decimal(12,2)", arrow::int32(), arrow::decimal128(12, 2), 3},
      {"date", "timestamp_ntz", arrow::date32(), arrow::timestamp(arrow::TimeUnit::MICRO), 1}};
  for (size_t i = 0; i < changes.size(); ++i) {
    const auto& change = changes[i];
    SCOPED_TRACE(change.from + " -> " + change.to);
    const auto table = Table("cast-" + std::to_string(i));
    WriteRows(table + "/part.parquet", arrow::schema({arrow::field("id", change.physical)}),
              {Json{{"id", change.value}}});
    Commit(table, 0,
           {WideningProtocol(), Metadata(Schema("id", change.to, TypeChange(change.from, change.to))),
            Add(table, "part.parquet", 1)});
    ASSERT_AND_ASSIGN(auto files, Explore(table));
    ASSERT_AND_ASSIGN(auto reader, Open(files[0]));
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
    EXPECT_TRUE(batch->column(0)->type()->Equals(change.logical));
    if (change.from == "float" && change.to == "double") {
      EXPECT_EQ(std::static_pointer_cast<arrow::DoubleArray>(batch->column(0))->Value(0), static_cast<double>(0.1f));
    }
    if (change.to == "timestamp_ntz")
      EXPECT_EQ(std::static_pointer_cast<arrow::TimestampArray>(batch->column(0))->Value(0), 86400000000LL);
    if (change.to == "decimal(12,4)") {
      ASSERT_AND_ASSIGN(auto scalar, batch->column(0)->GetScalar(0));
      EXPECT_EQ(scalar->ToString(), "12.3000");
    }
  }
}

TEST_F(DeltaReaderTest, AdaptsArrayAndMapValuesAndVoidColumns) {
  const auto table = Table();
  auto physical = arrow::schema({arrow::field("items", arrow::list(arrow::int32())),
                                 arrow::field("lookup", arrow::map(arrow::utf8(), arrow::int32()))});
  WriteRows(table + "/part.parquet", physical,
            {Json{{"items", Json::array({1, 2})}, {"lookup", Json{{"a", 3}}}},
             Json{{"items", nullptr}, {"lookup", Json::object()}}});
  auto schema = Schema("items");
  schema["fields"][0]["type"] = Json{{"type", "array"}, {"elementType", "long"}, {"containsNull", true}};
  schema["fields"][0]["metadata"] =
      Json{{"delta.typeChanges",
            Json::array({Json{{"fromType", "integer"}, {"toType", "long"}, {"fieldPath", "element"}}})}};
  auto map = Schema("lookup")["fields"][0];
  map["type"] = Json{{"type", "map"}, {"keyType", "string"}, {"valueType", "long"}, {"valueContainsNull", true}};
  map["metadata"] = Json{
      {"delta.typeChanges", Json::array({Json{{"fromType", "integer"}, {"toType", "long"}, {"fieldPath", "value"}}})}};
  schema["fields"].push_back(map);
  schema["fields"].push_back(Schema("empty", "void")["fields"][0]);
  Commit(table, 0, {WideningProtocol(), Metadata(schema), Add(table, "part.parquet", 2)});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  ASSERT_AND_ASSIGN(auto reader, Open(files[0]));
  ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
  auto items = std::static_pointer_cast<arrow::ListArray>(batch->column(0));
  EXPECT_EQ(items->values()->type_id(), arrow::Type::INT64);
  EXPECT_TRUE(items->IsNull(1));
  auto lookup = std::static_pointer_cast<arrow::MapArray>(batch->column(1));
  EXPECT_EQ(lookup->items()->type_id(), arrow::Type::INT64);
  EXPECT_EQ(batch->column(2)->type_id(), arrow::Type::NA);
  EXPECT_EQ(batch->column(2)->null_count(), 2);
}

TEST_F(DeltaReaderTest, CoalescesContextsAndIsolatesFilesystems) {
  const auto table = Table();
  WriteData(table + "/part.parquet");
  Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "part.parquet", 4)});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  using Trait = deltalake::DeltaLakeFormatReader::MetaTrait;
  std::vector<std::future<arrow::Result<Trait::MetadataPtr>>> pending;
  for (int i = 0; i < 8; ++i)
    pending.push_back(
        std::async(std::launch::async, [&] { return Trait::load_metadata(files[0], properties_, nullptr); }));
  ASSERT_AND_ASSIGN(auto first, pending[0].get());
  for (size_t i = 1; i < pending.size(); ++i) {
    ASSERT_AND_ASSIGN(auto metadata, pending[i].get());
    EXPECT_EQ(first->payload.table, metadata->payload.table);
    EXPECT_NE(first->payload.file, metadata->payload.file);
  }
  auto properties = std::make_shared<api::Properties>(*properties_);
  (*properties)[PROPERTY_FS_ROOT_PATH] = std::filesystem::canonical(root_).parent_path().string();
  ASSERT_AND_ASSIGN(auto isolated, Trait::load_metadata(files[0], properties, nullptr));
  EXPECT_NE(first->payload.table, isolated->payload.table);
}

TEST_F(DeltaReaderTest, RemoteSnapshotMatchesEquivalentEncodedFileNames) {
  const auto table = Table();
  WriteData(table + "/part.parquet");
  auto add = Add(table, "part.parquet", 4);
  add["add"]["path"] = "%70art.parquet";
  Commit(table, 0, {Protocol(), Metadata(Schema()), add});
  auto filesystem = std::make_shared<FileSystemProxy>(root_, std::make_shared<arrow::fs::LocalFileSystem>());
  ASSERT_AND_ASSIGN(auto snapshot, deltalake::BlockingDeltaSnapshot::LoadForReader("s3://bucket/table/", 0, filesystem,
                                                                                   {false, "s3", "bucket", ""}));
  api::ColumnGroupFile file{"table/part.parquet", 0, 4, {}};
  file.Set(api::kPropertyFileSize, std::filesystem::file_size(table + "/part.parquet"));
  ASSERT_AND_ASSIGN(auto metadata, parquet::ParquetFormatReader::MetaTrait::load_metadata(file, properties_, nullptr));
  ASSERT_AND_ASSIGN(auto kernel_file, snapshot->LoadFile("s3://bucket/table/part.parquet", metadata->file_schema, 4, 4,
                                                         file.Get<uint64_t>(api::kPropertyFileSize)));
  EXPECT_EQ(kernel_file->LiveRows(), 4);
}

TEST_F(DeltaReaderTest, RejectsMissingRequiredFieldsAndConflictingFieldIds) {
  const auto table = Table();
  WriteData(table + "/part.parquet", 4, "physical_id", arrow::int64(),
            arrow::key_value_metadata({"PARQUET:field_id"}, {"2"}));
  auto schema =
      Schema("id", "long", Json{{"delta.columnMapping.id", 1}, {"delta.columnMapping.physicalName", "physical_id"}});
  const auto protocol = Json{{"protocol", Json{{"minReaderVersion", 2}, {"minWriterVersion", 5}}}};
  Commit(table, 0,
         {protocol,
          Metadata(schema, Json::array(),
                   Json{{"delta.columnMapping.mode", "id"}, {"delta.columnMapping.maxColumnId", "1"}}),
          Add(table, "part.parquet", 4)});
  ASSERT_AND_ASSIGN(auto mapped, Explore(table));
  EXPECT_FALSE(Open(mapped[0]).ok());
  schema = Schema("missing");
  schema["fields"][0]["nullable"] = false;
  Commit(table, 0, {Protocol(), Metadata(schema), Add(table, "part.parquet", 4)});
  ASSERT_AND_ASSIGN(auto required, Explore(table));
  EXPECT_FALSE(Open(required[0]).ok());
}

TEST_F(DeltaReaderTest, ColumnGroupDispatchSupportsCachedAndUncachedReads) {
  const auto table = Table();
  WriteData(table + "/a.parquet");
  WriteData(table + "/b.parquet");
  Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "a.parquet", 4), Add(table, "b.parquet", 4)});
  ASSERT_AND_ASSIGN(auto files, Explore(table));
  auto group = std::make_shared<api::ColumnGroup>(api::ColumnGroup{{"id"}, LOON_FORMAT_DELTALAKE_TABLE, files});
  auto schema = arrow::schema({arrow::field("id", arrow::int64())});
  for (bool enabled : {false, true}) {
    ASSERT_AND_ASSIGN(auto reader, api::ColumnGroupReader::create(schema, group, {"id"}, properties_, nullptr, "",
                                                                  MetadataCache(enabled)));
    EXPECT_EQ(reader->total_rows(), 8);
    ASSERT_AND_ASSIGN(auto batches, reader->get_chunks({0, 1}, 2));
    ASSERT_EQ(batches.size(), 2);
    EXPECT_EQ(batches[0]->num_rows(), 4);
    EXPECT_EQ(batches[1]->num_rows(), 4);
    ASSERT_AND_ASSIGN(auto sizes, reader->get_chunk_column_estimated_sizes(0));
    EXPECT_EQ(sizes.size(), 1);

    auto properties = std::make_shared<api::Properties>(*properties_);
    (*properties)[PROPERTY_READER_METADATA_CACHE_ENABLE] = enabled;
    auto groups = std::make_shared<api::ColumnGroups>(api::ColumnGroups{group});
    auto public_reader = api::Reader::create(groups, schema, nullptr, properties);
    ASSERT_AND_ASSIGN(auto selected, public_reader->take({0, 3, 4, 7}, 2));
    ASSERT_AND_ASSIGN(auto selected_async, public_reader->take_async({0, 3, 4, 7}, 2).get());
    EXPECT_TRUE(selected->Equals(*selected_async));
    ASSERT_EQ(selected->num_rows(), 4);
    for (int row = 0; row < 4; ++row) {
      ASSERT_AND_ASSIGN(auto value, selected->column(0)->GetScalar(row));
      EXPECT_EQ(value->ToString(), row % 2 == 0 ? "0" : "3");
    }
    ASSERT_AND_ASSIGN(auto chunk_reader, public_reader->get_chunk_reader(0, nullptr));
    ASSERT_AND_ASSIGN(auto async_chunks, chunk_reader->get_chunks_async({0, 1}, 2).get());
    ASSERT_EQ(async_chunks.size(), batches.size());
    for (size_t index = 0; index < batches.size(); ++index) EXPECT_TRUE(async_chunks[index]->Equals(*batches[index]));
  }
}

TEST_F(DeltaReaderTest, ReadsEncryptedFilesWithoutSharingDecryptionCallbacks) {
  const std::string key = "0123456789abcdef0123456789abcdef";
  const std::string column_key = "abcdef0123456789abcdef0123456789";
  struct EncryptionLayout {
    bool plaintext_footer;
    bool separate_column_key;
  };
  for (const auto layout : {EncryptionLayout{false, false}, EncryptionLayout{true, false},
                            EncryptionLayout{false, true}, EncryptionLayout{true, true}}) {
    const auto table = Table(std::string(layout.plaintext_footer ? "plaintext-footer" : "encrypted-footer") +
                             (layout.separate_column_key ? "-column-key" : "-footer-key"));
    ::parquet::FileEncryptionProperties::Builder encryption(key);
    encryption.footer_key_metadata("delta-key");
    if (layout.plaintext_footer)
      encryption.set_plaintext_footer();
    if (layout.separate_column_key) {
      ::parquet::ColumnPathToEncryptionPropertiesMap columns;
      columns["id"] = ::parquet::ColumnEncryptionProperties::Builder("id")
                          .key(column_key)
                          ->key_metadata("delta-column-key")
                          ->build();
      encryption.encrypted_columns(columns);
    }
    WriteRows(table + "/part.parquet", arrow::schema({arrow::field("id", arrow::int64())}),
              {Json{{"id", 0}}, Json{{"id", 1}}, Json{{"id", 2}}, Json{{"id", 3}}}, 2, encryption.build());
    Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "part.parquet", 4)});
    ASSERT_AND_ASSIGN(auto files, Explore(table));
    ASSERT_AND_ASSIGN(auto* format, Format::get(LOON_FORMAT_DELTALAKE_TABLE));
    const KeyRetriever retrieve = [&](const std::string& metadata) {
      if (metadata == "delta-column-key")
        return column_key;
      EXPECT_EQ(metadata, "delta-key");
      return key;
    };
    ASSERT_AND_ASSIGN(auto reader, format->create_reader(nullptr, files[0], properties_, {"id"}, retrieve));
    ASSERT_AND_ASSIGN(auto first, reader->get_chunk(0));
    ASSERT_AND_ASSIGN(auto clone, reader->clone_reader());
    ASSERT_AND_ASSIGN(auto last, clone->get_chunk(1));
    EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>(first->column(0))->Value(1), 1);
    EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>(last->column(0))->Value(1), 3);
    // A live reader's decrypted metadata must not let another callback bypass
    // key validation, even though the table-level snapshot is shared.
    const KeyRetriever wrong_key = [](const std::string&) { return std::string(32, 'x'); };
    EXPECT_FALSE(format->create_reader(nullptr, files[0], properties_, {"id"}, wrong_key).ok());
    auto no_key = format->create_reader(nullptr, files[0], properties_, {"id"}, nullptr);
    EXPECT_TRUE(!no_key.ok() || !no_key.ValueOrDie()->get_chunk(0).ok());
    ASSERT_AND_ASSIGN(auto reopened, format->create_reader(nullptr, files[0], properties_, {"id"}, retrieve));
    EXPECT_TRUE(reopened->get_chunk(0).ok());
    if (!layout.plaintext_footer && !layout.separate_column_key) {
      // Exploration has no key callback for its count fallback. An encrypted
      // footer therefore requires usable snapshot statistics.
      Commit(table, 1, {Add(table, "part.parquet", 4, false)});
      EXPECT_FALSE(Explore(table).ok());
    }
  }
}

}  // namespace
}  // namespace milvus_storage
