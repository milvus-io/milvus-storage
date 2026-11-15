// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

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
#include <parquet/metadata.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/fs.h"
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
    std::ofstream output(table + "/deletions.bin", std::ios::binary);
    output.write(reinterpret_cast<const char*>(dv.bytes.data()), dv.bytes.size());
    output.close();
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
};

TEST(DeltaPathTest, NormalizesLocalIdentityAndChecksContainmentBeforeOpening) {
  deltalake::ResolvedLocation table;
  table.config.root_path = "/physical";
  table.binding.is_local = true;
  table.binding.local_root = "/bound";
  const auto resolve = [&](const std::string& path) { return deltalake::ResolveDataFile(path, table, nullptr); };
  for (const auto& url : {"file:///bound/%70art%20a%252F.parquet", "file:///bound/dir/../part%20a%252F.parquet"}) {
    ASSERT_AND_ASSIGN(auto file, resolve(url));
    EXPECT_EQ(file.url, "file:///bound/part%20a%252F.parquet");
    EXPECT_EQ(file.path, "part a%2F.parquet");
  }
  for (const auto& url : {"file:///bound/dir/../../outside.parquet", "file:///bound/%2E%2E/outside.parquet",
                          "file:///bound-other/part.parquet", "file:///physical/part.parquet"}) {
    EXPECT_TRUE(resolve(url).status().IsInvalid()) << url;
  }
}

TEST(DeltaPathTest, PreservesConfiguredAliasesWithPhysicalFilesystemIdentity) {
  ASSERT_AND_ASSIGN(auto temporary, arrow::internal::TemporaryDir::Make("delta-path-"));
  const auto root = std::filesystem::path(temporary->path().ToString());
  const auto physical = (root / "physical").generic_string();
  const auto alias = (root / "alias").generic_string();
  std::filesystem::create_directories(physical + "/table");
  std::filesystem::create_directory_symlink(physical, alias);
  std::filesystem::create_directory_symlink(physical + "/table", physical + "/table-alias");
  auto properties = std::make_shared<api::Properties>(
      api::Properties{{PROPERTY_FS_STORAGE_TYPE, std::string("local")}, {PROPERTY_FS_ROOT_PATH, alias}});

  ASSERT_AND_ASSIGN(auto location, deltalake::ResolveLocation("table-alias", properties));
  EXPECT_EQ(location.config.root_path, std::filesystem::canonical(physical).generic_string());
  EXPECT_EQ(location.binding.local_root, alias);
  EXPECT_EQ(location.url, "file://" + alias + "/table-alias/");
  // A usable snapshot count must not require this data file to exist.
  const auto url = location.url + "missing.parquet";
  ASSERT_AND_ASSIGN(auto file, deltalake::ResolveDataFile(url, location, properties));
  EXPECT_EQ(file.url, url);
  EXPECT_EQ(file.path, "table-alias/missing.parquet");
  EXPECT_FALSE(std::filesystem::exists(physical + "/table/missing.parquet"));
  EXPECT_TRUE(deltalake::ResolveDataFile("file://" + physical + "/table/missing.parquet", location, properties)
                  .status()
                  .IsInvalid());

  (*properties)[PROPERTY_FS_ROOT_PATH] = physical;
  ASSERT_AND_ASSIGN(auto direct, deltalake::ResolveLocation("table", properties));
  EXPECT_EQ(direct.filesystem_key, location.filesystem_key);
  EXPECT_EQ(direct.filesystem, location.filesystem);
  EXPECT_NE(direct.url, location.url);
}

TEST(DeltaPathTest, RetargetedRootAliasChangesFilesystemIdentity) {
  ASSERT_AND_ASSIGN(auto temporary, arrow::internal::TemporaryDir::Make("delta-path-"));
  const auto root = std::filesystem::path(temporary->path().ToString());
  std::filesystem::create_directories(root / "first");
  std::filesystem::create_directories(root / "second");
  const auto alias = (root / "alias").generic_string();
  std::filesystem::create_directory_symlink(root / "first", alias);
  auto properties = std::make_shared<api::Properties>(
      api::Properties{{PROPERTY_FS_STORAGE_TYPE, std::string("local")}, {PROPERTY_FS_ROOT_PATH, alias}});
  ASSERT_AND_ASSIGN(auto first, deltalake::ResolveLocation("table", properties));
  std::filesystem::remove(alias);
  std::filesystem::create_directory_symlink(root / "second", alias);
  ASSERT_AND_ASSIGN(auto second, deltalake::ResolveLocation("table", properties));
  EXPECT_EQ(first.url, second.url);
  EXPECT_EQ(first.binding.local_root, second.binding.local_root);
  EXPECT_NE(first.filesystem_key, second.filesystem_key);
  EXPECT_NE(first.filesystem, second.filesystem);
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

TEST_F(DeltaReaderTest, ResolvesRelativeRootsBeforeFilesystemCacheLookup) {
  const auto first_table = Table("first/table");
  const auto second_table = Table("second/table");
  WriteData(first_table + "/part.parquet", 2);
  WriteData(second_table + "/part.parquet", 3);
  Commit(first_table, 0, {Protocol(), Metadata(Schema()), Add(first_table, "part.parquet", 2)});
  Commit(second_table, 0, {Protocol(), Metadata(Schema()), Add(second_table, "part.parquet", 3)});
  auto properties = std::make_shared<api::Properties>(*properties_);
  (*properties)[PROPERTY_FS_ROOT_PATH] = std::string(".");
  properties_ = std::move(properties);
  const auto previous = std::filesystem::current_path();
  auto restore = folly::makeGuard([&] { std::filesystem::current_path(previous); });

  std::filesystem::current_path(std::filesystem::path(first_table).parent_path());
  // Warm the shared cache with the original relative-root properties. A later
  // Delta open must resolve its own absolute identity before using that cache.
  ASSERT_AND_ASSIGN(auto warmed, FilesystemCache::getInstance().get(properties_));
  ASSERT_AND_ASSIGN(auto first, Explore("table"));
  ASSERT_AND_ASSIGN(auto first_reader, Open(first[0]));

  std::filesystem::current_path(std::filesystem::path(second_table).parent_path());
  ASSERT_AND_ASSIGN(auto second, Explore("table"));
  ASSERT_EQ(second.size(), 1);
  EXPECT_EQ(second[0].end_index, 3);
  EXPECT_EQ(second[0].path, "file://" + std::filesystem::canonical(second_table).generic_string() + "/part.parquet");
  ASSERT_AND_ASSIGN(auto second_reader, Open(second[0]));
  ASSERT_AND_ASSIGN(auto second_batch, second_reader->get_chunk(0));
  EXPECT_EQ(second_batch->num_rows(), 3);
  ASSERT_AND_ASSIGN(auto clone, first_reader->clone_reader());
  ASSERT_AND_ASSIGN(auto first_batch, clone->get_chunk(0));
  EXPECT_EQ(first_batch->num_rows(), 2);
}

TEST_F(DeltaReaderTest, ResolvesPhysicalRootBeforeCollapsingSymlinkDotDot) {
  const auto table = Table("actual/dataset/table");
  std::filesystem::create_directories(root_ + "/actual/child");
  std::filesystem::create_directory_symlink(root_ + "/actual/child", root_ + "/link");
  WriteData(table + "/part.parquet", 3);
  Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "part.parquet", 3)});
  auto properties = std::make_shared<api::Properties>(*properties_);
  (*properties)[PROPERTY_FS_ROOT_PATH] = root_ + "/link/../dataset";
  properties_ = std::move(properties);
  const auto alias = root_ + "/link/../dataset/table";
  const auto logical_table = std::filesystem::path(alias).lexically_normal().generic_string();
  for (const auto& location : {std::string("table"), alias, "file://" + alias}) {
    ASSERT_AND_ASSIGN(auto files, Explore(location));
    ASSERT_EQ(files.size(), 1);
    EXPECT_EQ(files[0].path, "file://" + logical_table + "/part.parquet");
    EXPECT_EQ(Json::parse(files[0].Get<std::string>(api::kPropertyMetadata))["table_uri"],
              "file://" + logical_table + "/");
    ASSERT_AND_ASSIGN(auto reader, Open(files[0]));
    ASSERT_AND_ASSIGN(auto batch, reader->get_chunk(0));
    EXPECT_EQ(batch->num_rows(), 3);
  }
}

TEST_F(DeltaReaderTest, PersistsConfiguredRootAndTableAliases) {
  const auto table = Table("physical/table");
  const auto root = std::filesystem::path(root_);
  const auto alias = (root / "alias").generic_string();
  std::filesystem::create_directory_symlink(root / "physical", alias);
  std::filesystem::create_directory_symlink(table, root / "physical/table-alias");
  (*properties_)[PROPERTY_FS_ROOT_PATH] = alias;
  WriteData(table + "/relative.parquet");
  WriteData(table + "/absolute.parquet");
  for (const auto& name : {"table", "table-alias"}) {
    const auto logical_table = alias + "/" + name;
    auto absolute = Add(table, "absolute.parquet", 4);
    absolute["add"]["path"] = "file://" + logical_table + "/absolute.parquet";
    Commit(table, 0, {Protocol(), Metadata(Schema()), Add(table, "relative.parquet", 4), absolute});
    ASSERT_AND_ASSIGN(auto files, Explore(logical_table));
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
}

TEST_F(DeltaReaderTest, ReopensPersistedAliasesWithANewPhysicalRoot) {
  const auto first_table = Table("first/table");
  const auto second_table = Table("second/table");
  WriteData(first_table + "/part.parquet");
  std::filesystem::copy_file(first_table + "/part.parquet", second_table + "/part.parquet");
  Commit(first_table, 0, {Protocol(), Metadata(Schema()), Add(first_table, "part.parquet", 4)});
  auto second_schema = Schema();
  second_schema["fields"].push_back(Schema("added", "string")["fields"][0]);
  Commit(second_table, 0, {Protocol(), Metadata(second_schema), Add(second_table, "part.parquet", 4)});
  const auto root = std::filesystem::path(root_);
  const auto alias = (root / "alias").generic_string();
  std::filesystem::create_directory_symlink(root / "first", alias);
  (*properties_)[PROPERTY_FS_ROOT_PATH] = alias;
  ASSERT_AND_ASSIGN(auto files, Explore("table"));
  ASSERT_EQ(files.size(), 1);
  ASSERT_AND_ASSIGN(auto first_reader, Open(files[0]));
  EXPECT_EQ(first_reader->get_schema()->num_fields(), 1);

  // A different host may expose the same configured alias at another physical
  // path. Its exact-version context must not reuse the original filesystem.
  std::filesystem::remove(alias);
  std::filesystem::create_directory_symlink(root / "second", alias);
  ASSERT_AND_ASSIGN(auto second_reader, Open(files[0]));
  EXPECT_EQ(second_reader->get_schema()->num_fields(), 2);
  ASSERT_AND_ASSIGN(auto batch, second_reader->get_chunk(0));
  EXPECT_EQ(batch->column(1)->null_count(), 4);
  ASSERT_AND_ASSIGN(auto first_clone, first_reader->clone_reader());
  EXPECT_EQ(first_clone->get_schema()->num_fields(), 1);
  ASSERT_AND_ASSIGN(auto first_batch, first_clone->get_chunk(0));
  EXPECT_EQ(first_batch->num_rows(), 4);
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
      {"float", "double", arrow::float32(), arrow::float64(), 3},
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
    EXPECT_EQ(first->payload.files, metadata->payload.files);
  }
  auto equivalent_properties = std::make_shared<api::Properties>(*properties_);
  (*equivalent_properties)[PROPERTY_FS_ROOT_PATH] = (std::filesystem::path(root_) / ".").string();
  ASSERT_AND_ASSIGN(auto equivalent, Trait::load_metadata(files[0], equivalent_properties, nullptr));
  EXPECT_EQ(first->payload.table, equivalent->payload.table);
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
  ASSERT_AND_ASSIGN(auto metadata, parquet::ParquetFormatReader::MetaTrait::load_metadata_with_filesystem(
                                       file, filesystem, file.path, properties_, nullptr));
  ASSERT_AND_ASSIGN(auto kernel_file, snapshot->LoadFile("s3://bucket/table/part.parquet",
                                                         metadata->payload.parquet_metadata->SerializeToString(), 4, 4,
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
