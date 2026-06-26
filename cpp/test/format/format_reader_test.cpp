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

#include <random>

#include <arrow/api.h>
#include <arrow/extension_type.h>
#include <arrow/io/api.h>
#include <arrow/testing/gtest_util.h>
#include <parquet/arrow/schema.h>
#include <parquet/type_fwd.h>
#include <parquet/arrow/writer.h>

#include "milvus-storage/writer.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "test_env.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

namespace {

class NestedStructExtensionType : public arrow::ExtensionType {
  public:
  explicit NestedStructExtensionType(std::shared_ptr<arrow::DataType> storage_type)
      : arrow::ExtensionType(std::move(storage_type)) {}

  std::string extension_name() const override { return "milvus_storage.test.nested_struct"; }

  bool ExtensionEquals(const arrow::ExtensionType& other) const override {
    return extension_name() == other.extension_name() && storage_type()->Equals(*other.storage_type());
  }

  std::shared_ptr<arrow::Array> MakeArray(std::shared_ptr<arrow::ArrayData> data) const override {
    return std::make_shared<arrow::ExtensionArray>(std::move(data));
  }

  arrow::Result<std::shared_ptr<arrow::DataType>> Deserialize(std::shared_ptr<arrow::DataType> storage_type,
                                                              const std::string&) const override {
    return std::make_shared<NestedStructExtensionType>(std::move(storage_type));
  }

  std::string Serialize() const override { return ""; }
};

class ExtensionTypeRegistrationGuard {
  public:
  explicit ExtensionTypeRegistrationGuard(std::string extension_name) : extension_name_(std::move(extension_name)) {}

  ~ExtensionTypeRegistrationGuard() {
    auto status = arrow::UnregisterExtensionType(extension_name_);
    if (!status.ok() && !status.IsKeyError()) {
      ADD_FAILURE() << status.ToString();
    }
  }

  private:
  std::string extension_name_;
};

}  // namespace

class FormatReaderTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    // Create temporary directory for test files
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    // Mirror fs.* into extfs.default.* so resolve_config can match Lance's Milvus-format URI
    ArrowFileSystemConfig fs_config;
    if (ArrowFileSystemConfig::create_file_system_config(properties_, fs_config).ok() &&
        fs_config.storage_type == "remote") {
      api::SetValue(properties_, "extfs.default.storage_type", "remote");
      api::SetValue(properties_, "extfs.default.cloud_provider", fs_config.cloud_provider.c_str());
      api::SetValue(properties_, "extfs.default.address", fs_config.address.c_str());
      api::SetValue(properties_, "extfs.default.bucket_name", fs_config.bucket_name.c_str());
      api::SetValue(properties_, "extfs.default.region", fs_config.region.c_str());
      api::SetValue(properties_, "extfs.default.access_key_id", fs_config.access_key_id.c_str());
      api::SetValue(properties_, "extfs.default.access_key_value", fs_config.access_key_value.c_str());
      if (fs_config.use_ssl) {
        api::SetValue(properties_, "extfs.default.use_ssl", "true");
      }
      if (fs_config.use_iam) {
        api::SetValue(properties_, "extfs.default.use_iam", "true");
      }
    }

    base_path_ = GetTestBasePath("format-reader-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

    // Create a simple test schema with field IDs required by packed writer
    ASSERT_AND_ASSIGN(schema_, CreateTestSchema());

    // Create test data
    ASSERT_AND_ASSIGN(test_batch_, CreateTestData(schema_));
  }

  void TearDown() override {
    // Clean up test directory
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
  }

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;
  std::shared_ptr<arrow::RecordBatch> test_batch_;
  milvus_storage::api::Properties properties_;
};

TEST_P(FormatReaderTest, ReadParquetWithoutMeta) {
  std::string format = GetParam();
  if (format != LOON_FORMAT_PARQUET) {
    GTEST_SKIP() << "Test parquet only.";
  }

  // using arrow origin writer to write a parquet file
  auto parquet_writer_props = ::parquet::WriterProperties::Builder()
                                  .compression(::parquet::Compression::ZSTD)
                                  ->enable_dictionary()
                                  ->enable_statistics()
                                  ->build();

  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(base_path_ + "/test.parquet"));

  ASSERT_AND_ASSIGN(auto parquet_writer, ::parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(),
                                                                            sink, parquet_writer_props));

  for (size_t i = 0; i < 10; i++) {
    ASSERT_STATUS_OK(parquet_writer->NewBufferedRowGroup());
    ASSERT_STATUS_OK(parquet_writer->WriteRecordBatch(*test_batch_));
  }

  ASSERT_STATUS_OK(parquet_writer->Close());
  ASSERT_STATUS_OK(sink->Close());

  ASSERT_AND_ASSIGN(auto format_reader,
                    FormatReader::create(schema_, LOON_FORMAT_PARQUET,
                                         api::ColumnGroupFile{.path = base_path_ + "/test.parquet",
                                                              .start_index = 0,
                                                              .end_index = test_batch_->num_rows() * 10},
                                         properties_, std::vector<std::string>{"id"}, nullptr));

  ASSERT_AND_ASSIGN(auto row_group_infos, format_reader->get_row_group_infos());
  ASSERT_EQ(row_group_infos.size(), 10);

  for (size_t i = 0; i < row_group_infos.size(); i++) {
    ASSERT_EQ(row_group_infos[i].start_offset, i * test_batch_->num_rows());
    ASSERT_EQ(row_group_infos[i].end_offset, (i + 1) * test_batch_->num_rows());
    ASSERT_GT(row_group_infos[i].memory_size, 0);
    ASSERT_AND_ASSIGN(auto rb, format_reader->get_chunk(i));
    ASSERT_EQ(rb->num_rows(), test_batch_->num_rows());
  }

  std::vector<int> rg_indices_in_file(row_group_infos.size());
  std::iota(rg_indices_in_file.begin(), rg_indices_in_file.end(), 0);
  ASSERT_AND_ASSIGN(auto rbs, format_reader->get_chunks(rg_indices_in_file));

  size_t total_size = 0;
  for (const auto& rb : rbs) {
    total_size += rb->num_rows();
  }
  ASSERT_EQ(total_size, test_batch_->num_rows() * 10);
}

TEST_P(FormatReaderTest, TestReadWithRange) {
  std::string format = GetParam();
  ASSERT_AND_ASSIGN(auto two_cols_schema, CreateTestSchema(std::array<bool, 4>{true, false, false, true}));
  ASSERT_AND_ASSIGN(auto id_schema, CreateTestSchema(std::array<bool, 4>{true, false, false, false}));

  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, two_cols_schema));
  auto writer = Writer::create(base_path_, two_cols_schema, std::move(policy), properties_);

  ASSERT_AND_ASSIGN(
      auto rb_2560_rows,
      CreateTestData(two_cols_schema /* schema */, 0, false /* randdata */, 2560 /* num_rows */, 1024 /* vector_dim */,
                     0 /* str_length */, std::array<bool, 4>{true, false, false, true} /*needed_columns */));
  ASSERT_OK(writer->write(rb_2560_rows));
  ASSERT_OK(writer->flush());

  auto cgs_result = writer->close();

  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();
  auto cg = std::find_if(cgs->begin(), cgs->end(),
                         [](const std::shared_ptr<ColumnGroup>& cg) { return cg->columns[0] == "id"; });
  ASSERT_NE(cg, cgs->end());
  ASSERT_EQ((*cg)->files.size(), 1);

  ASSERT_AND_ASSIGN(auto format_reader, FormatReader::create(id_schema, format, (*cg)->files[0], properties_,
                                                             std::vector<std::string>{"id"}, nullptr));

  for (int i = 0; i < 10; ++i) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis1(0, 2560 - 1);
    int start = dis1(gen);
    std::uniform_int_distribution<> dis2(start + 1, 2560);
    int end = dis2(gen);

    ASSERT_AND_ASSIGN(auto rb_reader, format_reader->read_with_range(start, end));
    ASSERT_AND_ASSIGN(auto rbs, rb_reader->ToRecordBatches());
    ASSERT_AND_ASSIGN(auto rb, arrow::ConcatenateRecordBatches(rbs));
    // verify rb
    ASSERT_EQ(end - start, rb->num_rows());
    ASSERT_EQ(1, rb->num_columns());

    ASSERT_EQ(arrow::Type::INT64, rb->column(0)->type_id());
    auto i64array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
    for (size_t i = 0; i < rb->num_rows(); i++) {
      ASSERT_EQ(i64array->Value(i), (int64_t)(start + i));
    }
  }
}

TEST_P(FormatReaderTest, ParquetReadWithRangeReaderOutlivesFormatReader) {
  std::string format = GetParam();
  if (format != LOON_FORMAT_PARQUET) {
    GTEST_SKIP() << "Test parquet only.";
  }

  auto parquet_writer_props = ::parquet::WriterProperties::Builder()
                                  .compression(::parquet::Compression::ZSTD)
                                  ->enable_dictionary()
                                  ->enable_statistics()
                                  ->build();

  const auto file_path = base_path_ + "/range_outlives_reader.parquet";
  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(file_path));
  ASSERT_AND_ASSIGN(auto parquet_writer, ::parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(),
                                                                            sink, parquet_writer_props));
  ASSERT_STATUS_OK(parquet_writer->NewBufferedRowGroup());
  ASSERT_STATUS_OK(parquet_writer->WriteRecordBatch(*test_batch_));
  ASSERT_STATUS_OK(parquet_writer->Close());
  ASSERT_STATUS_OK(sink->Close());

  std::shared_ptr<arrow::RecordBatchReader> rb_reader;
  {
    std::shared_ptr<FormatReader> format_reader;
    ASSERT_AND_ASSIGN(format_reader, FormatReader::create(
                                         schema_, LOON_FORMAT_PARQUET,
                                         api::ColumnGroupFile{
                                             .path = file_path, .start_index = 0, .end_index = test_batch_->num_rows()},
                                         properties_, std::vector<std::string>{"id"}, nullptr));
    ASSERT_AND_ASSIGN(rb_reader, format_reader->read_with_range(1, 4));
  }

  ASSERT_AND_ASSIGN(auto rbs, rb_reader->ToRecordBatches());
  ASSERT_AND_ASSIGN(auto rb, arrow::ConcatenateRecordBatches(rbs));
  ASSERT_EQ(3, rb->num_rows());
  ASSERT_EQ(1, rb->num_columns());
  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(rb->column(0));
  ASSERT_NE(nullptr, id_array);
  EXPECT_EQ(1, id_array->Value(0));
  EXPECT_EQ(2, id_array->Value(1));
  EXPECT_EQ(3, id_array->Value(2));
}

TEST_P(FormatReaderTest, ParquetCreateFromMetadataReappliesProjection) {
  std::string format = GetParam();
  if (format != LOON_FORMAT_PARQUET) {
    GTEST_SKIP() << "Test parquet only.";
  }

  ASSERT_AND_ASSIGN(auto two_cols_schema, CreateTestSchema(std::array<bool, 4>{true, false, true, false}));
  ASSERT_AND_ASSIGN(auto two_cols_batch, CreateTestData(two_cols_schema, 0, false, 10, 4, 50,
                                                        std::array<bool, 4>{true, false, true, false}));

  const auto file_path = base_path_ + "/cached_projection.parquet";
  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(file_path));
  ASSERT_AND_ASSIGN(auto parquet_writer,
                    ::parquet::arrow::FileWriter::Open(*two_cols_schema, arrow::default_memory_pool(), sink));
  ASSERT_STATUS_OK(parquet_writer->NewBufferedRowGroup());
  ASSERT_STATUS_OK(parquet_writer->WriteRecordBatch(*two_cols_batch));
  ASSERT_STATUS_OK(parquet_writer->Close());
  ASSERT_STATUS_OK(sink->Close());

  api::ColumnGroupFile file{.path = file_path, .start_index = 0, .end_index = two_cols_batch->num_rows()};
  ASSERT_AND_ASSIGN(auto metadata,
                    FormatReader::load_metadata<parquet::ParquetFormatReader>(file, properties_, nullptr));
  auto id_metadata = metadata;
  auto value_metadata = metadata;
  ASSERT_EQ(id_metadata.get(), value_metadata.get());

  ASSERT_AND_ASSIGN(auto id_reader, FormatReader::create_from_metadata<parquet::ParquetFormatReader>(
                                        id_metadata, file, two_cols_schema, {"id"}, ""));
  ASSERT_AND_ASSIGN(auto id_batch, id_reader->get_chunk(0));
  ASSERT_EQ(id_batch->num_rows(), two_cols_batch->num_rows());
  ASSERT_EQ(id_batch->num_columns(), 1);
  ASSERT_EQ(id_batch->schema()->field(0)->name(), "id");
  auto id_array = std::dynamic_pointer_cast<arrow::Int64Array>(id_batch->column(0));
  ASSERT_NE(id_array, nullptr);
  for (int64_t i = 0; i < id_batch->num_rows(); ++i) {
    ASSERT_EQ(id_array->Value(i), i);
  }

  ASSERT_AND_ASSIGN(auto value_reader, FormatReader::create_from_metadata<parquet::ParquetFormatReader>(
                                           value_metadata, file, two_cols_schema, {"value"}, ""));
  ASSERT_AND_ASSIGN(auto value_batch, value_reader->get_chunk(0));
  ASSERT_EQ(value_batch->num_rows(), two_cols_batch->num_rows());
  ASSERT_EQ(value_batch->num_columns(), 1);
  ASSERT_EQ(value_batch->schema()->field(0)->name(), "value");
  auto value_array = std::dynamic_pointer_cast<arrow::DoubleArray>(value_batch->column(0));
  ASSERT_NE(value_array, nullptr);
  for (int64_t i = 0; i < value_batch->num_rows(); ++i) {
    ASSERT_DOUBLE_EQ(value_array->Value(i), i * 1.5);
  }
}

TEST_P(FormatReaderTest, NestedProjectionPreservesTopLevelColumns) {
  std::string format = GetParam();
  auto field_metadata = [](const std::string& field_id) {
    return arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {field_id});
  };

  auto profile_type =
      arrow::struct_({arrow::field("score", arrow::int32(), false), arrow::field("label", arrow::utf8(), false)});
  auto event_type =
      arrow::struct_({arrow::field("code", arrow::int32(), false), arrow::field("message", arrow::utf8(), false)});
  auto events_type = arrow::list(arrow::field("item", event_type, false));
  auto nested_schema = arrow::schema({
      arrow::field("id", arrow::int64(), false, field_metadata("0")),
      arrow::field("profile", profile_type, false, field_metadata("1")),
      arrow::field("events", events_type, false, field_metadata("2")),
      arrow::field("note", arrow::utf8(), false, field_metadata("3")),
  });

  arrow::Int64Builder id_builder;
  ASSERT_STATUS_OK(id_builder.AppendValues({0, 1, 2, 3}));
  ASSERT_AND_ASSIGN(auto ids, id_builder.Finish());

  arrow::Int32Builder score_builder;
  ASSERT_STATUS_OK(score_builder.AppendValues({10, 20, 30, 40}));
  ASSERT_AND_ASSIGN(auto scores, score_builder.Finish());
  arrow::StringBuilder label_builder;
  ASSERT_STATUS_OK(label_builder.AppendValues({"cold", "warm", "hot", "peak"}));
  ASSERT_AND_ASSIGN(auto labels, label_builder.Finish());
  auto profiles =
      std::make_shared<arrow::StructArray>(profile_type, 4, std::vector<std::shared_ptr<arrow::Array>>{scores, labels});

  arrow::Int32Builder event_code_builder;
  ASSERT_STATUS_OK(event_code_builder.AppendValues({1, 2, 3, 4, 5}));
  ASSERT_AND_ASSIGN(auto event_codes, event_code_builder.Finish());
  arrow::StringBuilder event_message_builder;
  ASSERT_STATUS_OK(event_message_builder.AppendValues({"created", "queued", "running", "done", "archived"}));
  ASSERT_AND_ASSIGN(auto event_messages, event_message_builder.Finish());
  auto event_values = std::make_shared<arrow::StructArray>(
      event_type, 5, std::vector<std::shared_ptr<arrow::Array>>{event_codes, event_messages});
  arrow::Int32Builder event_offsets_builder;
  ASSERT_STATUS_OK(event_offsets_builder.AppendValues({0, 2, 3, 3, 5}));
  ASSERT_AND_ASSIGN(auto event_offsets, event_offsets_builder.Finish());
  ASSERT_AND_ASSIGN(auto events, arrow::ListArray::FromArrays(events_type, *event_offsets, *event_values));

  arrow::StringBuilder note_builder;
  ASSERT_STATUS_OK(note_builder.AppendValues({"n0", "n1", "n2", "n3"}));
  ASSERT_AND_ASSIGN(auto notes, note_builder.Finish());

  auto record_batch = arrow::RecordBatch::Make(nested_schema, 4, {ids, profiles, events, notes});

  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, nested_schema));
  auto writer = Writer::create(base_path_, nested_schema, std::move(policy), properties_);
  ASSERT_OK(writer->write(record_batch));
  ASSERT_AND_ASSIGN(auto column_groups, writer->close());
  ASSERT_EQ(column_groups->size(), 1);
  ASSERT_EQ(column_groups->front()->files.size(), 1);

  auto assert_batch_equal = [](const std::shared_ptr<arrow::RecordBatch>& actual,
                               const std::shared_ptr<arrow::RecordBatch>& expected) {
    ASSERT_TRUE(actual->Equals(*expected)) << "expected:\n"
                                           << expected->ToString() << "\nactual:\n"
                                           << actual->ToString() << "\nexpected schema:\n"
                                           << expected->schema()->ToString(true) << "\nactual schema:\n"
                                           << actual->schema()->ToString(true);
  };

  const std::vector<std::vector<int>> projection_cases = {
      {0, 1, 2, 3}, {1}, {2}, {3, 2, 1}, {1, 3}, {0, 2},
  };
  for (size_t case_index = 0; case_index < projection_cases.size(); ++case_index) {
    SCOPED_TRACE(::testing::Message() << "projection case " << case_index);

    const auto& projected_field_indices = projection_cases[case_index];
    std::vector<std::string> needed_columns;
    needed_columns.reserve(projected_field_indices.size());
    for (const int field_index : projected_field_indices) {
      needed_columns.emplace_back(nested_schema->field(field_index)->name());
    }

    ASSERT_AND_ASSIGN(auto expected_batch, record_batch->SelectColumns(projected_field_indices));
    ASSERT_AND_ASSIGN(auto format_reader,
                      FormatReader::create(expected_batch->schema(), format, column_groups->front()->files.front(),
                                           properties_, needed_columns, nullptr));

    ASSERT_AND_ASSIGN(auto chunk, format_reader->get_chunk(0));
    assert_batch_equal(chunk, expected_batch);

    ASSERT_AND_ASSIGN(auto chunks, format_reader->get_chunks({0}));
    ASSERT_AND_ASSIGN(auto chunks_batch, arrow::ConcatenateRecordBatches(chunks));
    assert_batch_equal(chunks_batch, expected_batch);

    ASSERT_AND_ASSIGN(auto range_reader, format_reader->read_with_range(1, 3));
    ASSERT_TRUE(range_reader->schema()->Equals(*expected_batch->schema(), false))
        << "expected schema:\n"
        << expected_batch->schema()->ToString(true) << "\nactual schema:\n"
        << range_reader->schema()->ToString(true);
    ASSERT_AND_ASSIGN(auto range_batches, range_reader->ToRecordBatches());
    ASSERT_AND_ASSIGN(auto range_batch, arrow::ConcatenateRecordBatches(range_batches));
    assert_batch_equal(range_batch, expected_batch->Slice(1, 2));
  }

  if (format != LOON_FORMAT_PARQUET) {
    return;
  }

  auto extension_storage_type =
      arrow::struct_({arrow::field("real", arrow::float64(), false), arrow::field("imag", arrow::float64(), false)});
  auto extension_type = std::make_shared<NestedStructExtensionType>(extension_storage_type);
  auto unregister_status = arrow::UnregisterExtensionType(extension_type->extension_name());
  ASSERT_TRUE(unregister_status.ok() || unregister_status.IsKeyError()) << unregister_status.ToString();
  ASSERT_STATUS_OK(arrow::RegisterExtensionType(extension_type));
  ExtensionTypeRegistrationGuard extension_guard(extension_type->extension_name());

  auto extension_schema = arrow::schema({
      arrow::field("id", arrow::int64(), false, field_metadata("0")),
      arrow::field("extension_profile", extension_type, false, field_metadata("1")),
      arrow::field("note", arrow::utf8(), false, field_metadata("2")),
  });
  auto extension_storage_schema = arrow::schema({
      extension_schema->field(0),
      arrow::field("extension_profile", extension_storage_type, false, field_metadata("1")),
      extension_schema->field(2),
  });

  arrow::DoubleBuilder extension_real_builder;
  ASSERT_STATUS_OK(extension_real_builder.AppendValues({1.5, 2.5, 3.5, 4.5}));
  ASSERT_AND_ASSIGN(auto extension_reals, extension_real_builder.Finish());
  arrow::DoubleBuilder extension_imag_builder;
  ASSERT_STATUS_OK(extension_imag_builder.AppendValues({10.5, 20.5, 30.5, 40.5}));
  ASSERT_AND_ASSIGN(auto extension_imags, extension_imag_builder.Finish());
  auto extension_storage = std::make_shared<arrow::StructArray>(
      extension_storage_type, 4, std::vector<std::shared_ptr<arrow::Array>>{extension_reals, extension_imags});
  auto extension_profiles = arrow::ExtensionType::WrapArray(extension_type, extension_storage);

  auto extension_batch = arrow::RecordBatch::Make(extension_schema, 4, {ids, extension_profiles, notes});
  auto extension_storage_batch = arrow::RecordBatch::Make(extension_storage_schema, 4, {ids, extension_storage, notes});

  ASSERT_AND_ASSIGN(auto extension_policy, CreateSinglePolicy(format, extension_schema));
  auto extension_writer =
      Writer::create(base_path_ + "/extension", extension_schema, std::move(extension_policy), properties_);
  ASSERT_OK(extension_writer->write(extension_batch));
  ASSERT_AND_ASSIGN(auto extension_column_groups, extension_writer->close());
  ASSERT_EQ(extension_column_groups->size(), 1);
  ASSERT_EQ(extension_column_groups->front()->files.size(), 1);

  const auto& extension_file = extension_column_groups->front()->files.front();
  ASSERT_AND_ASSIGN(auto loaded_metadata,
                    parquet::ParquetFormatReader::MetaTrait::load_metadata(extension_file, properties_, nullptr));
  auto metadata = std::make_shared<parquet::ParquetFormatReader::MetaTrait::Metadata>(*loaded_metadata);

  // The file reader can return the extension column as its storage struct, which is allowed here.
  // This regression is only about projection planning: metadata->file_schema may still contain
  // the logical extension type, so leaf-column expansion must count leaves from storage_type().
  metadata->file_schema = extension_schema;

  const std::vector<int> extension_projected_field_indices = {2, 1};
  const std::vector<std::string> extension_needed_columns = {"note", "extension_profile"};
  ASSERT_AND_ASSIGN(auto extension_expected_batch,
                    extension_storage_batch->SelectColumns(extension_projected_field_indices));
  ASSERT_AND_ASSIGN(auto extension_reader,
                    parquet::ParquetFormatReader::MetaTrait::create_from_metadata(
                        metadata, extension_file, extension_expected_batch->schema(), extension_needed_columns, ""));

  ASSERT_AND_ASSIGN(auto extension_chunk, extension_reader->get_chunk(0));
  assert_batch_equal(extension_chunk, extension_expected_batch);
}

INSTANTIATE_TEST_SUITE_P(FormatReaderTestP,
                         FormatReaderTest,
                         ::testing::Values(LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX, LOON_FORMAT_LANCE_TABLE));

}  // namespace milvus_storage::test
