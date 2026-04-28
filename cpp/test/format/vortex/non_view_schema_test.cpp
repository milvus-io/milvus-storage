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

// Coverage for `PROPERTY_READER_VORTEX_SCHEMA_NON_VIEW`.
//
// Vortex's default `to_arrow_dtype` maps `DType::Utf8` -> `DataType::Utf8View`
// and `DType::Binary` -> `DataType::BinaryView` (its preferred Arrow output).
// milvus consumes plain non-view buffers downstream, so when the caller does
// not supply an explicit read schema, the bridge's override emits plain Utf8
// / Binary instead.
//
// Two angles for top-level columns:
//
// (A) Default behavior — property unset / false. Schemaless reads of Utf8 /
//     Binary columns come back as Utf8View / BinaryView (vortex's default).
//
// (B) Fixed behavior — property set to true. Schemaless reads of Utf8 /
//     Binary columns come back as plain Utf8 / Binary, matching what milvus
//     writes. GetFileSchema agrees with what reads produce.
//
// Plus nested-type cases (List, FixedSizeList, Struct) that exercise the
// recursion in the override's `to_arrow_dtype`.

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include <boost/filesystem/operations.hpp>

#include "test_env.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/format/vortex/vortex_writer.h"

namespace milvus_storage {

using namespace vortex;

class VortexNonViewSchemaTest : public ::testing::Test {
  protected:
  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    file_system_ = std::make_shared<arrow::fs::LocalFileSystem>();
  }

  void TearDown() override {
    auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
    if (storage_type == "local" || storage_type.empty()) {
      boost::filesystem::remove_all(utf8_file_);
      boost::filesystem::remove_all(binary_file_);
      boost::filesystem::remove_all(fsl_f32_file_);
      boost::filesystem::remove_all(list_utf8_file_);
      boost::filesystem::remove_all(struct_utf8_file_);
    }
  }

  void WriteVortex(const std::string& path, const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto writer = VortexFileWriter(file_system_, batch->schema(), path, properties_);
    ASSERT_STATUS_OK(writer.Write(batch));
    ASSERT_STATUS_OK(writer.Flush());
    ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
    ASSERT_EQ(batch->num_rows(), cgfile.end_index);
  }

  std::shared_ptr<arrow::RecordBatch> MakeBatch(int64_t count, const std::shared_ptr<arrow::DataType>& str_type) {
    arrow::Int32Builder id_b;
    for (int64_t i = 0; i < count; ++i) {
      EXPECT_TRUE(id_b.Append(static_cast<int32_t>(i)).ok());
    }
    std::shared_ptr<arrow::Array> id_arr;
    EXPECT_TRUE(id_b.Finish(&id_arr).ok());

    std::shared_ptr<arrow::Array> str_arr;
    if (str_type->id() == arrow::Type::STRING) {
      arrow::StringBuilder b;
      for (int64_t i = 0; i < count; ++i) {
        EXPECT_TRUE(b.Append("v" + std::to_string(i)).ok());
      }
      EXPECT_TRUE(b.Finish(&str_arr).ok());
    } else {
      arrow::BinaryBuilder b;
      for (int64_t i = 0; i < count; ++i) {
        EXPECT_TRUE(b.Append("v" + std::to_string(i)).ok());
      }
      EXPECT_TRUE(b.Finish(&str_arr).ok());
    }

    auto schema = arrow::schema({
        arrow::field("id", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
        arrow::field("str_col", str_type, false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
    });
    return arrow::RecordBatch::Make(schema, count, {id_arr, str_arr});
  }

  void WriteFile(const std::string& path, const std::shared_ptr<arrow::DataType>& str_type) {
    auto rb = MakeBatch(kRows, str_type);
    auto writer = VortexFileWriter(file_system_, rb->schema(), path, properties_);
    ASSERT_STATUS_OK(writer.Write(rb));
    ASSERT_STATUS_OK(writer.Flush());
    ASSERT_AND_ASSIGN(auto cgfile, writer.Close());
    ASSERT_EQ(kRows, cgfile.end_index);
  }

  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> file_system_;
  const std::string utf8_file_ = "test-non-view-utf8.vx";
  const std::string binary_file_ = "test-non-view-binary.vx";
  const std::string fsl_f32_file_ = "test-non-view-fsl-f32.vx";
  const std::string list_utf8_file_ = "test-non-view-list-utf8.vx";
  const std::string struct_utf8_file_ = "test-non-view-struct-utf8.vx";
  static constexpr int64_t kRows = 64;
};

// (A) Default: property unset → schemaless Utf8 read comes back as
// Utf8View (vortex's preferred mapping for DType::Utf8).
TEST_F(VortexNonViewSchemaTest, SchemalessUtf8DefaultsToUtf8View) {
  WriteFile(utf8_file_, arrow::utf8());

  std::shared_ptr<arrow::Schema> no_schema;  // schemaless
  auto reader =
      VortexFormatReader(file_system_, no_schema, utf8_file_, properties_, std::vector<std::string>{"id", "str_col"});
  ASSERT_STATUS_OK(reader.open());

  auto str_type = reader.get_schema()->GetFieldByName("str_col")->type();
  EXPECT_EQ(str_type->id(), arrow::Type::STRING_VIEW)
      << "Without the non-view property, vortex's schemaless path returns "
         "its preferred Utf8View.";

  ASSERT_AND_ASSIGN(auto table, reader.take({0, 1, 2, 3}));
  ASSERT_AND_ASSIGN(auto rb, table->CombineChunksToBatch());
  EXPECT_EQ(rb->column(1)->type_id(), arrow::Type::STRING_VIEW);
}

// (A') Same for Binary.
TEST_F(VortexNonViewSchemaTest, SchemalessBinaryDefaultsToBinaryView) {
  WriteFile(binary_file_, arrow::binary());

  std::shared_ptr<arrow::Schema> no_schema;
  auto reader =
      VortexFormatReader(file_system_, no_schema, binary_file_, properties_, std::vector<std::string>{"id", "str_col"});
  ASSERT_STATUS_OK(reader.open());
  EXPECT_EQ(reader.get_schema()->GetFieldByName("str_col")->type()->id(), arrow::Type::BINARY_VIEW);

  ASSERT_AND_ASSIGN(auto table, reader.take({0, 1, 2, 3}));
  ASSERT_AND_ASSIGN(auto rb, table->CombineChunksToBatch());
  EXPECT_EQ(rb->column(1)->type_id(), arrow::Type::BINARY_VIEW);
}

// (B) Fix: PROPERTY_READER_VORTEX_SCHEMA_NON_VIEW=true on a schemaless
// reader. GetFileSchema and schemaless take()/blocking_read() all return
// plain Utf8 (3-buffer StringArray). Round-trip data values verified.
TEST_F(VortexNonViewSchemaTest, NonViewPropertyForcesUtf8) {
  WriteFile(utf8_file_, arrow::utf8());

  api::Properties props = properties_;
  props[PROPERTY_READER_VORTEX_SCHEMA_NON_VIEW] = true;

  std::shared_ptr<arrow::Schema> no_schema;
  auto reader =
      VortexFormatReader(file_system_, no_schema, utf8_file_, props, std::vector<std::string>{"id", "str_col"});
  ASSERT_STATUS_OK(reader.open());

  // GetFileSchema also reflects the non-view choice.
  auto str_type = reader.get_schema()->GetFieldByName("str_col")->type();
  EXPECT_EQ(str_type->id(), arrow::Type::STRING) << "non_view=true should make GetFileSchema report STRING, not "
                                                    "STRING_VIEW.";

  // take() — backed by ImportChunkedArray on the C-stream.
  ASSERT_AND_ASSIGN(auto table, reader.take({0, 1, 2, 3}));
  ASSERT_AND_ASSIGN(auto rb, table->CombineChunksToBatch());
  ASSERT_EQ(rb->column(1)->type_id(), arrow::Type::STRING);
  auto strs = std::dynamic_pointer_cast<arrow::StringArray>(rb->column(1));
  ASSERT_NE(strs, nullptr);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(strs->GetString(i), "v" + std::to_string(i));
  }

  // blocking_read() goes through the same code path; spot-check it too.
  ASSERT_AND_ASSIGN(auto chunked, reader.blocking_read(0, kRows));
  ASSERT_GT(chunked->num_chunks(), 0);
  ASSERT_AND_ASSIGN(auto rb2, arrow::RecordBatch::FromStructArray(chunked->chunk(0)));
  EXPECT_EQ(rb2->column(1)->type_id(), arrow::Type::STRING);
}

// (B') Same for Binary.
TEST_F(VortexNonViewSchemaTest, NonViewPropertyForcesBinary) {
  WriteFile(binary_file_, arrow::binary());

  api::Properties props = properties_;
  props[PROPERTY_READER_VORTEX_SCHEMA_NON_VIEW] = true;

  std::shared_ptr<arrow::Schema> no_schema;
  auto reader =
      VortexFormatReader(file_system_, no_schema, binary_file_, props, std::vector<std::string>{"id", "str_col"});
  ASSERT_STATUS_OK(reader.open());
  EXPECT_EQ(reader.get_schema()->GetFieldByName("str_col")->type()->id(), arrow::Type::BINARY);

  ASSERT_AND_ASSIGN(auto table, reader.take({0, 1, 2, 3}));
  ASSERT_AND_ASSIGN(auto rb, table->CombineChunksToBatch());
  EXPECT_EQ(rb->column(1)->type_id(), arrow::Type::BINARY);
}

// (B'') With non_view=true but the caller still supplies an explicit
// read_schema, the explicit schema wins (the property only acts on the
// schemaless code path). Pins down the precedence.
TEST_F(VortexNonViewSchemaTest, ExplicitSchemaOverridesNonViewProperty) {
  WriteFile(utf8_file_, arrow::utf8());

  api::Properties props = properties_;
  props[PROPERTY_READER_VORTEX_SCHEMA_NON_VIEW] = true;

  // Explicit read schema = utf8_view. The property is set, but it should
  // NOT override an explicit caller schema — vortex must honor utf8_view.
  auto explicit_schema = arrow::schema({
      arrow::field("id", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
      arrow::field("str_col", arrow::utf8_view(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
  });

  auto reader =
      VortexFormatReader(file_system_, explicit_schema, utf8_file_, props, std::vector<std::string>{"id", "str_col"});
  ASSERT_STATUS_OK(reader.open());

  ASSERT_AND_ASSIGN(auto table, reader.take({0, 1, 2, 3}));
  ASSERT_AND_ASSIGN(auto rb, table->CombineChunksToBatch());
  EXPECT_EQ(rb->column(1)->type_id(), arrow::Type::STRING_VIEW)
      << "explicit read_schema (utf8_view) should override the non_view "
         "property.";
}

// (C) Nested-type coverage. The fix's `to_arrow_dtype` recurses through
// List / FixedSizeList / Struct; without these tests the recursion is
// unverified and a future change to either branch would silently regress.
// All three nested cases use schemaless reads (the path milvus actually
// hits at load time).

// FixedSizeList<f32> is the dense-vector shape (FloatVector). vortex maps
// it to plain FixedSizeList by default — never a view type — so this is a
// regression guard rather than a bug case. If someone changes the override
// to also touch FSL, this test catches it.
TEST_F(VortexNonViewSchemaTest, FixedSizeListFloatRoundTripsAsPlain) {
  constexpr int kDim = 8;
  arrow::Int32Builder id_b;
  arrow::FloatBuilder values_b;
  for (int64_t i = 0; i < kRows; ++i) {
    EXPECT_TRUE(id_b.Append(static_cast<int32_t>(i)).ok());
    for (int j = 0; j < kDim; ++j) {
      EXPECT_TRUE(values_b.Append(static_cast<float>(i * kDim + j)).ok());
    }
  }
  std::shared_ptr<arrow::Array> id_arr, values_arr;
  EXPECT_TRUE(id_b.Finish(&id_arr).ok());
  EXPECT_TRUE(values_b.Finish(&values_arr).ok());
  ASSERT_AND_ASSIGN(auto fsl_arr, arrow::FixedSizeListArray::FromArrays(values_arr, kDim));

  auto schema = arrow::schema({
      arrow::field("id", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
      arrow::field("vec", arrow::fixed_size_list(arrow::float32(), kDim), false,
                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
  });
  auto rb = arrow::RecordBatch::Make(schema, kRows, {id_arr, fsl_arr});
  WriteVortex(fsl_f32_file_, rb);

  std::shared_ptr<arrow::Schema> no_schema;
  auto reader =
      VortexFormatReader(file_system_, no_schema, fsl_f32_file_, properties_, std::vector<std::string>{"id", "vec"});
  ASSERT_STATUS_OK(reader.open());

  auto vec_type = reader.get_schema()->GetFieldByName("vec")->type();
  ASSERT_EQ(vec_type->id(), arrow::Type::FIXED_SIZE_LIST);
  EXPECT_EQ(vec_type->field(0)->type()->id(), arrow::Type::FLOAT);

  ASSERT_AND_ASSIGN(auto table, reader.take({0, 1, 2, 3}));
  ASSERT_AND_ASSIGN(auto rb_out, table->CombineChunksToBatch());
  ASSERT_EQ(rb_out->column(1)->type_id(), arrow::Type::FIXED_SIZE_LIST);
  auto fsl_out = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(rb_out->column(1));
  ASSERT_NE(fsl_out, nullptr);
  auto values_out = std::dynamic_pointer_cast<arrow::FloatArray>(fsl_out->values());
  ASSERT_NE(values_out, nullptr);
  // Row 0 = floats [0, 1, ..., kDim-1].
  for (int j = 0; j < kDim; ++j) {
    EXPECT_FLOAT_EQ(values_out->Value(j), static_cast<float>(j));
  }
}

// Helper to build a List<Utf8> column: each row holds two strings.
static std::shared_ptr<arrow::RecordBatch> MakeListUtf8Batch(int64_t count) {
  arrow::Int32Builder id_b;
  arrow::ListBuilder list_b(arrow::default_memory_pool(), std::make_shared<arrow::StringBuilder>());
  auto* str_b = static_cast<arrow::StringBuilder*>(list_b.value_builder());
  for (int64_t i = 0; i < count; ++i) {
    EXPECT_TRUE(id_b.Append(static_cast<int32_t>(i)).ok());
    EXPECT_TRUE(list_b.Append().ok());
    EXPECT_TRUE(str_b->Append("a" + std::to_string(i)).ok());
    EXPECT_TRUE(str_b->Append("b" + std::to_string(i)).ok());
  }
  std::shared_ptr<arrow::Array> id_arr, list_arr;
  EXPECT_TRUE(id_b.Finish(&id_arr).ok());
  EXPECT_TRUE(list_b.Finish(&list_arr).ok());

  auto schema = arrow::schema({
      arrow::field("id", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
      arrow::field("tags", arrow::list(arrow::utf8()), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
  });
  return arrow::RecordBatch::Make(schema, count, {id_arr, list_arr});
}

// Default: vortex's `to_arrow_dtype` recurses into List, so the inner Utf8
// becomes Utf8View — the bug shape extends to nested types.
TEST_F(VortexNonViewSchemaTest, SchemalessListUtf8DefaultsToListOfUtf8View) {
  WriteVortex(list_utf8_file_, MakeListUtf8Batch(kRows));

  std::shared_ptr<arrow::Schema> no_schema;
  auto reader =
      VortexFormatReader(file_system_, no_schema, list_utf8_file_, properties_, std::vector<std::string>{"id", "tags"});
  ASSERT_STATUS_OK(reader.open());

  auto tags_type = reader.get_schema()->GetFieldByName("tags")->type();
  ASSERT_EQ(tags_type->id(), arrow::Type::LIST);
  EXPECT_EQ(tags_type->field(0)->type()->id(), arrow::Type::STRING_VIEW)
      << "Without the non-view property, vortex's recursive default mapping "
         "propagates Utf8View into List children too.";
}

// Fix: with the property on, the override's `to_arrow_dtype` recurses
// through List and forces the inner Utf8 to plain Utf8.
TEST_F(VortexNonViewSchemaTest, NonViewPropertyForcesListUtf8Children) {
  WriteVortex(list_utf8_file_, MakeListUtf8Batch(kRows));

  api::Properties props = properties_;
  props[PROPERTY_READER_VORTEX_SCHEMA_NON_VIEW] = true;

  std::shared_ptr<arrow::Schema> no_schema;
  auto reader =
      VortexFormatReader(file_system_, no_schema, list_utf8_file_, props, std::vector<std::string>{"id", "tags"});
  ASSERT_STATUS_OK(reader.open());

  auto tags_type = reader.get_schema()->GetFieldByName("tags")->type();
  ASSERT_EQ(tags_type->id(), arrow::Type::LIST);
  EXPECT_EQ(tags_type->field(0)->type()->id(), arrow::Type::STRING)
      << "With non-view property, the override's recursion must reach the "
         "List element and replace Utf8View with plain Utf8.";

  ASSERT_AND_ASSIGN(auto table, reader.take({0, 1}));
  ASSERT_AND_ASSIGN(auto rb_out, table->CombineChunksToBatch());
  auto list_out = std::dynamic_pointer_cast<arrow::ListArray>(rb_out->column(1));
  ASSERT_NE(list_out, nullptr);
  auto values_out = std::dynamic_pointer_cast<arrow::StringArray>(list_out->values());
  ASSERT_NE(values_out, nullptr) << "List values should be StringArray, not StringViewArray";
  // Row 0 = ["a0","b0"], row 1 = ["a1","b1"].
  EXPECT_EQ(values_out->GetString(0), "a0");
  EXPECT_EQ(values_out->GetString(1), "b0");
  EXPECT_EQ(values_out->GetString(2), "a1");
  EXPECT_EQ(values_out->GetString(3), "b1");
}

// Same recursion concern but for Struct fields — vortex's default also
// recurses through Struct, so a Utf8 field inside Struct flips to Utf8View
// without the override.
TEST_F(VortexNonViewSchemaTest, NonViewPropertyForcesStructUtf8Children) {
  arrow::Int32Builder id_b;
  arrow::StringBuilder name_b;
  for (int64_t i = 0; i < kRows; ++i) {
    EXPECT_TRUE(id_b.Append(static_cast<int32_t>(i)).ok());
    EXPECT_TRUE(name_b.Append("n" + std::to_string(i)).ok());
  }
  std::shared_ptr<arrow::Array> id_arr, name_arr;
  EXPECT_TRUE(id_b.Finish(&id_arr).ok());
  EXPECT_TRUE(name_b.Finish(&name_arr).ok());

  auto name_field = arrow::field("name", arrow::utf8(), false);
  ASSERT_AND_ASSIGN(auto struct_arr, arrow::StructArray::Make({name_arr}, {name_field}));

  auto schema = arrow::schema({
      arrow::field("id", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
      arrow::field("info", arrow::struct_({name_field}), false,
                   arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
  });
  auto rb = arrow::RecordBatch::Make(schema, kRows, {id_arr, struct_arr});
  WriteVortex(struct_utf8_file_, rb);

  api::Properties props = properties_;
  props[PROPERTY_READER_VORTEX_SCHEMA_NON_VIEW] = true;

  std::shared_ptr<arrow::Schema> no_schema;
  auto reader =
      VortexFormatReader(file_system_, no_schema, struct_utf8_file_, props, std::vector<std::string>{"id", "info"});
  ASSERT_STATUS_OK(reader.open());

  auto info_type = reader.get_schema()->GetFieldByName("info")->type();
  ASSERT_EQ(info_type->id(), arrow::Type::STRUCT);
  EXPECT_EQ(info_type->field(0)->type()->id(), arrow::Type::STRING)
      << "With non-view property, the override's recursion must reach Struct "
         "fields and replace Utf8View with plain Utf8.";
}

}  // namespace milvus_storage
