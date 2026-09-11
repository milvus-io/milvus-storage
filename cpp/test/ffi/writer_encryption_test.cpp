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

#include <gtest/gtest.h>
#include <vector>
#include <arrow/c/bridge.h>
#include <arrow/type.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/record_batch.h>
#include <arrow/util/base64.h>
#include <arrow/util/io_util.h>
#include <parquet/column_reader.h>
#include <parquet/encryption/encryption.h>
#include <parquet/exception.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/ffi_error_code.h"

namespace {
struct WriterResources {
  LoonProperties props{};
  ArrowSchema schema{};
  ArrowArray array{};
  LoonWriterHandle writer = 0;
  LoonColumnGroups* groups = nullptr;
  ~WriterResources() {
    loon_column_groups_destroy(groups);
    loon_writer_destroy(writer);
    if (array.release)
      array.release(&array);
    if (schema.release)
      schema.release(&schema);
    loon_properties_free(&props);
  }
};

void AssertSuccess(LoonFFIResult result) {
  const bool success = loon_ffi_is_success(&result);
  const std::string message = result.message ? result.message : "";
  loon_ffi_free_result(&result);
  ASSERT_TRUE(success) << message;
}
}  // namespace

TEST(FFIWriterEncryption, Base64KeyWritesParquetReadableWithOriginalBinaryKey) {
  for (size_t key_size : {16, 24, 32}) {
    std::vector<size_t> nul_offsets{0, key_size / 2, key_size - 1};
    if (key_size == 32) {
      nul_offsets.push_back(24);
    }
    for (size_t offset : nul_offsets) {
      SCOPED_TRACE(key_size);
      SCOPED_TRACE(offset);
      std::string key(key_size, '\xff');
      key[offset] = '\0';
      const auto encoded = arrow::util::base64_encode(key);
      auto dir_result = arrow::internal::TemporaryDir::Make("loon-binary-key-");
      ASSERT_TRUE(dir_result.ok()) << dir_result.status();
      auto dir = std::move(dir_result).ValueOrDie();
      const auto root = dir->path().ToString();
      WriterResources r;
      const char* names[] = {"fs.storage_type",   "fs.root_path",   "writer.policy",  "writer.format",
                             "writer.enc.enable", "writer.enc.key", "writer.enc.meta"};
      const char* values[] = {"local", root.c_str(), "single", "parquet", "true", encoded.c_str(), "key-id"};
      ASSERT_NO_FATAL_FAILURE(AssertSuccess(loon_properties_create(names, values, 7, &r.props)));

      const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
      arrow::Int64Builder builder;
      ASSERT_TRUE(builder.AppendValues({0, 17, 511}).ok());
      auto array_result = builder.Finish();
      ASSERT_TRUE(array_result.ok());
      auto batch = arrow::RecordBatch::Make(schema, 3, {array_result.ValueOrDie()});
      ASSERT_TRUE(arrow::ExportSchema(*schema, &r.schema).ok());
      ASSERT_TRUE(arrow::ExportRecordBatch(*batch, &r.array).ok());
      ASSERT_NO_FATAL_FAILURE(AssertSuccess(loon_writer_new("data", &r.schema, &r.props, &r.writer)));
      ASSERT_NE(r.writer, 0);
      ASSERT_NO_FATAL_FAILURE(AssertSuccess(loon_writer_write(r.writer, &r.array)));
      ASSERT_NO_FATAL_FAILURE(AssertSuccess(loon_writer_close(r.writer, nullptr, nullptr, 0, &r.groups)));
      ASSERT_NE(r.groups, nullptr);
      ASSERT_EQ(r.groups->num_of_column_groups, 1);
      const auto& group = r.groups->column_group_array[0];
      ASSERT_EQ(group.num_of_files, 1);

      // Bypass Loon and its Base64 decoder when reading: the original binary
      // key must decrypt both the footer and the actual column values.
      parquet::ReaderProperties reader_props;
      reader_props.file_decryption_properties(parquet::FileDecryptionProperties::Builder().footer_key(key)->build());
      auto reader = parquet::ParquetFileReader::OpenFile(root + group.files[0].path, false, reader_props);
      ASSERT_EQ(reader->metadata()->num_rows(), 3);
      auto column = std::static_pointer_cast<parquet::Int64Reader>(reader->RowGroup(0)->Column(0));
      int64_t actual[3]{};
      int64_t count = 0;
      ASSERT_EQ(column->ReadBatch(3, nullptr, nullptr, actual, &count), 3);
      ASSERT_EQ(count, 3);
      EXPECT_EQ(std::vector<int64_t>(actual, actual + 3), (std::vector<int64_t>{0, 17, 511}));

      // These prefixes are valid AES lengths, which let the old C-string
      // truncation bug silently write data with the wrong key.
      if (key_size == 32 && (offset == 16 || offset == 24)) {
        parquet::ReaderProperties truncated_key_props;
        truncated_key_props.file_decryption_properties(
            parquet::FileDecryptionProperties::Builder().footer_key(key.substr(0, offset))->build());
        EXPECT_THROW(parquet::ParquetFileReader::OpenFile(root + group.files[0].path, false, truncated_key_props),
                     parquet::ParquetException);
      }
    }
  }
}

TEST(FFIWriterEncryption, RejectsMalformedOrInvalidLengthBase64Key) {
  // Includes input whose valid prefix decodes to an AES key: accepting that
  // prefix would silently write with a different key than the supplied text.
  const char* invalid_keys[] = {"",
                                "not-base64!",
                                "YQ==",
                                "Zm9vdGVyX2tleV8xNkJfXw==trailing",
                                "Zm9vdGVyX2tleV8xNkJfXw",
                                "Zm9vdGVyX2tleV8xNkJfXx=="};
  for (size_t i = 0; i < sizeof(invalid_keys) / sizeof(invalid_keys[0]); ++i) {
    SCOPED_TRACE(i);
    const char* keys[] = {"fs.storage_type", "fs.root_path",      "writer.policy",
                          "writer.format",   "writer.enc.enable", "writer.enc.key"};
    const char* values[] = {"local", "/tmp", "single", "parquet", "true", invalid_keys[i]};
    LoonProperties props{};
    auto result = loon_properties_create(keys, values, 6, &props);
    ASSERT_TRUE(loon_ffi_is_success(&result));
    loon_ffi_free_result(&result);

    ArrowSchema schema{};
    ASSERT_TRUE(arrow::ExportSchema(*arrow::schema({arrow::field("id", arrow::int64())}), &schema).ok());
    LoonWriterHandle writer = 0;
    result = loon_writer_new("invalid-base64-key", &schema, &props, &writer);
    EXPECT_EQ(result.err_code, LOON_INVALID_PROPERTIES);
    EXPECT_EQ(writer, 0);
    if (!loon_ffi_is_success(&result)) {
      EXPECT_NE(std::string(loon_ffi_get_errmsg(&result)).find("writer.enc.key"), std::string::npos);
      if (invalid_keys[i][0] != '\0') {
        EXPECT_EQ(std::string(loon_ffi_get_errmsg(&result)).find(invalid_keys[i]), std::string::npos);
      }
    }
    loon_ffi_free_result(&result);
    loon_writer_destroy(writer);
    if (schema.release) {
      schema.release(&schema);
    }
    loon_properties_free(&props);
  }
}
