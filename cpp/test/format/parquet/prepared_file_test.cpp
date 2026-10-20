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

#include <atomic>
#include <thread>
#include <gtest/gtest.h>
#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/memory.h>
#include <parquet/arrow/writer.h>
#include <parquet/encryption/encryption.h>

#include "milvus-storage/common/constants.h"
#include "milvus-storage/format/parquet/file_reader.h"

namespace milvus_storage::test {
namespace {

// Every input gets its own BufferReader. Track footer intersections separately
// from data reads; creating a prepared reader must issue neither kind of read.
struct ReadCounts {
  std::atomic<int> opens{0};
  std::atomic<int> handles{0};
  std::atomic<int> footer_reads{0};
  std::atomic<int> data_reads{0};
  std::atomic<bool> fail_reads{false};
  std::atomic<bool> fail_opens{false};
};

class CountedInput : public arrow::io::RandomAccessFile {
  public:
  CountedInput(std::shared_ptr<arrow::Buffer> data, std::shared_ptr<ReadCounts> counts)
      : input_(std::move(data)), counts_(std::move(counts)) {
    ++counts_->handles;
    auto buffer = input_.buffer();
    if (buffer->size() < 8) {
      footer_start_ = 0;
      return;
    }
    const auto* tail = buffer->data() + buffer->size() - 8;
    uint32_t footer_size = static_cast<uint32_t>(tail[0]) | static_cast<uint32_t>(tail[1]) << 8 |
                           static_cast<uint32_t>(tail[2]) << 16 | static_cast<uint32_t>(tail[3]) << 24;
    footer_start_ = buffer->size() - 8 - footer_size;
  }
  ~CountedInput() override { (void)CountedInput::Close(); }
  arrow::Status Close() override {
    if (!input_.closed()) {
      --counts_->handles;
    }
    return input_.Close();
  }
  bool closed() const override { return input_.closed(); }
  arrow::Result<int64_t> Tell() const override { return input_.Tell(); }
  arrow::Status Seek(int64_t position) override { return input_.Seek(position); }
  arrow::Result<int64_t> GetSize() override { return input_.GetSize(); }
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    ARROW_ASSIGN_OR_RAISE(auto pos, Tell());
    ARROW_RETURN_NOT_OK(Record(pos, nbytes));
    return input_.Read(nbytes, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto pos, Tell());
    ARROW_RETURN_NOT_OK(Record(pos, nbytes));
    return input_.Read(nbytes);
  }
  arrow::Result<int64_t> ReadAt(int64_t pos, int64_t nbytes, void* out) override {
    ARROW_RETURN_NOT_OK(Record(pos, nbytes));
    return input_.ReadAt(pos, nbytes, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t pos, int64_t nbytes) override {
    ARROW_RETURN_NOT_OK(Record(pos, nbytes));
    return input_.ReadAt(pos, nbytes);
  }

  private:
  arrow::Status Record(int64_t pos, int64_t nbytes) {
    if (counts_->fail_reads) {
      return arrow::Status::IOError("injected read failure");
    }
    if (pos + nbytes > footer_start_) {
      ++counts_->footer_reads;
    } else {
      ++counts_->data_reads;
    }
    return arrow::Status::OK();
  }
  arrow::io::BufferReader input_;
  std::shared_ptr<ReadCounts> counts_;
  int64_t footer_start_;
};

// Keep the key verbatim; any root addition or path normalization fails the test.
class CountedFileSystem : public arrow::fs::SubTreeFileSystem {
  public:
  CountedFileSystem(std::shared_ptr<arrow::Buffer> data, std::string key)
      : arrow::fs::SubTreeFileSystem("", std::make_shared<arrow::fs::LocalFileSystem>()),
        data_(std::move(data)),
        key_(std::move(key)),
        counts(std::make_shared<ReadCounts>()) {}
  using arrow::fs::SubTreeFileSystem::OpenInputFile;
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    if (counts->fail_opens || path != key_) {
      return arrow::Status::IOError("missing file: ", path);
    }
    ++counts->opens;
    return std::make_shared<CountedInput>(data_, counts);
  }
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    return OpenInputFile(info.path());
  }
  std::shared_ptr<ReadCounts> counts;

  private:
  std::shared_ptr<arrow::Buffer> data_;
  std::string key_;
};

const std::string kKey = "root//objects/../packed.parquet";
const std::string kEncryptionKey = "0123456789abcdef0123456789abcdef";

arrow::Result<std::shared_ptr<arrow::Buffer>> MakePackedFile(
    int32_t base = 0, std::shared_ptr<parquet::FileEncryptionProperties> encryption = nullptr, bool field_id = true) {
  RowGroupMetadataVector groups;
  for (int i = 0; i < 3; ++i) {
    groups.Add(RowGroupMetadata(8, 2, 2 * i));
  }
  GroupFieldIDList fields(std::vector<std::vector<int>>{{100}});
  auto kv = arrow::key_value_metadata({ROW_GROUP_META_KEY, STORAGE_VERSION_KEY, GROUP_FIELD_ID_LIST_META_KEY},
                                      {groups.Serialize(), "test", fields.Serialize()});
  auto schema = arrow::schema(
      {arrow::field("value", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))}, kv);
  if (!field_id) {
    schema = arrow::schema({schema->field(0)->RemoveMetadata()}, kv);
  }
  arrow::Int32Builder values;
  ARROW_RETURN_NOT_OK(values.AppendValues({base, base + 1, base + 2, base + 3, base + 4, base + 5}));
  ARROW_ASSIGN_OR_RAISE(auto array, values.Finish());
  auto table = arrow::Table::Make(schema, {array});
  ARROW_ASSIGN_OR_RAISE(auto output, arrow::io::BufferOutputStream::Create());
  parquet::WriterProperties::Builder props;
  if (encryption) {
    props.encryption(std::move(encryption));
  }
  ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), output, 2, props.build(),
                                                 parquet::ArrowWriterProperties::Builder().store_schema()->build()));
  return output->Finish();
}

#define PREPARED_ASSIGN(lhs, expr)                         \
  auto lhs##_result = (expr);                              \
  ASSERT_TRUE(lhs##_result.ok()) << lhs##_result.status(); \
  auto lhs = std::move(lhs##_result).ValueOrDie()

arrow::Result<int32_t> ReadGroup(FileRowGroupReader& reader, int group) {
  ARROW_RETURN_NOT_OK(reader.SetRowGroupOffsetAndCount(group, 1));
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(reader.ReadNextRowGroup(&table));
  if (!table || table->num_rows() != 2) {
    return arrow::Status::Invalid("unexpected batch");
  }
  return std::static_pointer_cast<arrow::Int32Array>(table->column(0)->chunk(0))->Value(0);
}

class PreparedFileTest : public ::testing::Test {
  protected:
  void SetUp() override {
    PREPARED_ASSIGN(data, MakePackedFile());
    fs = std::make_shared<CountedFileSystem>(data, kKey);
  }
  std::shared_ptr<CountedFileSystem> fs;
};

TEST_F(PreparedFileTest, ReusesFooterWithoutIOAndReleasesProbe) {
  PREPARED_ASSIGN(prepared, PreparedParquetFile::Open(fs, kKey));
  EXPECT_EQ(fs->counts->opens, 1);
  EXPECT_EQ(fs->counts->handles, 0);
  EXPECT_GT(fs->counts->footer_reads, 0);
  auto footer_reads = fs->counts->footer_reads.load();
  EXPECT_EQ(prepared->row_group_metadata().size(), 3);
  EXPECT_EQ(prepared->field_id_mapping().size(), 1);
  EXPECT_EQ(prepared->storage_version(), "test");
  PREPARED_ASSIGN(first, prepared->MakeReader(nullptr, 8));
  PREPARED_ASSIGN(second, prepared->MakeReader(nullptr, 16));
  EXPECT_EQ(fs->counts->footer_reads, footer_reads);
  EXPECT_EQ(fs->counts->data_reads, 0);
  EXPECT_EQ(first->file_metadata()->GetParquetMetadata(), second->file_metadata()->GetParquetMetadata());
  PREPARED_ASSIGN(value, ReadGroup(*first, 0));
  EXPECT_EQ(value, 0);
  EXPECT_TRUE(first->Close().ok());
  PREPARED_ASSIGN(other, ReadGroup(*second, 2));
  EXPECT_EQ(other, 4);
  EXPECT_EQ(fs->counts->footer_reads, footer_reads);
  EXPECT_GT(fs->counts->data_reads, 0);
}

TEST_F(PreparedFileTest, ConcurrentIndependentRangesAndCapturedOwner) {
  parquet::ReaderProperties props;
  parquet::FileDecryptionProperties::Builder allowed;
  props.file_decryption_properties(allowed.footer_key(kEncryptionKey)->plaintext_files_allowed()->build());
  PREPARED_ASSIGN(prepared, PreparedParquetFile::Open(fs, kKey, props));
  auto footer_reads = fs->counts->footer_reads.load();
  std::weak_ptr<PreparedParquetFile> owner = prepared;
  std::atomic<bool> start{false};
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(12);
  for (int i = 0; i < 12; ++i) {
    workers.emplace_back([context = prepared, &start, &failures, i]() {
      while (!start.load()) {
        std::this_thread::yield();
      }
      auto reader = context->MakeReader(nullptr, 8);
      if (!reader.ok()) {
        ++failures;
        return;
      }
      auto value = ReadGroup(*reader.ValueOrDie(), i % 3);
      if (!value.ok() || value.ValueOrDie() != 2 * (i % 3)) {
        ++failures;
      }
    });
  }
  prepared.reset();
  EXPECT_FALSE(owner.expired());
  start = true;
  for (auto& worker : workers) {
    worker.join();
  }
  EXPECT_EQ(failures, 0);
  EXPECT_TRUE(owner.expired());
  EXPECT_EQ(fs->counts->footer_reads, footer_reads);
  EXPECT_EQ(fs->counts->handles, 0);
}

TEST_F(PreparedFileTest, MetadataOutlivesOwnerButNotLastReader) {
  PREPARED_ASSIGN(prepared, PreparedParquetFile::Open(fs, kKey));
  PREPARED_ASSIGN(reader, prepared->MakeReader());
  std::weak_ptr<parquet::FileMetaData> metadata = reader->file_metadata()->GetParquetMetadata();
  prepared.reset();
  PREPARED_ASSIGN(value, ReadGroup(*reader, 1));
  EXPECT_EQ(value, 2);
  EXPECT_TRUE(reader->Close().ok());
  EXPECT_FALSE(metadata.expired());  // Close retains the wrapper's metadata.
  reader.reset();
  EXPECT_TRUE(metadata.expired());
  EXPECT_EQ(fs->counts->handles, 0);
}

TEST_F(PreparedFileTest, LegacyAndDistinctContextsOpenIndependently) {
  PREPARED_ASSIGN(first, PreparedParquetFile::Open(fs, kKey));
  PREPARED_ASSIGN(second, PreparedParquetFile::Open(fs, kKey));
  PREPARED_ASSIGN(a, first->MakeReader());
  PREPARED_ASSIGN(b, second->MakeReader());
  PREPARED_ASSIGN(legacy, FileRowGroupReader::Make(fs, kKey));
  EXPECT_NE(a->file_metadata()->GetParquetMetadata(), b->file_metadata()->GetParquetMetadata());
  EXPECT_NE(a->file_metadata()->GetParquetMetadata(), legacy->file_metadata()->GetParquetMetadata());
  EXPECT_EQ(fs->counts->opens, 5);
}

TEST_F(PreparedFileTest, SamePathDifferentFilesystemsAreIsolated) {
  PREPARED_ASSIGN(data, MakePackedFile(100));
  auto other_fs = std::make_shared<CountedFileSystem>(data, kKey);
  PREPARED_ASSIGN(first, PreparedParquetFile::Open(fs, kKey));
  PREPARED_ASSIGN(second, PreparedParquetFile::Open(other_fs, kKey));
  PREPARED_ASSIGN(a, first->MakeReader());
  PREPARED_ASSIGN(b, second->MakeReader());
  PREPARED_ASSIGN(value_a, ReadGroup(*a, 0));
  PREPARED_ASSIGN(value_b, ReadGroup(*b, 0));
  EXPECT_EQ(value_a, 0);
  EXPECT_EQ(value_b, 100);
}

TEST_F(PreparedFileTest, MoreThan256ContextsKeepTheirFooters) {
  PREPARED_ASSIGN(data, MakePackedFile());
  std::vector<std::shared_ptr<PreparedParquetFile>> contexts;
  std::vector<std::shared_ptr<CountedFileSystem>> filesystems;
  contexts.reserve(257);
  filesystems.reserve(257);
  for (int i = 0; i < 257; ++i) {
    auto key = kKey + std::to_string(i);
    auto filesystem = std::make_shared<CountedFileSystem>(data, key);
    PREPARED_ASSIGN(context, PreparedParquetFile::Open(filesystem, key));
    contexts.push_back(std::move(context));
    filesystems.push_back(std::move(filesystem));
  }
  for (size_t i = 0; i < contexts.size(); ++i) {
    auto footer_reads = filesystems[i]->counts->footer_reads.load();
    {
      PREPARED_ASSIGN(reader, contexts[i]->MakeReader());
      PREPARED_ASSIGN(value, ReadGroup(*reader, 0));
      EXPECT_EQ(value, 0);
    }
    EXPECT_EQ(filesystems[i]->counts->footer_reads, footer_reads);
    EXPECT_EQ(filesystems[i]->counts->handles, 0);
  }
}

TEST_F(PreparedFileTest, FailedOpenCanBeRetriedAndReadFailureIsIsolated) {
  EXPECT_FALSE(PreparedParquetFile::Open(nullptr, kKey).ok());
  auto missing = PreparedParquetFile::Open(fs, "absent");
  ASSERT_FALSE(missing.ok());
  EXPECT_TRUE(missing.status().IsIOError());
  fs->counts->fail_reads = true;
  auto failed = PreparedParquetFile::Open(fs, kKey);
  ASSERT_FALSE(failed.ok());
  EXPECT_TRUE(failed.status().IsIOError());
  EXPECT_EQ(fs->counts->handles, 0);
  fs->counts->fail_reads = false;
  PREPARED_ASSIGN(prepared, PreparedParquetFile::Open(fs, kKey));
  PREPARED_ASSIGN(reader, prepared->MakeReader());
  fs->counts->fail_reads = true;
  EXPECT_FALSE(ReadGroup(*reader, 0).ok());
  reader.reset();
  fs->counts->fail_reads = false;
  PREPARED_ASSIGN(other, prepared->MakeReader());
  EXPECT_TRUE(ReadGroup(*other, 0).ok());
}

TEST_F(PreparedFileTest, ReaderCreationFailuresDoNotInvalidateContext) {
  PREPARED_ASSIGN(prepared, PreparedParquetFile::Open(fs, kKey));
  fs->counts->fail_opens = true;
  auto failed = prepared->MakeReader();
  ASSERT_FALSE(failed.ok());
  EXPECT_TRUE(failed.status().IsIOError());
  fs->counts->fail_opens = false;
  auto invalid_schema = arrow::schema({arrow::field("no_field_id", arrow::int32())});
  EXPECT_FALSE(prepared->MakeReader(invalid_schema).ok());
  EXPECT_EQ(fs->counts->handles, 0);
  PREPARED_ASSIGN(reader, prepared->MakeReader());
  EXPECT_TRUE(ReadGroup(*reader, 0).ok());
}

TEST_F(PreparedFileTest, MissingFieldIdReturnsError) {
  PREPARED_ASSIGN(data, MakePackedFile(0, nullptr, false));
  auto invalid_fs = std::make_shared<CountedFileSystem>(data, kKey);
  auto prepared = PreparedParquetFile::Open(invalid_fs, kKey);
  ASSERT_FALSE(prepared.ok());
  EXPECT_TRUE(prepared.status().IsInvalid());
  auto legacy = FileRowGroupReader::Make(invalid_fs, kKey);
  ASSERT_FALSE(legacy.ok());
  EXPECT_TRUE(legacy.status().IsInvalid());
  EXPECT_EQ(invalid_fs->counts->handles, 0);
}

TEST_F(PreparedFileTest, CorruptFooterReturnsErrorWithoutRetainingHandles) {
  auto corrupt_fs = std::make_shared<CountedFileSystem>(arrow::Buffer::FromString("invalid footer"), kKey);
  auto legacy = FileRowGroupReader::Make(corrupt_fs, kKey);
  auto prepared = PreparedParquetFile::Open(corrupt_fs, kKey);
  ASSERT_FALSE(legacy.ok());
  ASSERT_FALSE(prepared.ok());
  EXPECT_EQ(legacy.status().code(), prepared.status().code());
  EXPECT_EQ(corrupt_fs->counts->handles, 0);
}

TEST_F(PreparedFileTest, ProjectionAndMissingFields) {
  PREPARED_ASSIGN(prepared, PreparedParquetFile::Open(fs, kKey));
  auto schema = arrow::schema(
      {arrow::field("renamed", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
       arrow::field("missing", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"999"}))});
  PREPARED_ASSIGN(reader, prepared->MakeReader(schema));
  ASSERT_TRUE(reader->SetRowGroupOffsetAndCount(1, 1).ok());
  std::shared_ptr<arrow::Table> table;
  ASSERT_TRUE(reader->ReadNextRowGroup(&table).ok());
  EXPECT_EQ(table->column(1)->null_count(), 2);
  EXPECT_TRUE(reader->schema()->field(1)->nullable());
  EXPECT_EQ(std::static_pointer_cast<arrow::Int32Array>(table->column(0)->chunk(0))->Value(0), 2);
}

TEST_F(PreparedFileTest, PlaintextDecryptionPolicyAndConfigurationIsolation) {
  parquet::ReaderProperties props;
  parquet::FileDecryptionProperties::Builder denied;
  props.file_decryption_properties(denied.footer_key(kEncryptionKey)->build());
  auto legacy = FileRowGroupReader::Make(fs, kKey, DEFAULT_READ_BUFFER_SIZE, props);
  auto prepared_denied = PreparedParquetFile::Open(fs, kKey, props);
  ASSERT_FALSE(legacy.ok());
  ASSERT_FALSE(prepared_denied.ok());
  EXPECT_EQ(legacy.status().code(), prepared_denied.status().code());

  parquet::FileDecryptionProperties::Builder allowed;
  props.file_decryption_properties(allowed.footer_key(kEncryptionKey)->plaintext_files_allowed()->build());
  PREPARED_ASSIGN(prepared, PreparedParquetFile::Open(fs, kKey, props));
  // Mutate the caller's configuration after Open; context keeps its snapshot.
  parquet::FileDecryptionProperties::Builder replacement;
  props.file_decryption_properties(replacement.footer_key(kEncryptionKey)->build());
  auto reads = fs->counts->footer_reads.load();
  PREPARED_ASSIGN(first, prepared->MakeReader());
  PREPARED_ASSIGN(second, prepared->MakeReader());
  EXPECT_EQ(fs->counts->footer_reads, reads);
  EXPECT_EQ(first->file_metadata()->GetParquetMetadata(), second->file_metadata()->GetParquetMetadata());
  EXPECT_TRUE(ReadGroup(*first, 0).ok());
  EXPECT_TRUE(ReadGroup(*second, 2).ok());
}

TEST_F(PreparedFileTest, EncryptedFooterAndEncryptedColumnsAlwaysOpenFresh) {
  for (bool plaintext_footer : {false, true}) {
    parquet::FileEncryptionProperties::Builder enc(kEncryptionKey);
    if (plaintext_footer) {
      enc.set_plaintext_footer();
    }
    PREPARED_ASSIGN(data, MakePackedFile(0, enc.build()));
    auto encrypted_fs = std::make_shared<CountedFileSystem>(data, kKey);
    parquet::ReaderProperties props;
    parquet::FileDecryptionProperties::Builder dec;
    props.file_decryption_properties(dec.footer_key(kEncryptionKey)->build());
    PREPARED_ASSIGN(prepared, PreparedParquetFile::Open(encrypted_fs, kKey, props));
    EXPECT_EQ(encrypted_fs->counts->handles, 0);
    auto reads = encrypted_fs->counts->footer_reads.load();
    PREPARED_ASSIGN(first, prepared->MakeReader());
    EXPECT_GT(encrypted_fs->counts->footer_reads, reads);
    PREPARED_ASSIGN(second, prepared->MakeReader());
    EXPECT_NE(first->file_metadata()->GetParquetMetadata(), second->file_metadata()->GetParquetMetadata());
    EXPECT_TRUE(ReadGroup(*first, 0).ok());
    ASSERT_TRUE(first->Close().ok());
    EXPECT_TRUE(ReadGroup(*second, 2).ok());
    parquet::FileDecryptionProperties::Builder wrong;
    props.file_decryption_properties(wrong.footer_key("abcdef0123456789abcdef0123456789")->build());
    EXPECT_FALSE(PreparedParquetFile::Open(encrypted_fs, kKey, props).ok());
  }
}

#undef PREPARED_ASSIGN
}  // namespace
}  // namespace milvus_storage::test
