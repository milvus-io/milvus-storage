

#include <memory>
#include <type_traits>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include "milvus-storage/file/fragment.h"
#include "gmock/gmock.h"
#include "milvus-storage/reader/multi_files_sequential_reader.h"
#include "milvus-storage/storage/options.h"
#include "test_util.h"
#include "arrow/table.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/config.h"
namespace milvus_storage {
TEST(MultiFilesSeqReaderTest, ReadTest) {
  auto arrow_schema = CreateArrowSchema({"pk_field"}, {arrow::int64()});
  arrow::Int64Builder pk_builder;
  ASSERT_STATUS_OK(pk_builder.Append(1));
  ASSERT_STATUS_OK(pk_builder.Append(2));
  ASSERT_STATUS_OK(pk_builder.Append(3));
  std::shared_ptr<arrow::Array> pk_array;
  ASSERT_STATUS_OK(pk_builder.Finish(&pk_array));
  auto rec_batch = arrow::RecordBatch::Make(arrow_schema, 3, {pk_array});

  auto conf = ArrowFileSystemConfig();
  conf.storage_type = "local";
  conf.root_path = "/tmp";
  ArrowFileSystemSingleton::GetInstance().Init(conf);
  ArrowFileSystemPtr fs = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();
  ASSERT_AND_ARROW_ASSIGN(auto f1, fs->OpenOutputStream("/tmp/file1"));
  ASSERT_AND_ARROW_ASSIGN(auto w1, parquet::arrow::FileWriter::Open(*arrow_schema, arrow::default_memory_pool(), f1));
  ASSERT_STATUS_OK(w1->WriteRecordBatch(*rec_batch));
  ASSERT_STATUS_OK(w1->Close());
  ASSERT_STATUS_OK(f1->Close());
  ASSERT_AND_ARROW_ASSIGN(auto f2, fs->OpenOutputStream("/tmp/file2"));
  ASSERT_AND_ARROW_ASSIGN(auto w2, parquet::arrow::FileWriter::Open(*arrow_schema, arrow::default_memory_pool(), f2));
  ASSERT_STATUS_OK(w2->WriteRecordBatch(*rec_batch));
  ASSERT_STATUS_OK(w2->Close());
  ASSERT_STATUS_OK(f2->Close());

  Fragment frag(1);
  frag.add_file("/tmp/file1");
  frag.add_file("/tmp/file2");
  ReadOptions opt{.columns = {"pk_field"}};
  SchemaOptions schema_options{.primary_column = "pk_field"};
  MultiFilesSequentialReader r(*fs, {frag}, arrow_schema, schema_options, opt);
  ASSERT_AND_ARROW_ASSIGN(auto table, r.ToTable());
  ASSERT_AND_ARROW_ASSIGN(auto combined_table, table->CombineChunks());
  auto pk_res = std::dynamic_pointer_cast<arrow::Int64Array>(combined_table->GetColumnByName("pk_field")->chunk(0));
  std::vector<int64_t> pks;
  pks.reserve(pk_res->length());
  for (int i = 0; i < pk_res->length(); ++i) {
    pks.push_back(pk_res->Value(i));
  }
  ASSERT_THAT(pks, testing::ElementsAre(1, 2, 3, 1, 2, 3));
  ASSERT_STATUS_OK(r.Close());
  ArrowFileSystemSingleton::GetInstance().Release();
}

}  // namespace milvus_storage
