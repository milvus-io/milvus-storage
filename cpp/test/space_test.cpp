#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>

#include <memory>

#include "arrow/array/builder_binary.h"
#include "arrow/array/builder_primitive.h"
#include "filter/constant_filter.h"
#include "gtest/gtest.h"

TEST(SpaceTest, SpaceCtor) {
  // arrow::SchemaBuilder schema_builder;
  // auto pk_field = std::make_shared<arrow::Field>("pk_field", arrow::int64());
  // auto ts_field = std::make_shared<arrow::Field>("ts_field", arrow::int64());
  // auto vec_field = std::make_shared<arrow::Field>("vec_field", arrow::fixed_size_binary(10));
  // auto status = schema_builder.AddField(pk_field);
  // ASSERT_TRUE(status.ok());
  // status = schema_builder.AddField(ts_field);
  // ASSERT_TRUE(status.ok());
  // status = schema_builder.AddField(vec_field);
  // ASSERT_TRUE(status.ok());

  // auto options = std::make_shared<SpaceOption>();
  // options->primary_column = "pk_field";
  // options->version_column = "ts_field";
  // options->vector_column = "vec_field";
  // auto schema = schema_builder.Finish().ValueOrDie();
  // DefaultSpace space(schema, options);

  // arrow::Int64Builder pk_builder;
  // pk_builder.Append(1);
  // pk_builder.Append(2);
  // pk_builder.Append(3);
  // arrow::Int64Builder ts_builder;
  // ts_builder.Append(1);
  // ts_builder.Append(2);
  // ts_builder.Append(3);
  // arrow::FixedSizeBinaryBuilder vec_builder(arrow::fixed_size_binary(10));
  // vec_builder.Append("1234567890");
  // vec_builder.Append("1234567890");
  // vec_builder.Append("1234567890");

  // std::shared_ptr<arrow::Array> pk_array;
  // pk_builder.Finish(&pk_array);
  // std::shared_ptr<arrow::Array> ts_array;
  // ts_builder.Finish(&ts_array);
  // std::shared_ptr<arrow::Array> vec_array;
  // vec_builder.Finish(&vec_array);

  // auto rec_batch = arrow::RecordBatch::Make(schema, 3, {pk_array, ts_array, vec_array});
  // auto reader = arrow::RecordBatchReader::Make({rec_batch}, schema).ValueOrDie();

  // auto write_option = WriteOption{10};
  // space.Write(reader.get(), &write_option);

  // auto read_option = std::make_shared<ReadOptions>();
  // auto res_reader = space.Read(read_option);
  // auto batch_count = 0;
  // auto rec_count = 0;
  // for (auto rec = res_reader->Next(); rec.ok(); rec = res_reader->Next()) {
  //   if (rec.ValueOrDie() == nullptr) {
  //     break;
  //   }
  //   batch_count++;
  //   rec_count += rec.ValueOrDie()->num_rows();
  // }
  // ASSERT_EQ(1, batch_count);
  // ASSERT_EQ(3, rec_count);

  // arrow::Int64Builder pk_delete_builder;
  // pk_delete_builder.Append(1);
  // pk_delete_builder.Finish(&pk_array);
  // arrow::Int64Builder ts_delete_builder;
  // ts_delete_builder.Append(0);
  // ts_delete_builder.Finish(&ts_array);
  // arrow::SchemaBuilder delete_schema_builder;
  // delete_schema_builder.AddField(pk_field);
  // delete_schema_builder.AddField(ts_field);
  // schema = delete_schema_builder.Finish().ValueOrDie();
  // auto delete_batch = arrow::RecordBatch::Make(schema, 1, {pk_array, ts_array});
  // auto delete_reader = arrow::RecordBatchReader::Make({delete_batch}, schema).ValueOrDie();

  // space.Delete(delete_reader.get());
  // res_reader = space.Read(read_option);
  // rec_count = 0;
  // for (auto rec = res_reader->Next(); rec.ok(); rec = res_reader->Next()) {
  //   if (rec.ValueOrDie() == nullptr) {
  //     break;
  //   }
  //   rec_count += rec.ValueOrDie()->num_rows();
  // }
  // ASSERT_EQ(3, rec_count);
}