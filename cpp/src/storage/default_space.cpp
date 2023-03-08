#include "default_space.h"

#include <arrow/type.h>

#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>
#include <memory>
#include <utility>

#include "../exception.h"
#include "../format/parquet/file_writer.h"
#include "arrow/filesystem/mockfs.h"
#include "arrow/record_batch.h"

void WriteManifestFile(const Manifest *manifest);

DefaultSpace::DefaultSpace(std::shared_ptr<arrow::Schema> schema,
                           std::shared_ptr<SpaceOption> &options)
    : Space(options) {
  if (!schema->GetFieldByName(options->primary_column) ||
      !options->version_column.empty() &&
          !schema->GetFieldByName(options->version_column)) {
    throw StorageException("version column not found");
  }

  arrow::SchemaBuilder scalar_schema_builder;
  arrow::SchemaBuilder vector_schema_builder;

  arrow::Status status;
  for (const auto &field : schema->fields()) {
    if (field->name() == options->primary_column ||
        field->name() == options->vector_column) {
      status = vector_schema_builder.AddField(field);
    } else {
      status = scalar_schema_builder.AddField(field);
    }
    if (!status.ok()) {
      throw StorageException(status.CodeAsString());
    }
  }

  manifest_ = std::make_unique<Manifest>(
      schema, scalar_schema_builder.Finish().ValueOrDie(),
      vector_schema_builder.Finish().ValueOrDie());
  fs_ =
      std::make_unique<arrow::fs::internal::MockFileSystem>(arrow::fs::kNoTime);
}

void DefaultSpace::Write(arrow::RecordBatchReader *reader,
                         WriteOption *option) {
  if (!reader->schema()->Equals(*this->manifest_->get_schema())) {
    throw StorageException("schema not match");
  }

  auto scalar_schema = this->manifest_->get_scalar_schema(),
       vector_schema = this->manifest_->get_vector_schema();

  std::vector<std::shared_ptr<arrow::Array>> scalar_cols;
  std::vector<std::shared_ptr<arrow::Array>> vector_cols;

  ParquetFileWriter *scalar_writer = nullptr;
  ParquetFileWriter *vector_writer = nullptr;

  std::vector<std::string> scalar_files;
  std::vector<std::string> vector_files;

  for (auto rec = reader->Next(); rec.ok(); rec = reader->Next()) {
    auto batch = rec.ValueOrDie();
    if (batch->num_rows() == 0) {
      continue;
    }
    auto cols = batch->columns();
    for (int i = 0; i < cols.size(); ++i) {
      if (scalar_schema->GetFieldByName(batch->column_name(i))) {
        scalar_cols.emplace_back(cols[i]);
      }
      if (vector_schema->GetFieldByName(batch->column_name(i))) {
        vector_cols.emplace_back(cols[i]);
      }
    }

    auto scalar_record =
        arrow::RecordBatch::Make(scalar_schema, batch->num_rows(), scalar_cols);
    auto vector_record =
        arrow::RecordBatch::Make(vector_schema, batch->num_rows(), vector_cols);

    if (!scalar_writer) {
      auto scalar_file_id = boost::uuids::random_generator()();
      auto scalar_file_path =
          boost::uuids::to_string(scalar_file_id) + ".parquet";
      ParquetFileWriter new_scalar_writer(scalar_schema.get(), fs_.get(),
                                          scalar_file_path);
      scalar_writer = &new_scalar_writer;

      auto vector_file_id = boost::uuids::random_generator()();
      auto vector_file_path =
          boost::uuids::to_string(vector_file_id) + ".parquet";
      ParquetFileWriter new_vector_writer(vector_schema.get(), fs_.get(),
                                          vector_file_path);
      vector_writer = &new_vector_writer;

      scalar_files.emplace_back(scalar_file_path);
      vector_files.emplace_back(vector_file_path);
    }

    scalar_writer->Write(scalar_record.get());
    vector_writer->Write(vector_record.get());

    if (scalar_writer->count() >= option->max_record_per_file) {
      scalar_writer->Close();
      vector_writer->Close();
      scalar_writer = nullptr;
      vector_writer = nullptr;
    }
  }

  if (scalar_writer) {
    scalar_writer->Close();
    vector_writer->Close();
    scalar_writer = nullptr;
    vector_writer = nullptr;
  }

  manifest_->AddDataFiles(scalar_files, vector_files);
  WriteManifestFile(manifest_.get());
}

std::shared_ptr<arrow::RecordBatch> DefaultSpace::Read(
    std::shared_ptr<ReadOption> option) {
  return nullptr;
}

void WriteManifestFile(const Manifest *manifest) {}