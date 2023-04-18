
#include <numeric>

#include "arrow/array/builder_primitive.h"
#include "common/fs_util.h"
#include "format/parquet/file_writer.h"
#include "reader/record_reader.h"
#include "storage/default_space.h"
#include "storage/deleteset.h"
#include "arrow/util/uri.h"
#include "common/utils.h"
namespace milvus_storage {

DefaultSpace::DefaultSpace(std::shared_ptr<Schema> schema, std::shared_ptr<SpaceOptions>& options)
    : schema_(std::move(schema)), Space(options) {
  delete_set_ = std::make_unique<DeleteSet>(*this);
  manifest_ = std::make_unique<Manifest>(options, schema_);
}

Status DefaultSpace::Init() {
  RETURN_NOT_OK(delete_set_->Build());
  ASSIGN_OR_RETURN_NOT_OK(fs_, BuildFileSystem(options_->uri));
  arrow::internal::Uri uri_parser;
  RETURN_ARROW_NOT_OK(uri_parser.Parse(options_->uri));
  base_path_ = uri_parser.path();
  return Status::OK();
}

Status DefaultSpace::Write(arrow::RecordBatchReader* reader, WriteOption* option) {
  if (!reader->schema()->Equals(*this->schema_->schema())) {
    return Status::InvalidArgument("Schema not match");
  }

  // remove duplicated codes
  auto scalar_schema = this->schema_->scalar_schema(), vector_schema = this->schema_->vector_schema();

  std::vector<std::shared_ptr<arrow::Array>> scalar_cols;
  std::vector<std::shared_ptr<arrow::Array>> vector_cols;

  FileWriter* scalar_writer = nullptr;
  FileWriter* vector_writer = nullptr;

  std::vector<std::string> scalar_files;
  std::vector<std::string> vector_files;

  for (auto rec = reader->Next(); rec.ok(); rec = reader->Next()) {
    auto batch = rec.ValueOrDie();
    if (batch == nullptr) {
      break;
    }
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

    // add offset column
    std::vector<int64_t> offset_values(batch->num_rows());
    std::iota(offset_values.begin(), offset_values.end(), 0);
    arrow::NumericBuilder<arrow::Int64Type> builder;
    auto offset_col = builder.AppendValues(offset_values);
    scalar_cols.emplace_back(builder.Finish().ValueOrDie());

    auto scalar_record = arrow::RecordBatch::Make(scalar_schema, batch->num_rows(), scalar_cols);
    auto vector_record = arrow::RecordBatch::Make(vector_schema, batch->num_rows(), vector_cols);

    if (!scalar_writer) {
      auto scalar_file_path = GetNewParquetFilePath(manifest_->space_options()->uri);
      scalar_writer = new ParquetFileWriter(scalar_schema, fs_, scalar_file_path);
      RETURN_NOT_OK(scalar_writer->Init());

      auto vector_file_path = GetNewParquetFilePath(manifest_->space_options()->uri);
      vector_writer = new ParquetFileWriter(vector_schema, fs_, vector_file_path);
      RETURN_NOT_OK(scalar_writer->Init());

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

  manifest_->add_scalar_files(scalar_files);
  manifest_->add_vector_files(vector_files);
  RETURN_NOT_OK(SafeSaveManifest());
  return Status::OK();
}

Status DefaultSpace::Delete(arrow::RecordBatchReader* reader) {
  FileWriter* writer = nullptr;
  std::string delete_file;
  for (auto rec = reader->Next(); rec.ok(); rec = reader->Next()) {
    auto batch = rec.ValueOrDie();
    if (batch == nullptr) {
      break;
    }

    if (!writer) {
      delete_file = GetNewParquetFilePath(manifest_->space_options()->uri);
      writer = new ParquetFileWriter(schema_->delete_schema(), fs_, delete_file);
      RETURN_NOT_OK(writer->Init());
    }

    if (batch->num_rows() == 0) {
      continue;
    }

    writer->Write(batch.get());
    delete_set_->Add(batch);
  }

  if (!writer) {
    writer->Close();
    manifest_->add_delete_file(delete_file);
    RETURN_NOT_OK(SafeSaveManifest());
  }
}

std::unique_ptr<arrow::RecordBatchReader> DefaultSpace::Read(std::shared_ptr<ReadOptions> option) {
  return RecordReader::GetRecordReader(*this, option);
}

Status DefaultSpace::SafeSaveManifest() {
  auto tmp_manifest_file_path = GetManifestTmpFilePath(manifest_->space_options()->uri);
  auto manifest_file_path = GetManifestFilePath(manifest_->space_options()->uri);

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto output, fs_->OpenOutputStream(tmp_manifest_file_path));
  Manifest::WriteManifestFile(manifest_.get(), output.get());
  RETURN_ARROW_NOT_OK(output->Flush());
  RETURN_ARROW_NOT_OK(output->Close());

  RETURN_ARROW_NOT_OK(fs_->Move(tmp_manifest_file_path, manifest_file_path));
  RETURN_ARROW_NOT_OK(fs_->DeleteFile(tmp_manifest_file_path));
}
}  // namespace milvus_storage