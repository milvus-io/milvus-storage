
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/type_fwd.h>
#include <algorithm>
#include <atomic>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>

#include "arrow/array/builder_primitive.h"
#include "common/fs_util.h"
#include "common/macro.h"
#include "file/delete_fragment.h"
#include "filter/constant_filter.h"
#include "format/parquet/file_writer.h"
#include "storage/space.h"
#include "arrow/util/uri.h"
#include "common/utils.h"
#include "storage/manifest.h"
#include "reader/record_reader.h"
#include "common/status.h"
namespace milvus_storage {

Status Space::Init() {
  for (const auto& fragment : manifest_->delete_fragments()) {
    // FIXME: delete fragments may be copied many times, considering to change to smart pointer
    ASSIGN_OR_RETURN_NOT_OK(auto delete_fragment, DeleteFragment::Make(fs_, manifest_->schema(), fragment));
    delete_fragments_.push_back(delete_fragment);
  }
  return Status::OK();
}

Status Space::Write(arrow::RecordBatchReader* reader, WriteOption* option) {
  if (!reader->schema()->Equals(*this->manifest_->schema()->schema())) {
    return Status::InvalidArgument("Schema not match");
  }

  // remove duplicated codes
  auto scalar_schema = this->manifest_->schema()->scalar_schema(),
       vector_schema = this->manifest_->schema()->vector_schema();

  std::vector<std::shared_ptr<arrow::Array>> scalar_cols;
  std::vector<std::shared_ptr<arrow::Array>> vector_cols;

  FileWriter* scalar_writer = nullptr;
  FileWriter* vector_writer = nullptr;

  Fragment scalar_fragment;
  Fragment vector_fragment;

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

    // Only add offset column to scalar columns; vector columns not changed
    std::vector<int64_t> offset_values(batch->num_rows());
    std::iota(offset_values.begin(), offset_values.end(), 0);
    arrow::NumericBuilder<arrow::Int64Type> builder;
    RETURN_ARROW_NOT_OK(builder.AppendValues(offset_values));
    auto offset_col = builder.Finish().ValueOrDie();
    scalar_cols.emplace_back(offset_col);

    auto scalar_record = arrow::RecordBatch::Make(scalar_schema, batch->num_rows(), scalar_cols);
    auto vector_record = arrow::RecordBatch::Make(vector_schema, batch->num_rows(), vector_cols);

    if (scalar_writer == nullptr) {
      auto scalar_file_path = GetNewParquetFilePath(path_);
      scalar_writer = new ParquetFileWriter(scalar_schema, fs_, scalar_file_path);
      RETURN_NOT_OK(scalar_writer->Init());
      scalar_fragment.add_file(scalar_file_path);
    }

    if (vector_writer == nullptr) {
      auto vector_file_path = GetNewParquetFilePath(path_);
      vector_writer = new ParquetFileWriter(vector_schema, fs_, vector_file_path);
      RETURN_NOT_OK(vector_writer->Init());
      vector_fragment.add_file(vector_file_path);
    }

    RETURN_NOT_OK(scalar_writer->Write(scalar_record.get()));
    RETURN_NOT_OK(vector_writer->Write(vector_record.get()));

    if (scalar_writer->count() >= option->max_record_per_file) {
      scalar_writer->Close();
      vector_writer->Close();
      scalar_writer = nullptr;
      vector_writer = nullptr;
    }
  }

  if (scalar_writer != nullptr) {
    scalar_writer->Close();
    vector_writer->Close();
    scalar_writer = nullptr;
    vector_writer = nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto copied = new Manifest(*manifest_);
  auto old_version = manifest_->version();
  scalar_fragment.set_id(old_version + 1);
  vector_fragment.set_id(old_version + 1);
  copied->set_version(old_version + 1);
  copied->add_scalar_fragment(std::move(scalar_fragment));
  copied->add_vector_fragment(std::move(vector_fragment));
  RETURN_NOT_OK(SafeSaveManifest(fs_, path_, copied));
  manifest_.reset(copied);

  return Status::OK();
}

Status Space::Delete(arrow::RecordBatchReader* reader) {
  FileWriter* writer = nullptr;
  Fragment fragment;
  auto delete_fragment = std::make_shared<DeleteFragment>(fs_, manifest_->schema());
  std::string delete_file;
  for (auto rec = reader->Next(); rec.ok(); rec = reader->Next()) {
    auto batch = rec.ValueOrDie();
    if (batch == nullptr) {
      break;
    }

    if (!writer) {
      delete_file = GetNewParquetFilePath(path_);
      writer = new ParquetFileWriter(manifest_->schema()->delete_schema(), fs_, delete_file);
      RETURN_NOT_OK(writer->Init());
    }

    if (batch->num_rows() == 0) {
      continue;
    }

    writer->Write(batch.get());
    delete_fragment->Add(batch);
  }

  if (writer) {
    writer->Close();
    std::lock_guard<std::mutex> lock(mutex_);
    auto old_version = manifest_->version();
    auto copied = new Manifest(*manifest_);
    fragment.add_file(delete_file);
    fragment.set_id(old_version + 1);
    copied->set_version(old_version + 1);
    copied->add_delete_fragment(std::move(fragment));
    RETURN_NOT_OK(SafeSaveManifest(fs_, path_, copied));
    manifest_.reset(copied);
  }
  return Status::OK();
}

std::unique_ptr<arrow::RecordBatchReader> Space::Read(std::shared_ptr<ReadOptions> option) {
  if (option->has_version()) {
    assert(manifest_->schema()->options()->has_version_column());
    option->filters.push_back(std::make_unique<ConstantFilter>(
        ComparisonType::GREATER_EQUAL, manifest_->schema()->options()->version_column, option->version));
  }
  // TODO: remove second argument
  return RecordReader::MakeRecordReader(manifest_, manifest_->schema(), fs_, delete_fragments_, option);
}

Status Space::SafeSaveManifest(std::shared_ptr<arrow::fs::FileSystem> fs,
                               const std::string& path,
                               const Manifest* manifest) {
  auto tmp_manifest_file_path = GetManifestTmpFilePath(path, manifest->version());
  auto manifest_file_path = GetManifestFilePath(path, manifest->version());

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto output, fs->OpenOutputStream(tmp_manifest_file_path));
  Manifest::WriteManifestFile(manifest, output.get());
  RETURN_ARROW_NOT_OK(output->Flush());
  RETURN_ARROW_NOT_OK(output->Close());

  RETURN_ARROW_NOT_OK(fs->Move(tmp_manifest_file_path, manifest_file_path));
  return Status::OK();
}

Result<std::unique_ptr<Space>> Space::Open(const std::string& uri, Options options) {
  std::shared_ptr<arrow::fs::FileSystem> fs;
  std::shared_ptr<Manifest> manifest;
  std::string path;
  std::atomic_int64_t next_manifest_version = 1;

  ASSIGN_OR_RETURN_NOT_OK(fs, BuildFileSystem(uri));
  arrow::internal::Uri uri_parser;
  RETURN_ARROW_NOT_OK(uri_parser.Parse(uri));
  path = uri_parser.path();

  RETURN_ARROW_NOT_OK(fs->CreateDir(GetManifestDir(path)));

  ASSIGN_OR_RETURN_NOT_OK(auto files, FindAllManifest(fs, path));
  std::vector<arrow::fs::FileInfo> info_vec;
  std::copy_if(files.begin(), files.end(), std::back_inserter(info_vec),
               [](arrow::fs::FileInfo& f) { return ParseVersionFromFileName(f.base_name()) != -1; });

  std::cout << info_vec.size() << std::endl;
  if (info_vec.empty()) {
    // create the first manifest
    if (options.schema == nullptr) {
      return Status::InvalidArgument("schema should not be nullptr");
    }
    manifest = std::make_shared<Manifest>(options.schema);
    RETURN_NOT_OK(SafeSaveManifest(fs, path, manifest.get()));
  } else {
    arrow::fs::FileInfo file_info;
    if (options.version == -1) {
      // find latest manifest
      auto max_manifest =
          std::max_element(info_vec.begin(), info_vec.end(), [](arrow::fs::FileInfo& f1, arrow::fs::FileInfo& f2) {
            return ParseVersionFromFileName(f1.base_name()) < ParseVersionFromFileName(f2.base_name());
          });
      file_info = *max_manifest;
      next_manifest_version = ParseVersionFromFileName(file_info.base_name()) + 1;
    } else {
      auto iter = std::find_if(info_vec.begin(), info_vec.end(), [&](arrow::fs::FileInfo& f) {
        return ParseVersionFromFileName(f.base_name()) == options.version;
      });
      if (iter == info_vec.end()) {
        return Status::ManifestNotFound();
      }
      file_info = *iter;
      next_manifest_version = options.version + 1;
    }

    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto istream, fs->OpenInputStream(file_info));
    ASSIGN_OR_RETURN_NOT_OK(manifest, Manifest::ParseFromFile(istream, file_info));
  }

  auto space = std::make_unique<Space>();
  space->fs_ = fs;
  space->path_ = path;
  space->manifest_ = manifest;
  space->next_manifest_version_.store(next_manifest_version);

  RETURN_NOT_OK(space->Init());
  return space;
}

Result<arrow::fs::FileInfoVector> Space::FindAllManifest(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                         const std::string& path) {
  arrow::fs::FileSelector selector;
  selector.allow_not_found = true;
  selector.base_dir = path;

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto info_vec, fs->GetFileInfo(selector));
  return info_vec;
}

}  // namespace milvus_storage
