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

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/filesystem/type_fwd.h>
#include <arrow/status.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>

#include "arrow/array/builder_primitive.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/file/delete_fragment.h"
#include "milvus-storage/format/parquet/file_writer.h"
#include "milvus-storage/storage/space.h"
#include "milvus-storage/common/utils.h"
#include "milvus-storage/storage/manifest.h"
#include "milvus-storage/reader/record_reader.h"
#include "milvus-storage/common/status.h"
namespace milvus_storage {

Status Space::Init() {
  for (const auto& fragment : manifest_->delete_fragments()) {
    // FIXME: delete fragments may be copied many times, considering to change to smart pointer
    ASSIGN_OR_RETURN_NOT_OK(auto delete_fragment, DeleteFragment::Make(*fs_, manifest_->schema(), fragment));
    delete_fragments_.push_back(delete_fragment);
  }
  return Status::OK();
}

Status Space::Write(arrow::RecordBatchReader& reader, const WriteOption& option) {
  if (!reader.schema()->Equals(*this->manifest_->schema()->schema())) {
    return Status::InvalidArgument("Schema not match");
  }

  // remove duplicated codes
  auto scalar_schema = this->manifest_->schema()->scalar_schema(),
       vector_schema = this->manifest_->schema()->vector_schema();

  std::vector<std::shared_ptr<arrow::Array>> scalar_cols;
  std::vector<std::shared_ptr<arrow::Array>> vector_cols;

  std::shared_ptr<FileWriter> scalar_writer;
  std::shared_ptr<FileWriter> vector_writer;

  Fragment scalar_fragment;
  Fragment vector_fragment;

  for (auto rec = reader.Next(); rec.ok(); rec = reader.Next()) {
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

    auto conf = StorageConfig();
    if (scalar_writer == nullptr) {
      auto scalar_file_path = GetNewParquetFilePath(GetScalarDataDir(path_));
      scalar_writer.reset(new ParquetFileWriter(scalar_schema, fs_, scalar_file_path, conf));
      RETURN_NOT_OK(scalar_writer->Init());
      scalar_fragment.add_file(scalar_file_path);
    }

    if (vector_writer == nullptr) {
      auto vector_file_path = GetNewParquetFilePath(GetVectorDataDir(path_));
      vector_writer.reset(new ParquetFileWriter(vector_schema, fs_, vector_file_path, conf));
      RETURN_NOT_OK(vector_writer->Init());
      vector_fragment.add_file(vector_file_path);
    }

    RETURN_NOT_OK(scalar_writer->Write(*scalar_record));
    RETURN_NOT_OK(vector_writer->Write(*vector_record));

    if (scalar_writer->count() >= option.max_record_per_file) {
      scalar_writer->Close();
      vector_writer->Close();
      scalar_writer.reset();
      vector_writer.reset();
    }
  }

  if (scalar_writer != nullptr) {
    scalar_writer->Close();
    vector_writer->Close();
    scalar_writer.reset();
    vector_writer.reset();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto copied = new Manifest(*manifest_);
  auto next_version = next_manifest_version_++;
  scalar_fragment.set_id(next_version);
  vector_fragment.set_id(next_version);
  copied->set_version(next_version);
  copied->add_scalar_fragment(std::move(scalar_fragment));
  copied->add_vector_fragment(std::move(vector_fragment));
  RETURN_NOT_OK(SafeSaveManifest(*fs_, path_, *copied));
  manifest_.reset(copied);

  return Status::OK();
}

Status Space::Delete(arrow::RecordBatchReader& reader) {
  FileWriter* writer = nullptr;
  Fragment fragment;
  auto delete_fragment = std::make_shared<DeleteFragment>(*fs_, manifest_->schema());
  std::string delete_file;
  for (auto rec = reader.Next(); rec.ok(); rec = reader.Next()) {
    auto batch = rec.ValueOrDie();
    if (batch == nullptr) {
      break;
    }

    if (!writer) {
      delete_file = GetNewParquetFilePath(GetDeleteDataDir(path_));
      auto conf = StorageConfig();
      writer = new ParquetFileWriter(manifest_->schema()->delete_schema(), fs_, delete_file, conf);
      RETURN_NOT_OK(writer->Init());
    }

    if (batch->num_rows() == 0) {
      continue;
    }

    writer->Write(*batch);
    delete_fragment->Add(batch);
  }

  if (writer) {
    writer->Close();
    std::lock_guard<std::mutex> lock(mutex_);
    auto next_version = next_manifest_version_++;
    auto copied = new Manifest(*manifest_);
    fragment.add_file(delete_file);
    fragment.set_id(next_version);
    copied->set_version(next_version);
    copied->add_delete_fragment(std::move(fragment));
    RETURN_NOT_OK(SafeSaveManifest(*fs_, path_, *copied));
    manifest_.reset(copied);
  }
  return Status::OK();
}

std::unique_ptr<arrow::RecordBatchReader> Space::Read(const ReadOptions& option) const {
  // TODO: remove second argument
  return internal::MakeRecordReader(manifest_, manifest_->schema(), *fs_, delete_fragments_, option);
}

Status Space::WriteBlob(const std::string& name, const void* blob, int64_t length, bool replace) {
  if (!replace && manifest_->has_blob(name)) {
    return Status::InvalidArgument("blob already exist");
  }

  std::string blob_file_path = GetNewBlobFilePath(path_);
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto output, fs_->OpenOutputStream(blob_file_path));
  RETURN_ARROW_NOT_OK(output->Write(blob, length));
  RETURN_ARROW_NOT_OK(output->Close());

  std::lock_guard<std::mutex> lock(mutex_);
  auto next_version = next_manifest_version_++;
  auto copied = new Manifest(*manifest_);
  copied->set_version(next_version);
  copied->remove_blob_if_exist(name);
  copied->add_blob({name, length, blob_file_path});
  RETURN_NOT_OK(SafeSaveManifest(*fs_, path_, *copied));
  manifest_.reset(copied);
  return Status::OK();
}

Status Space::ReadBlob(const std::string& name, void* target) const {
  auto manifest = manifest_;
  ASSIGN_OR_RETURN_NOT_OK(auto blob, manifest->get_blob(name));
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto file, fs_->OpenInputFile(blob.file));
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto _, file->Read(blob.size, target));
  return Status::OK();
}

Result<int64_t> Space::GetBlobByteSize(const std::string& name) const {
  auto manifest = manifest_;
  ASSIGN_OR_RETURN_NOT_OK(auto blob, manifest->get_blob(name));
  return blob.size;
}

Status Space::SafeSaveManifest(arrow::fs::FileSystem& fs, const std::string& path, const Manifest& manifest) {
  auto tmp_manifest_file_path = GetManifestTmpFilePath(path, manifest.version());
  auto manifest_file_path = GetManifestFilePath(path, manifest.version());

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto output, fs.OpenOutputStream(tmp_manifest_file_path));
  Manifest::WriteManifestFile(manifest, *output);
  RETURN_ARROW_NOT_OK(output->Flush());
  RETURN_ARROW_NOT_OK(output->Close());

  RETURN_ARROW_NOT_OK(fs.Move(tmp_manifest_file_path, manifest_file_path));
  return Status::OK();
}

Result<std::unique_ptr<Space>> Space::Open(const std::string& root_path, const Options& options) {
  std::shared_ptr<Manifest> manifest;
  std::string path;
  std::atomic_int64_t next_manifest_version = 1;

  auto conf = ArrowFileSystemConfig();
  conf.storage_type = "local";
  conf.root_path = root_path;
  arrow::util::Uri uri_parser;
  auto uri = "file://" + root_path;
  RETURN_ARROW_NOT_OK(uri_parser.Parse(uri));
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::LocalFileSystemOptions::FromUri(uri_parser, &path))

  ArrowFileSystemSingleton::GetInstance().Init(conf);
  ArrowFileSystemPtr fs = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();

  LOG_STORAGE_INFO_ << "Open space: " << path;
  RETURN_ARROW_NOT_OK(fs->CreateDir(GetManifestDir(path)));
  RETURN_ARROW_NOT_OK(fs->CreateDir(GetScalarDataDir(path)));
  RETURN_ARROW_NOT_OK(fs->CreateDir(GetVectorDataDir(path)));
  RETURN_ARROW_NOT_OK(fs->CreateDir(GetDeleteDataDir(path)));
  RETURN_ARROW_NOT_OK(fs->CreateDir(GetBlobDir(path)));

  ASSIGN_OR_RETURN_NOT_OK(auto info_vec, FindAllManifest(*fs, path));
  if (info_vec.empty()) {
    // create the first manifest
    if (options.schema == nullptr) {
      return Status::InvalidArgument("schema should not be nullptr");
    }

    RETURN_NOT_OK(options.schema->Validate());
    manifest = std::make_shared<Manifest>(options.schema);
    RETURN_NOT_OK(SafeSaveManifest(*fs, path, *manifest));
  } else {
    arrow::fs::FileInfo file_info;
    auto max_cmp = [](arrow::fs::FileInfo& f1, arrow::fs::FileInfo& f2) {
      return ParseVersionFromFileName(f1.base_name()) < ParseVersionFromFileName(f2.base_name());
    };
    auto latest = std::max_element(info_vec.begin(), info_vec.end(), max_cmp);
    if (options.version == -1) {
      // find latest manifest
      file_info = *latest;
    } else {
      auto iter = std::find_if(info_vec.begin(), info_vec.end(), [&](arrow::fs::FileInfo& f) {
        return ParseVersionFromFileName(f.base_name()) == options.version;
      });
      if (iter == info_vec.end()) {
        return Status::FileNotFound();
      }
      file_info = *iter;
    }
    next_manifest_version = ParseVersionFromFileName(file_info.base_name()) + 1;

    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto istream, fs->OpenInputStream(file_info));
    ASSIGN_OR_RETURN_NOT_OK(manifest, Manifest::ParseFromFile(istream, file_info));
  }

  auto space = std::make_unique<Space>();
  space->fs_ = std::move(fs);
  space->path_ = path;
  space->manifest_ = manifest;
  space->next_manifest_version_ = next_manifest_version;

  RETURN_NOT_OK(space->Init());
  ArrowFileSystemSingleton::GetInstance().Release();
  return space;
}

Result<arrow::fs::FileInfoVector> Space::FindAllManifest(arrow::fs::FileSystem& fs, const std::string& path) {
  arrow::fs::FileSelector selector;
  selector.allow_not_found = true;
  selector.base_dir = GetManifestDir(path);

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto files, fs.GetFileInfo(selector));
  std::vector<arrow::fs::FileInfo> info_vec;
  std::copy_if(files.begin(), files.end(), std::back_inserter(info_vec),
               [](arrow::fs::FileInfo& f) { return ParseVersionFromFileName(f.base_name()) != -1; });
  return info_vec;
}

const std::vector<Blob>& Space::StatisticsBlobs() const { return manifest_->blobs(); }

std::unique_ptr<arrow::RecordBatchReader> Space::ScanDelete() const {
  return internal::MakeScanDeleteReader(manifest_, *fs_);
}

std::unique_ptr<arrow::RecordBatchReader> Space::ScanData(const std::set<std::string>& columns) const {
  return internal::MakeScanDataReader(manifest_, *fs_, ReadOptions{.columns = columns});
}

std::shared_ptr<Schema> Space::schema() const { return manifest_->schema(); }

int64_t Space::GetCurrentVersion() const { return manifest_->version(); }

}  // namespace milvus_storage
