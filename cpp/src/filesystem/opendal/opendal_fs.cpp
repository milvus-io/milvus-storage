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

#ifdef MILVUS_OPENDAL
#include "filesystem/opendal/opendal_fs.h"
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/path_util.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/memory_pool.h>
#include <arrow/buffer.h>
#include <arrow/util/async_generator.h>
#include <arrow/util/logging.h>
#include <opendal.h>

namespace milvus_storage {

std::string ToString(opendal_bytes& bs) { return {reinterpret_cast<const char*>(bs.data), bs.len}; }

arrow::Result<OpendalOptions> OpendalOptions::FromUri(const arrow::internal::Uri& uri, std::string* out_path) {
  OpendalOptions options;
  const auto bucket = uri.host();
  auto path = uri.path();
  if (bucket.empty()) {
    if (!path.empty()) {
      return arrow::Status::Invalid("Missing bucket name in Opendal URI");
    }
  } else {
    if (path.empty()) {
      path = bucket;
    } else {
      if (path[0] != '/') {
        return arrow::Status::Invalid("Opendal URI should be absolute, not relative");
      }
      path = bucket + path;
    }
  }
  if (out_path != nullptr) {
    *out_path = std::string(arrow::fs::internal::RemoveTrailingSlash(path));
  }

  ARROW_ASSIGN_OR_RAISE(const auto options_items, uri.query_items());
  for (const auto& kv : options_items) {
    options.options_.emplace(kv.first, kv.second);
  }
  return options;
}

class OpendalInputFile : public arrow::io::RandomAccessFile {
  public:
  explicit OpendalInputFile(arrow::io::IOContext io_context, opendal_reader* reader, opendal_metadata* metadata)
      : io_context_(io_context), reader_(reader), metadata_(metadata) {
    content_length_ = opendal_metadata_content_length(metadata_);
  }

  arrow::Status Close() override {
    closed_ = true;
    return arrow::Status::OK();
  }

  bool closed() const override { return closed_; }

  arrow::Status CheckClosed() const {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed stream");
    }
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override {
    RETURN_NOT_OK(CheckClosed());
    return pos_;
  }

  arrow::Result<int64_t> GetSize() override {
    RETURN_NOT_OK(CheckClosed());
    return content_length_;
  }

  arrow::Status Seek(int64_t position) override {
    RETURN_NOT_OK(CheckClosed());
    pos_ = position;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    return arrow::Status::NotImplemented("Not implemented");
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    return arrow::Status::NotImplemented("Not implemented");
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    RETURN_NOT_OK(CheckClosed());
    nbytes = std::min(nbytes, content_length_ - pos_);
    if (nbytes == 0) {
      return 0;
    }

    auto result = opendal_reader_read(reader_, static_cast<uint8_t*>(out), nbytes);
    if (result.error != nullptr) {
      auto msg = "read failed: " + ToString(result.error->message);
      opendal_error_free(result.error);
      return arrow::Status::IOError(msg);
    }
    pos_ += result.size;
    return result.size;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    RETURN_NOT_OK(CheckClosed());
    nbytes = std::min(nbytes, content_length_ - pos_);

    ARROW_ASSIGN_OR_RAISE(auto buf, arrow::AllocateResizableBuffer(nbytes, io_context_.pool()));
    if (nbytes > 0) {
      ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, Read(nbytes, buf->mutable_data()));
      DCHECK_LE(bytes_read, nbytes);
      RETURN_NOT_OK(buf->Resize(bytes_read));
    }
    pos_ += buf->size();
    return std::move(buf);
  }

  private:
  const arrow::io::IOContext io_context_;
  opendal_reader* reader_;
  opendal_metadata* metadata_;

  bool closed_ = false;
  int64_t pos_ = 0;
  int64_t content_length_;
};

arrow::Result<std::shared_ptr<OpendalInputFile>> read(arrow::io::IOContext& io_context,
                                                      opendal_operator* op,
                                                      const std::string& path) {
  auto reader = opendal_operator_reader(op, path.c_str());
  if (reader.error != nullptr) {
    auto msg = "open reader failed: " + ToString(reader.error->message);
    opendal_error_free(reader.error);
    return arrow::Status::Invalid(msg);
  }
  auto stat = opendal_operator_stat(op, path.c_str());
  if (stat.error != nullptr) {
    auto msg = "open reader failed: " + ToString(reader.error->message);
    opendal_error_free(stat.error);
    opendal_reader_free(reader.reader);
    return arrow::Status::Invalid(msg);
  }
  auto file = std::make_shared<OpendalInputFile>(io_context, reader.reader, stat.meta);
  opendal_reader_free(reader.reader);
  return file;
}

class OpendalOutputStream : public arrow::io::OutputStream {};

arrow::Result<std::unique_ptr<OpendalFileSystem>> OpendalFileSystem::Make(const OpendalOptions& options,
                                                                          const arrow::io::IOContext& ctx) {
  auto fs = std::unique_ptr<OpendalFileSystem>(new OpendalFileSystem(options, ctx));
  opendal_operator_options* op_options_ = opendal_operator_options_new();
  for (auto& option : options.options()) {
    opendal_operator_options_set(op_options_, option.first.c_str(), option.second.c_str());
  }
  auto op = opendal_operator_new(options.at("scheme").c_str(), op_options_);
  if (op.error != nullptr) {
    auto msg = "open opendal operator failed: " + ToString(op.error->message);
    opendal_error_free(op.error);
    return arrow::Status::Invalid(msg);
  }
  fs->operator_ = op.op;
  opendal_operator_options_free(op_options_);
  return fs;
}

OpendalFileSystem::OpendalFileSystem(const OpendalOptions& options, const arrow::io::IOContext& io_context)
    : FileSystem(io_context), options_(options) {}

OpendalFileSystem::~OpendalFileSystem() {
  if (operator_ != nullptr) {
    opendal_operator_free(operator_);
  }
}

arrow::Result<arrow::fs::FileInfo> OpendalFileSystem::GetFileInfo(const std::string& path) {
  auto stat = opendal_operator_stat(operator_, path.c_str());
  if (stat.error != nullptr) {
    auto msg = "stat failed: " + ToString(stat.error->message);
    opendal_error_free(stat.error);
    return arrow::Status::Invalid(msg);
  }
  auto file_info = arrow::fs::FileInfo{};
  file_info.set_path(path);
  file_info.set_size(opendal_metadata_content_length(stat.meta));
  file_info.set_type(opendal_metadata_is_dir(stat.meta)    ? arrow::fs::FileType::Directory
                     : opendal_metadata_is_file(stat.meta) ? arrow::fs::FileType::File
                                                           : arrow::fs::FileType::Unknown);
  std::chrono::milliseconds mtime(opendal_metadata_last_modified_ms(stat.meta));
  file_info.set_mtime(arrow::fs::TimePoint(mtime));
  opendal_metadata_free(stat.meta);

  return file_info;
}

bool OpendalFileSystem::Equals(const FileSystem& other) const {
  if (this == &other) {
    return true;
  }
  if (other.type_name() != type_name()) {
    return false;
  }
  return options_.options() == static_cast<const OpendalFileSystem*>(&other)->options_.options();
}

arrow::Result<std::vector<arrow::fs::FileInfo>> OpendalFileSystem::GetFileInfo(const arrow::fs::FileSelector& select) {
  std::vector<arrow::fs::FileInfo> file_infos;
  auto lister = opendal_operator_list(operator_, select.base_dir.c_str());
  if (lister.error != nullptr) {
    auto msg = "list failed: " + ToString(lister.error->message);
    opendal_error_free(lister.error);
    return arrow::Status::Invalid(msg);
  }
  while (true) {
    auto entry = opendal_lister_next(lister.lister);
    if (entry.entry == nullptr) {
      break;
    }
    if (entry.error != nullptr) {
      auto msg = "list failed: " + ToString(entry.error->message);
      opendal_error_free(entry.error);
      return arrow::Status::Invalid(msg);
    }
    char* de_path = opendal_entry_path(entry.entry);
    ARROW_ASSIGN_OR_RAISE(auto info, GetFileInfo(de_path));
    file_infos.push_back(info);
    opendal_entry_free(entry.entry);
  }
  return file_infos;
}

arrow::Status OpendalFileSystem::CreateDir(const std::string& path, bool recursive) {
  auto error = opendal_operator_create_dir(operator_, path.c_str());
  if (error != nullptr) {
    auto msg = "create dir failed: " + ToString(error->message);
    opendal_error_free(error);
    return arrow::Status::Invalid(msg);
  }
  return arrow::Status::OK();
}

arrow::Status OpendalFileSystem::DeleteDir(const std::string& path) {
  auto error = opendal_operator_delete(operator_, path.c_str());
  if (error != nullptr) {
    auto msg = "delete dir failed: " + ToString(error->message);
    opendal_error_free(error);
    return arrow::Status::Invalid(msg);
  }
  return arrow::Status::OK();
}

arrow::Status OpendalFileSystem::DeleteDirContents(const std::string& path, bool missing_dir_ok) {
  return DeleteDir(path);
}

arrow::Status OpendalFileSystem::DeleteFile(const std::string& path) { return DeleteDir(path); }

arrow::Status OpendalFileSystem::Move(const std::string& src, const std::string& dest) {
  auto error = opendal_operator_rename(operator_, src.c_str(), dest.c_str());
  if (error != nullptr) {
    auto msg = "move failed: " + ToString(error->message);
    opendal_error_free(error);
    return arrow::Status::Invalid(msg);
  }
  return arrow::Status::OK();
}

arrow::Status OpendalFileSystem::CopyFile(const std::string& src, const std::string& dest) {
  auto error = opendal_operator_copy(operator_, src.c_str(), dest.c_str());
  if (error != nullptr) {
    auto msg = "copy failed: " + ToString(error->message);
    opendal_error_free(error);
    return arrow::Status::Invalid(msg);
  }
  return arrow::Status::OK();
}
arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpendalFileSystem::OpenInputStream(const std::string& path) {
  return read(io_context_, operator_, path);
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpendalFileSystem::OpenInputFile(const std::string& path) {
  return read(io_context_, operator_, path);
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpendalFileSystem::OpenOutputStream(
    const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) {
  return arrow::Status::NotImplemented("OpenOutputStream Not implemented");
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpendalFileSystem::OpenAppendStream(
    const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) {
  return arrow::Status::NotImplemented("OpendAppendStream Not implemented");
}

}  // namespace milvus_storage

#endif