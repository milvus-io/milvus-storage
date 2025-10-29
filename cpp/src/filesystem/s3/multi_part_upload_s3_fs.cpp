// Copyright 2024 Zilliz
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

#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"
#include "milvus-storage/filesystem/s3/util_internal.h"

#include "milvus-storage/common/path_util.h"
#include "milvus-storage/filesystem/io/io_util.h"
#include "milvus-storage/filesystem/s3/s3_client.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/logging.h"
#include "arrow/buffer.h"
#include "arrow/result.h"
#include "arrow/io/memory.h"
#include "arrow/util/future.h"
#include "arrow/util/thread_pool.h"
#include "arrow/filesystem/path_util.h"
#include "arrow/io/interfaces.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/filesystem/type_fwd.h"
#include "arrow/util/string.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>

#include <aws/core/Aws.h>
#include <aws/core/Region.h>
#include <aws/core/VersionConfig.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/auth/STSCredentialsProvider.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/client/RetryStrategy.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/core/utils/xml/XmlSerializer.h>
#include <aws/identity-management/auth/STSAssumeRoleCredentialsProvider.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/S3Errors.h>
#include <aws/s3/model/AbortMultipartUploadRequest.h>
#include <aws/s3/model/CompleteMultipartUploadRequest.h>
#include <aws/s3/model/CompletedMultipartUpload.h>
#include <aws/s3/model/CompletedPart.h>
#include <aws/s3/model/CopyObjectRequest.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/CreateMultipartUploadRequest.h>
#include <aws/s3/model/DeleteBucketRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/DeleteObjectsRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListBucketsResult.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/ObjectCannedACL.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/UploadPartRequest.h>

static constexpr const char kSep = '/';

using ::arrow::Buffer;
using ::arrow::Future;
using ::arrow::Result;
using ::arrow::Status;
using ::arrow::fs::FileInfo;
using ::arrow::fs::FileInfoGenerator;
using ::arrow::fs::FileInfoVector;
using ::arrow::fs::FileSelector;
using ::arrow::fs::FileType;
using ::arrow::fs::kNoSize;
using ::arrow::fs::S3FileSystem;
using ::arrow::fs::S3LogLevel;
using ::arrow::fs::S3Options;
using ::arrow::fs::S3RetryStrategy;
using ::arrow::fs::internal::DetectS3Backend;
using ::arrow::fs::internal::ErrorToStatus;
using ::arrow::fs::internal::FromAwsDatetime;
using ::arrow::fs::internal::FromAwsString;
using ::arrow::fs::internal::IsAlreadyExists;
using ::arrow::fs::internal::IsNotFound;
using ::arrow::fs::internal::OutcomeToResult;
using ::arrow::fs::internal::OutcomeToStatus;
using ::arrow::fs::internal::RemoveTrailingSlash;
using ::arrow::fs::internal::S3Backend;
using ::arrow::fs::internal::ToAwsString;
using ::Aws::Client::AWSError;
using ::Aws::S3::S3Errors;

namespace S3Model = Aws::S3::Model;

namespace milvus_storage {

// -----------------------------------------------------------------------
// ExtendedS3Options implementation

ExtendedS3Options::ExtendedS3Options() {
  DCHECK(IsS3Initialized()) << "Must initialize S3 before using ExtendedS3Options";
}

// -----------------------------------------------------------------------
// MultiPartUploadS3FS implementation

static constexpr const char kAwsEndpointUrlEnvVar[] = "AWS_ENDPOINT_URL";
static constexpr const char kAwsEndpointUrlS3EnvVar[] = "AWS_ENDPOINT_URL_S3";
static constexpr const char kAwsDirectoryContentType[] = "application/x-directory";

bool IsDirectory(std::string_view key, const S3Model::HeadObjectResult& result) {
  // If it has a non-zero length, it's a regular file. We do this even if
  // the key has a trailing slash, as directory markers should never have
  // any data associated to them.
  if (result.GetContentLength() > 0) {
    return false;
  }
  // Otherwise, if it has a trailing slash, it's a directory
  if (arrow::fs::internal::HasTrailingSlash(key)) {
    return true;
  }
  // Otherwise, if its content type starts with "application/x-directory",
  // it's a directory
  if (::arrow::internal::StartsWith(result.GetContentType(), kAwsDirectoryContentType)) {
    return true;
  }
  // Otherwise, it's a regular file.
  return false;
}

template <typename ObjectRequest>
struct ObjectMetadataSetter {
  using Setter = std::function<Status(const std::string& value, ObjectRequest* req)>;

  static std::unordered_map<std::string, Setter> GetSetters() {
    return {{"ACL", CannedACLSetter()},
            {"Cache-Control", StringSetter(&ObjectRequest::SetCacheControl)},
            {"Content-Type", ContentTypeSetter()},
            {"Content-Language", StringSetter(&ObjectRequest::SetContentLanguage)},
            {"Expires", DateTimeSetter(&ObjectRequest::SetExpires)}};
  }

  private:
  static Setter StringSetter(void (ObjectRequest::*req_method)(Aws::String&&)) {
    return [req_method](const std::string& v, ObjectRequest* req) {
      (req->*req_method)(ToAwsString(v));
      return arrow::Status::OK();
    };
  }

  static Setter DateTimeSetter(void (ObjectRequest::*req_method)(Aws::Utils::DateTime&&)) {
    return [req_method](const std::string& v, ObjectRequest* req) {
      (req->*req_method)(Aws::Utils::DateTime(v.data(), Aws::Utils::DateFormat::ISO_8601));
      return arrow::Status::OK();
    };
  }

  static Setter CannedACLSetter() {
    return [](const std::string& v, ObjectRequest* req) {
      ARROW_ASSIGN_OR_RAISE(auto acl, ParseACL(v));
      req->SetACL(acl);
      return arrow::Status::OK();
    };
  }

  /** We need a special setter here and can not use `StringSetter` because for e.g. the
   * `PutObjectRequest`, the setter is located in the base class (instead of the concrete
   * class). */
  static Setter ContentTypeSetter() {
    return [](const std::string& str, ObjectRequest* req) {
      req->SetContentType(str);
      return arrow::Status::OK();
    };
  }

  static arrow::Result<S3Model::ObjectCannedACL> ParseACL(const std::string& v) {
    if (v.empty()) {
      return S3Model::ObjectCannedACL::NOT_SET;
    }
    auto acl = S3Model::ObjectCannedACLMapper::GetObjectCannedACLForName(ToAwsString(v));
    if (acl == S3Model::ObjectCannedACL::NOT_SET) {
      // XXX This actually never happens, as the AWS SDK dynamically
      // expands the enum range using Aws::GetEnumOverflowContainer()
      return arrow::Status::Invalid("Invalid S3 canned ACL: '", v, "'");
    }
    return acl;
  }
};

struct S3Path {
  std::string full_path;
  std::string bucket;
  std::string key;
  std::vector<std::string> key_parts;

  static arrow::Result<S3Path> FromString(const std::string& s) {
    if (arrow::fs::internal::IsLikelyUri(s)) {
      return arrow::Status::Invalid("Expected an S3 object path of the form 'bucket/key...', got a URI: '", s, "'");
    }
    const auto src = RemoveTrailingSlash(s);
    auto first_sep = src.find_first_of(kSep);
    if (first_sep == 0) {
      return arrow::Status::Invalid("Path cannot start with a separator ('", s, "')");
    }
    if (first_sep == std::string::npos) {
      return S3Path{std::string(src), std::string(src), "", {}};
    }
    S3Path path;
    path.full_path = std::string(src);
    path.bucket = std::string(src.substr(0, first_sep));
    path.key = std::string(src.substr(first_sep + 1));
    path.key_parts = arrow::fs::internal::SplitAbstractPath(path.key);
    ARROW_RETURN_NOT_OK(Validate(path));
    return path;
  }

  static arrow::Status Validate(const S3Path& path) {
    auto st = arrow::fs::internal::ValidateAbstractPath(path.full_path);
    if (!st.ok()) {
      return arrow::Status::Invalid(st.message(), " in path ", path.full_path);
    }
    return arrow::Status::OK();
  }

  Aws::String ToAwsString() const {
    Aws::String res(bucket.begin(), bucket.end());
    res.reserve(bucket.size() + key.size() + 1);
    res += kSep;
    res.append(key.begin(), key.end());
    return res;
  }

  S3Path parent() const {
    DCHECK(!key_parts.empty());
    auto parent = S3Path{"", bucket, "", key_parts};
    parent.key_parts.pop_back();
    parent.key = arrow::fs::internal::JoinAbstractPath(parent.key_parts);
    parent.full_path = parent.bucket + kSep + parent.key;
    return parent;
  }

  bool has_parent() const { return !key.empty(); }

  bool empty() const { return bucket.empty() && key.empty(); }

  bool operator==(const S3Path& other) const { return bucket == other.bucket && key == other.key; }
};

arrow::Status PathNotFound(const S3Path& path) { return ::arrow::fs::internal::PathNotFound(path.full_path); }

arrow::Status PathNotFound(const std::string& bucket, const std::string& key) {
  return ::arrow::fs::internal::PathNotFound(bucket + kSep + key);
}

arrow::Status NotAFile(const S3Path& path) { return NotAFile(path.full_path); }

arrow::Status ValidateFilePath(const S3Path& path) {
  if (path.bucket.empty() || path.key.empty()) {
    return NotAFile(path);
  }
  return arrow::Status::OK();
};

arrow::Status CheckS3Initialized() {
  if (!IsS3Initialized()) {
    if (IsS3Finalized()) {
      return arrow::Status::Invalid("S3 subsystem is finalized");
    }
    return arrow::Status::Invalid(
        "S3 subsystem is not initialized; please call InitializeS3() "
        "before carrying out any S3-related operation");
  }
  return arrow::Status::OK();
};

template <typename ObjectRequest>
arrow::Status SetObjectMetadata(const std::shared_ptr<const arrow::KeyValueMetadata>& metadata, ObjectRequest* req) {
  static auto setters = ObjectMetadataSetter<ObjectRequest>::GetSetters();

  DCHECK(metadata != nullptr);
  const auto& keys = metadata->keys();
  const auto& values = metadata->values();

  for (size_t i = 0; i < keys.size(); ++i) {
    auto it = setters.find(keys[i]);
    if (it != setters.end()) {
      RETURN_NOT_OK(it->second(values[i], req));
    }
  }
  return arrow::Status::OK();
}

class StringViewStream : Aws::Utils::Stream::PreallocatedStreamBuf, public std::iostream {
  public:
  StringViewStream(const void* data, int64_t nbytes)
      : Aws::Utils::Stream::PreallocatedStreamBuf(reinterpret_cast<unsigned char*>(const_cast<void*>(data)),
                                                  static_cast<size_t>(nbytes)),
        std::iostream(this) {}
};

std::string FormatRange(int64_t start, int64_t length) {
  // Format a HTTP range header value
  std::stringstream ss;
  ss << "bytes=" << start << "-" << start + length - 1;
  return ss.str();
}

Aws::IOStreamFactory AwsWriteableStreamFactory(void* data, int64_t nbytes) {
  return [=]() { return Aws::New<StringViewStream>("", data, nbytes); };
}

arrow::Result<S3Model::GetObjectResult> GetObjectRange(
    Aws::S3::S3Client* client, const S3Path& path, int64_t start, int64_t length, void* out) {
  S3Model::GetObjectRequest req;
  req.SetBucket(ToAwsString(path.bucket));
  req.SetKey(ToAwsString(path.key));
  req.SetRange(ToAwsString(FormatRange(start, length)));
  req.SetResponseStreamFactory(AwsWriteableStreamFactory(out, length));
  return OutcomeToResult("GetObject", client->GetObject(req));
}

template <typename ObjectResult>
std::shared_ptr<const arrow::KeyValueMetadata> GetObjectMetadata(const ObjectResult& result) {
  auto md = std::make_shared<arrow::KeyValueMetadata>();

  auto push = [&](std::string k, const Aws::String& v) {
    if (!v.empty()) {
      md->Append(std::move(k), std::string(FromAwsString(v)));
    }
  };
  auto push_datetime = [&](std::string k, const Aws::Utils::DateTime& v) {
    if (v != Aws::Utils::DateTime(0.0)) {
      push(std::move(k), v.ToGmtString(Aws::Utils::DateFormat::ISO_8601));
    }
  };

  md->Append("Content-Length", ToChars(result.GetContentLength()));
  push("Cache-Control", result.GetCacheControl());
  push("Content-Type", result.GetContentType());
  push("Content-Language", result.GetContentLanguage());
  push("ETag", result.GetETag());
  push("VersionId", result.GetVersionId());
  push_datetime("Last-Modified", result.GetLastModified());
  push_datetime("Expires", result.GetExpires());
  // NOTE the "canned ACL" isn't available for reading (one can get an expanded
  // ACL using a separate GetObjectAcl request)
  return md;
}

class ObjectInputFile final : public arrow::io::RandomAccessFile {
  public:
  ObjectInputFile(std::shared_ptr<S3ClientHolder> holder,
                  const arrow::io::IOContext& io_context,
                  const S3Path& path,
                  int64_t size = kNoSize)
      : holder_(std::move(holder)), io_context_(io_context), path_(path), content_length_(size) {}

  arrow::Status Init() {
    // Issue a HEAD Object to get the content-length and ensure any
    // errors (e.g. file not found) don't wait until the first Read() call.
    if (content_length_ != kNoSize) {
      DCHECK_GE(content_length_, 0);
      return arrow::Status::OK();
    }

    S3Model::HeadObjectRequest req;
    req.SetBucket(ToAwsString(path_.bucket));
    req.SetKey(ToAwsString(path_.key));

    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());
    auto outcome = client_lock.Move()->HeadObject(req);
    if (!outcome.IsSuccess()) {
      if (IsNotFound(outcome.GetError())) {
        return PathNotFound(path_);
      } else {
        return ErrorToStatus(std::forward_as_tuple("When reading information for key '", path_.key, "' in bucket '",
                                                   path_.bucket, "': "),
                             "HeadObject", outcome.GetError());
      }
    }
    content_length_ = outcome.GetResult().GetContentLength();
    DCHECK_GE(content_length_, 0);
    metadata_ = GetObjectMetadata(outcome.GetResult());
    return arrow::Status::OK();
  }

  arrow::Status CheckClosed() const {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed stream");
    }
    return arrow::Status::OK();
  }

  arrow::Status CheckPosition(int64_t position, const char* action) const {
    if (position < 0) {
      return arrow::Status::Invalid("Cannot ", action, " from negative position");
    }
    if (position > content_length_) {
      return arrow::Status::IOError("Cannot ", action, " past end of file");
    }
    return arrow::Status::OK();
  }

  // RandomAccessFile APIs

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override { return metadata_; }

  Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    return metadata_;
  }

  arrow::Status Close() override {
    holder_ = nullptr;
    closed_ = true;
    return arrow::Status::OK();
  }

  bool closed() const override { return closed_; }

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
    RETURN_NOT_OK(CheckPosition(position, "seek"));

    pos_ = position;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    RETURN_NOT_OK(CheckClosed());
    RETURN_NOT_OK(CheckPosition(position, "read"));

    nbytes = std::min(nbytes, content_length_ - position);
    if (nbytes == 0) {
      return 0;
    }

    // Read the desired range of bytes
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());
    ARROW_ASSIGN_OR_RAISE(S3Model::GetObjectResult result,
                          GetObjectRange(client_lock.get(), path_, position, nbytes, out));

    auto& stream = result.GetBody();
    stream.ignore(nbytes);
    // NOTE: the stream is a stringstream by default, there is no actual error
    // to check for.  However, stream.fail() may return true if EOF is reached.
    return stream.gcount();
  }

  arrow::Result<std::shared_ptr<Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    RETURN_NOT_OK(CheckClosed());
    RETURN_NOT_OK(CheckPosition(position, "read"));

    // No need to allocate more than the remaining number of bytes
    nbytes = std::min(nbytes, content_length_ - position);

    ARROW_ASSIGN_OR_RAISE(auto buf, AllocateResizableBuffer(nbytes, io_context_.pool()));
    if (nbytes > 0) {
      ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(position, nbytes, buf->mutable_data()));
      DCHECK_LE(bytes_read, nbytes);
      RETURN_NOT_OK(buf->Resize(bytes_read));
    }
    // R build with openSUSE155 requires an explicit shared_ptr construction
    return std::shared_ptr<Buffer>(std::move(buf));
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(pos_, nbytes, out));
    pos_ += bytes_read;
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
    pos_ += buffer->size();
    return buffer;
  }

  protected:
  std::shared_ptr<S3ClientHolder> holder_;
  const arrow::io::IOContext io_context_;
  S3Path path_;

  bool closed_ = false;
  int64_t pos_ = 0;
  int64_t content_length_ = kNoSize;
  std::shared_ptr<const arrow::KeyValueMetadata> metadata_;
};

void FileObjectToInfo(std::string_view key, const S3Model::HeadObjectResult& obj, FileInfo* info) {
  if (IsDirectory(key, obj)) {
    info->set_type(FileType::Directory);
  } else {
    info->set_type(FileType::File);
  }
  info->set_size(static_cast<int64_t>(obj.GetContentLength()));
  info->set_mtime(FromAwsDatetime(obj.GetLastModified()));
}

void FileObjectToInfo(const S3Model::Object& obj, FileInfo* info) {
  info->set_type(arrow::fs::FileType::File);
  info->set_size(static_cast<int64_t>(obj.GetSize()));
  info->set_mtime(FromAwsDatetime(obj.GetLastModified()));
}

class CustomOutputStream final : public arrow::io::OutputStream {
  protected:
  struct UploadState;

  public:
  CustomOutputStream(std::shared_ptr<S3ClientHolder> holder,
                     const arrow::io::IOContext& io_context,
                     const S3Path& path,
                     const S3Options& options,
                     const std::shared_ptr<const arrow::KeyValueMetadata>& metadata,
                     const int64_t part_size)
      : holder_(std::move(holder)),
        io_context_(io_context),
        path_(path),
        metadata_(metadata),
        default_metadata_(options.default_metadata),
        background_writes_(options.background_writes),
        part_upload_size_(part_size),
        allow_delayed_open_(false) {}

  ~CustomOutputStream() override {
    // For compliance with the rest of the IO stack, Close rather than Abort,
    // even though it may be more expensive.
    CloseFromDestructor(this);
  }

  template <typename ObjectRequest>
  arrow::Status SetMetadataInRequest(ObjectRequest* request) {
    std::shared_ptr<const arrow::KeyValueMetadata> metadata;

    if (metadata_ && metadata_->size() != 0) {
      metadata = metadata_;
    } else if (default_metadata_ && default_metadata_->size() != 0) {
      metadata = default_metadata_;
    }

    bool is_content_type_set{false};
    if (metadata) {
      RETURN_NOT_OK(SetObjectMetadata(metadata, request));

      is_content_type_set = metadata->Contains("Content-Type");
    }

    if (!is_content_type_set) {
      // If we do not set anything then the SDK will default to application/xml
      // which confuses some tools (https://github.com/apache/arrow/issues/11934)
      // So we instead default to application/octet-stream which is less misleading
      request->SetContentType("application/octet-stream");
    }

    return arrow::Status::OK();
  }

  std::shared_ptr<CustomOutputStream> Self() {
    return std::dynamic_pointer_cast<CustomOutputStream>(shared_from_this());
  }

  arrow::Status CreateMultipartUpload() {
    DCHECK(ShouldBeMultipartUpload());

    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    // Initiate the multi-part upload
    S3Model::CreateMultipartUploadRequest req;
    req.SetBucket(ToAwsString(path_.bucket));
    req.SetKey(ToAwsString(path_.key));
    RETURN_NOT_OK(SetMetadataInRequest(&req));

    auto outcome = client_lock.Move()->CreateMultipartUpload(req);
    if (!outcome.IsSuccess()) {
      return ErrorToStatus(std::forward_as_tuple("When initiating multiple part upload for key '", path_.key,
                                                 "' in bucket '", path_.bucket, "': "),
                           "CreateMultipartUpload", outcome.GetError());
    }
    multipart_upload_id_ = outcome.GetResult().GetUploadId();

    return arrow::Status::OK();
  }

  arrow::Status Init() {
    // If we are allowed to do delayed I/O, we can use a single request to upload the
    // data. If not, we use a multi-part upload and initiate it here to
    // sanitize that writing to the bucket is possible.
    if (!allow_delayed_open_) {
      RETURN_NOT_OK(CreateMultipartUpload());
    }

    upload_state_ = std::make_shared<UploadState>();
    closed_ = false;
    return arrow::Status::OK();
  }

  arrow::Status Abort() override {
    if (closed_) {
      return arrow::Status::OK();
    }

    if (IsMultipartCreated()) {
      ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

      S3Model::AbortMultipartUploadRequest req;
      req.SetBucket(ToAwsString(path_.bucket));
      req.SetKey(ToAwsString(path_.key));
      req.SetUploadId(multipart_upload_id_);

      auto outcome = client_lock.Move()->AbortMultipartUpload(req);
      if (!outcome.IsSuccess()) {
        return ErrorToStatus(std::forward_as_tuple("When aborting multiple part upload for key '", path_.key,
                                                   "' in bucket '", path_.bucket, "': "),
                             "AbortMultipartUpload", outcome.GetError());
      }
    }

    current_part_.reset();
    holder_ = nullptr;
    closed_ = true;

    return arrow::Status::OK();
  }

  // OutputStream interface

  bool ShouldBeMultipartUpload() const { return pos_ > part_upload_size_ - 1 || !allow_delayed_open_; }

  bool IsMultipartCreated() const { return !multipart_upload_id_.empty(); }

  arrow::Status EnsureReadyToFlushFromClose() {
    if (ShouldBeMultipartUpload()) {
      if (current_part_) {
        // Upload last part
        RETURN_NOT_OK(CommitCurrentPart());
      }

      // S3 mandates at least one part, upload an empty one if necessary
      if (part_number_ == 1) {
        RETURN_NOT_OK(UploadPart("", 0));
      }
    } else {
      RETURN_NOT_OK(UploadUsingSingleRequest());
    }

    return arrow::Status::OK();
  }

  arrow::Status CleanupAfterClose() {
    holder_ = nullptr;
    closed_ = true;
    return arrow::Status::OK();
  }

  arrow::Status FinishPartUploadAfterFlush() {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    // At this point, all part uploads have finished successfully
    DCHECK_GT(part_number_, 1);
    DCHECK_EQ(upload_state_->completed_parts.size(), static_cast<size_t>(part_number_ - 1));

    S3Model::CompletedMultipartUpload completed_upload;
    completed_upload.SetParts(upload_state_->completed_parts);
    S3Model::CompleteMultipartUploadRequest req;
    req.SetBucket(ToAwsString(path_.bucket));
    req.SetKey(ToAwsString(path_.key));
    req.SetUploadId(multipart_upload_id_);
    req.SetMultipartUpload(std::move(completed_upload));

    auto outcome = client_lock.Move()->CompleteMultipartUploadWithErrorFixup(std::move(req));
    if (!outcome.IsSuccess()) {
      return ErrorToStatus(std::forward_as_tuple("When completing multiple part upload for key '", path_.key,
                                                 "' in bucket '", path_.bucket, "': "),
                           "CompleteMultipartUpload", outcome.GetError());
    }

    return arrow::Status::OK();
  }

  arrow::Status CleanupIfFailed(Status status) {
    if (!status.ok()) {
      RETURN_NOT_OK(CleanupAfterClose());
      return status;
    }
    return arrow::Status::OK();
  }

  arrow::Status Close() override {
    if (closed_)
      return arrow::Status::OK();

    RETURN_NOT_OK(CleanupIfFailed(EnsureReadyToFlushFromClose()));

    RETURN_NOT_OK(CleanupIfFailed(Flush()));

    if (IsMultipartCreated()) {
      RETURN_NOT_OK(CleanupIfFailed(FinishPartUploadAfterFlush()));
    }

    return CleanupAfterClose();
  }

  Future<> CloseAsync() override {
    if (closed_)
      return arrow::Status::OK();

    RETURN_NOT_OK(CleanupIfFailed(EnsureReadyToFlushFromClose()));

    // Wait for in-progress uploads to finish (if async writes are enabled)
    return FlushAsync().Then([self = Self()]() {
      if (self->IsMultipartCreated()) {
        RETURN_NOT_OK(self->CleanupIfFailed(self->FinishPartUploadAfterFlush()));
      }
      return self->CleanupAfterClose();
    });
  }

  bool closed() const override { return closed_; }

  arrow::Result<int64_t> Tell() const override {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed stream");
    }
    return pos_;
  }

  arrow::Status Write(const std::shared_ptr<Buffer>& buffer) override {
    return DoWrite(buffer->data(), buffer->size(), buffer);
  }

  arrow::Status Write(const void* data, int64_t nbytes) override { return DoWrite(data, nbytes); }

  arrow::Status DoWrite(const void* data, int64_t nbytes, std::shared_ptr<Buffer> owned_buffer = nullptr) {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed stream");
    }

    const int8_t* data_ptr = reinterpret_cast<const int8_t*>(data);
    auto advance_ptr = [&data_ptr, &nbytes](const int64_t offset) {
      data_ptr += offset;
      nbytes -= offset;
    };

    // Handle case where we have some bytes buffered from prior calls.
    if (current_part_size_ > 0) {
      // Try to fill current buffer
      const int64_t to_copy = std::min(nbytes, part_upload_size_ - current_part_size_);
      RETURN_NOT_OK(current_part_->Write(data_ptr, to_copy));
      current_part_size_ += to_copy;
      advance_ptr(to_copy);
      pos_ += to_copy;

      // If buffer isn't full, break
      if (current_part_size_ < part_upload_size_) {
        return arrow::Status::OK();
      }

      RETURN_NOT_OK(CommitCurrentPart());
    }

    // We can upload chunks without copying them into a buffer
    while (nbytes >= part_upload_size_) {
      RETURN_NOT_OK(UploadPart(data_ptr, part_upload_size_));
      advance_ptr(part_upload_size_);
      pos_ += part_upload_size_;
    }

    // Buffer remaining bytes
    if (nbytes > 0) {
      current_part_size_ = nbytes;
      ARROW_ASSIGN_OR_RAISE(current_part_,
                            arrow::io::BufferOutputStream::Create(part_upload_size_, io_context_.pool()));
      RETURN_NOT_OK(current_part_->Write(data_ptr, current_part_size_));
      pos_ += current_part_size_;
    }

    return arrow::Status::OK();
  }

  arrow::Status Flush() override {
    auto fut = FlushAsync();
    return fut.status();
  }

  Future<> FlushAsync() {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed stream");
    }
    // Wait for background writes to finish
    std::unique_lock<std::mutex> lock(upload_state_->mutex);
    return upload_state_->pending_uploads_completed;
  }

  // Upload-related helpers

  arrow::Status CommitCurrentPart() {
    if (!IsMultipartCreated()) {
      RETURN_NOT_OK(CreateMultipartUpload());
    }

    ARROW_ASSIGN_OR_RAISE(auto buf, current_part_->Finish());
    current_part_.reset();
    current_part_size_ = 0;
    return UploadPart(buf);
  }

  arrow::Status UploadUsingSingleRequest() {
    std::shared_ptr<Buffer> buf;
    if (current_part_ == nullptr) {
      // In case the stream is closed directly after it has been opened without writing
      // anything, we'll have to create an empty buffer.
      buf = std::make_shared<Buffer>("");
    } else {
      ARROW_ASSIGN_OR_RAISE(buf, current_part_->Finish());
    }

    current_part_.reset();
    current_part_size_ = 0;
    return UploadUsingSingleRequest(buf);
  }

  template <typename RequestType, typename OutcomeType>
  using UploadResultCallbackFunction = std::function<Status(
      const RequestType& request, std::shared_ptr<UploadState>, int32_t part_number, OutcomeType outcome)>;

  static arrow::Result<Aws::S3::Model::PutObjectOutcome> TriggerUploadRequest(
      const Aws::S3::Model::PutObjectRequest& request, const std::shared_ptr<S3ClientHolder>& holder) {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder->Lock());
    return client_lock.Move()->PutObject(request);
  }

  static arrow::Result<Aws::S3::Model::UploadPartOutcome> TriggerUploadRequest(
      const Aws::S3::Model::UploadPartRequest& request, const std::shared_ptr<S3ClientHolder>& holder) {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder->Lock());
    return client_lock.Move()->UploadPart(request);
  }

  template <typename RequestType, typename OutcomeType>
  arrow::Status Upload(RequestType&& req,
                       UploadResultCallbackFunction<RequestType, OutcomeType> sync_result_callback,
                       UploadResultCallbackFunction<RequestType, OutcomeType> async_result_callback,
                       const void* data,
                       int64_t nbytes,
                       std::shared_ptr<Buffer> owned_buffer = nullptr) {
    req.SetBucket(ToAwsString(path_.bucket));
    req.SetKey(ToAwsString(path_.key));
    req.SetBody(std::make_shared<StringViewStream>(data, nbytes));
    req.SetContentLength(nbytes);

    if (!background_writes_) {
      req.SetBody(std::make_shared<StringViewStream>(data, nbytes));

      ARROW_ASSIGN_OR_RAISE(auto outcome, TriggerUploadRequest(req, holder_));

      RETURN_NOT_OK(sync_result_callback(req, upload_state_, part_number_, outcome));
    } else {
      // If the data isn't owned, make an immutable copy for the lifetime of the closure
      if (owned_buffer == nullptr) {
        ARROW_ASSIGN_OR_RAISE(owned_buffer, AllocateBuffer(nbytes, io_context_.pool()));
        memcpy(owned_buffer->mutable_data(), data, nbytes);
      } else {
        DCHECK_EQ(data, owned_buffer->data());
        DCHECK_EQ(nbytes, owned_buffer->size());
      }
      req.SetBody(std::make_shared<StringViewStream>(owned_buffer->data(), owned_buffer->size()));

      {
        std::unique_lock<std::mutex> lock(upload_state_->mutex);
        if (upload_state_->uploads_in_progress++ == 0) {
          upload_state_->pending_uploads_completed = Future<>::Make();
        }
      }

      // The closure keeps the buffer and the upload state alive
      auto deferred = [owned_buffer, holder = holder_, req = std::move(req), state = upload_state_,
                       async_result_callback, part_number = part_number_]() mutable -> arrow::Status {
        ARROW_ASSIGN_OR_RAISE(auto outcome, TriggerUploadRequest(req, holder));

        return async_result_callback(req, state, part_number, outcome);
      };
      ARROW_RETURN_NOT_OK(SubmitIO(io_context_, std::move(deferred)));
    }

    ++part_number_;

    return arrow::Status::OK();
  }

  static arrow::Status UploadUsingSingleRequestError(const Aws::S3::Model::PutObjectRequest& request,
                                                     const Aws::S3::Model::PutObjectOutcome& outcome) {
    return ErrorToStatus(std::forward_as_tuple("When uploading object with key '", request.GetKey(), "' in bucket '",
                                               request.GetBucket(), "': "),
                         "PutObject", outcome.GetError());
  }

  arrow::Status UploadUsingSingleRequest(std::shared_ptr<Buffer> buffer) {
    return UploadUsingSingleRequest(buffer->data(), buffer->size(), buffer);
  }

  arrow::Status UploadUsingSingleRequest(const void* data,
                                         int64_t nbytes,
                                         std::shared_ptr<Buffer> owned_buffer = nullptr) {
    auto sync_result_callback = [](const Aws::S3::Model::PutObjectRequest& request, std::shared_ptr<UploadState> state,
                                   int32_t part_number, Aws::S3::Model::PutObjectOutcome outcome) {
      if (!outcome.IsSuccess()) {
        return UploadUsingSingleRequestError(request, outcome);
      }
      return arrow::Status::OK();
    };

    auto async_result_callback = [](const Aws::S3::Model::PutObjectRequest& request, std::shared_ptr<UploadState> state,
                                    int32_t part_number, Aws::S3::Model::PutObjectOutcome outcome) {
      HandleUploadUsingSingleRequestOutcome(state, request, outcome);
      return arrow::Status::OK();
    };

    Aws::S3::Model::PutObjectRequest req{};
    RETURN_NOT_OK(SetMetadataInRequest(&req));

    return Upload<Aws::S3::Model::PutObjectRequest, Aws::S3::Model::PutObjectOutcome>(
        std::move(req), std::move(sync_result_callback), std::move(async_result_callback), data, nbytes,
        std::move(owned_buffer));
  }

  arrow::Status UploadPart(std::shared_ptr<Buffer> buffer) {
    return UploadPart(buffer->data(), buffer->size(), buffer);
  }

  static arrow::Status UploadPartError(const Aws::S3::Model::UploadPartRequest& request,
                                       const Aws::S3::Model::UploadPartOutcome& outcome) {
    return ErrorToStatus(std::forward_as_tuple("When uploading part for key '", request.GetKey(), "' in bucket '",
                                               request.GetBucket(), "': "),
                         "UploadPart", outcome.GetError());
  }

  arrow::Status UploadPart(const void* data, int64_t nbytes, std::shared_ptr<Buffer> owned_buffer = nullptr) {
    if (!IsMultipartCreated()) {
      RETURN_NOT_OK(CreateMultipartUpload());
    }

    Aws::S3::Model::UploadPartRequest req{};
    req.SetPartNumber(part_number_);
    req.SetUploadId(multipart_upload_id_);

    auto sync_result_callback = [](const Aws::S3::Model::UploadPartRequest& request, std::shared_ptr<UploadState> state,
                                   int32_t part_number, Aws::S3::Model::UploadPartOutcome outcome) {
      if (!outcome.IsSuccess()) {
        return UploadPartError(request, outcome);
      } else {
        AddCompletedPart(state, part_number, outcome.GetResult());
      }

      return arrow::Status::OK();
    };

    auto async_result_callback = [](const Aws::S3::Model::UploadPartRequest& request,
                                    std::shared_ptr<UploadState> state, int32_t part_number,
                                    Aws::S3::Model::UploadPartOutcome outcome) {
      HandleUploadPartOutcome(state, part_number, request, outcome);
      return arrow::Status::OK();
    };

    return Upload<Aws::S3::Model::UploadPartRequest, Aws::S3::Model::UploadPartOutcome>(
        std::move(req), std::move(sync_result_callback), std::move(async_result_callback), data, nbytes,
        std::move(owned_buffer));
  }

  static void HandleUploadUsingSingleRequestOutcome(const std::shared_ptr<UploadState>& state,
                                                    const S3Model::PutObjectRequest& req,
                                                    const S3Model::PutObjectOutcome& outcome) {
    std::unique_lock<std::mutex> lock(state->mutex);
    if (!outcome.IsSuccess()) {
      state->status &= UploadUsingSingleRequestError(req, outcome);
    }

    // GH-41862: avoid potential deadlock if the Future's callback is called
    // with the mutex taken.
    auto fut = state->pending_uploads_completed;
    lock.unlock();
    fut.MarkFinished(state->status);
  }

  static void HandleUploadPartOutcome(const std::shared_ptr<UploadState>& state,
                                      int part_number,
                                      const S3Model::UploadPartRequest& req,
                                      const S3Model::UploadPartOutcome& outcome) {
    std::unique_lock<std::mutex> lock(state->mutex);
    if (!outcome.IsSuccess()) {
      state->status &= UploadPartError(req, outcome);
    } else {
      AddCompletedPart(state, part_number, outcome.GetResult());
    }

    // Notify completion
    if (--state->uploads_in_progress == 0) {
      // GH-41862: avoid potential deadlock if the Future's callback is called
      // with the mutex taken.
      auto fut = state->pending_uploads_completed;
      lock.unlock();
      // State could be mutated concurrently if another thread writes to the
      // stream, but in this case the Flush() call is only advisory anyway.
      // Besides, it's not generally sound to write to an OutputStream from
      // several threads at once.
      fut.MarkFinished(state->status);
    }
  }

  static void AddCompletedPart(const std::shared_ptr<UploadState>& state,
                               int part_number,
                               const S3Model::UploadPartResult& result) {
    S3Model::CompletedPart part;
    // Append ETag and part number for this uploaded part
    // (will be needed for upload completion in Close())
    part.SetPartNumber(part_number);
    part.SetETag(result.GetETag());
    int slot = part_number - 1;
    if (state->completed_parts.size() <= static_cast<size_t>(slot)) {
      state->completed_parts.resize(slot + 1);
    }
    DCHECK(!state->completed_parts[slot].PartNumberHasBeenSet());
    state->completed_parts[slot] = std::move(part);
  }

  protected:
  std::shared_ptr<S3ClientHolder> holder_;
  const arrow::io::IOContext io_context_;
  const S3Path path_;
  const std::shared_ptr<const arrow::KeyValueMetadata> metadata_;
  const std::shared_ptr<const arrow::KeyValueMetadata> default_metadata_;
  const bool background_writes_;
  const bool allow_delayed_open_;

  int64_t part_upload_size_;

  Aws::String multipart_upload_id_;
  bool closed_ = true;
  int64_t pos_ = 0;
  int32_t part_number_ = 1;
  std::shared_ptr<arrow::io::BufferOutputStream> current_part_;
  int64_t current_part_size_ = 0;

  // This struct is kept alive through background writes to avoid problems
  // in the completion handler.
  struct UploadState {
    std::mutex mutex;
    // Only populated for multi-part uploads.
    Aws::Vector<S3Model::CompletedPart> completed_parts;
    int64_t uploads_in_progress = 0;
    arrow::Status status;
    arrow::Future<> pending_uploads_completed = arrow::Future<>::MakeFinished(arrow::Status::OK());
  };
  std::shared_ptr<UploadState> upload_state_;
};

class MultiPartUploadS3FS::Impl : public std::enable_shared_from_this<MultiPartUploadS3FS::Impl> {
  public:
  ClientBuilder builder_;
  const arrow::io::IOContext io_context_;
  std::shared_ptr<S3ClientHolder> holder_;
  std::optional<S3Backend> backend_;

  static constexpr int32_t kListObjectsMaxKeys = 1000;
  // At most 1000 keys per multiple-delete request
  static constexpr int32_t kMultipleDeleteMaxKeys = 1000;

  explicit Impl(S3Options options, arrow::io::IOContext io_context)
      : builder_(std::move(options)), io_context_(io_context) {}

  arrow::Status Init() {
    auto result = builder_.BuildClient(io_context_);
    if (!result.ok()) {
      return arrow::Status::IOError("Failed to build S3 client: ", result.status().ToString());
    }
    return std::move(result).Value(&holder_);
  }

  const S3Options& options() const { return builder_.options(); }

  std::string region() const { return std::string(FromAwsString(builder_.config().region)); }

  template <typename Error>
  void SaveBackend(const Aws::Client::AWSError<Error>& error) {
    if (!backend_ || *backend_ == S3Backend::Other) {
      backend_ = DetectS3Backend(error);
    }
  }

  // Tests to see if a bucket exists
  arrow::Result<bool> BucketExists(const std::string& bucket) {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    S3Model::HeadBucketRequest req;
    req.SetBucket(ToAwsString(bucket));

    auto outcome = client_lock.Move()->HeadBucket(req);
    if (!outcome.IsSuccess()) {
      if (!IsNotFound(outcome.GetError())) {
        return ErrorToStatus(std::forward_as_tuple("When testing for existence of bucket '", bucket, "': "),
                             "HeadBucket", outcome.GetError());
      }
      return false;
    }
    return true;
  }

  // Create a bucket.  Successful if bucket already exists.
  arrow::Status CreateBucket(const std::string& bucket) {
    // Check bucket exists first.
    {
      S3Model::HeadBucketRequest req;
      req.SetBucket(ToAwsString(bucket));
      ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());
      auto outcome = client_lock.Move()->HeadBucket(req);

      if (outcome.IsSuccess()) {
        return arrow::Status::OK();
      } else if (!IsNotFound(outcome.GetError())) {
        return ErrorToStatus(std::forward_as_tuple("When creating bucket '", bucket, "': "), "HeadBucket",
                             outcome.GetError());
      }

      if (!options().allow_bucket_creation) {
        return arrow::Status::IOError("Bucket '", bucket, "' not found. ",
                                      "To create buckets, enable the allow_bucket_creation option.");
      }
    }

    S3Model::CreateBucketConfiguration config;
    S3Model::CreateBucketRequest req;
    auto _region = region();
    // AWS S3 treats the us-east-1 differently than other regions
    // https://docs.aws.amazon.com/cli/latest/reference/s3api/create-bucket.html
    if (_region != "us-east-1") {
      config.SetLocationConstraint(
          S3Model::BucketLocationConstraintMapper::GetBucketLocationConstraintForName(ToAwsString(_region)));
    }
    req.SetBucket(ToAwsString(bucket));
    req.SetCreateBucketConfiguration(config);

    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());
    auto outcome = client_lock.Move()->CreateBucket(req);
    if (!outcome.IsSuccess() && !IsAlreadyExists(outcome.GetError())) {
      return ErrorToStatus(std::forward_as_tuple("When creating bucket '", bucket, "': "), "CreateBucket",
                           outcome.GetError());
    }
    return arrow::Status::OK();
  }

  // Create a directory-like object with empty contents.  Successful if already exists.
  arrow::Status CreateEmptyDir(const std::string& bucket, std::string_view key_view) {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    auto key = EnsureTrailingSlash(key_view);
    S3Model::PutObjectRequest req;
    req.SetBucket(ToAwsString(bucket));
    req.SetKey(ToAwsString(key));
    req.SetContentType(kAwsDirectoryContentType);
    return OutcomeToStatus(std::forward_as_tuple("When creating key '", key, "' in bucket '", bucket, "': "),
                           "PutObject", client_lock.Move()->PutObject(req));
  }

  arrow::Status DeleteObject(const std::string& bucket, const std::string& key) {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    S3Model::DeleteObjectRequest req;
    req.SetBucket(ToAwsString(bucket));
    req.SetKey(ToAwsString(key));
    return OutcomeToStatus(std::forward_as_tuple("When delete key '", key, "' in bucket '", bucket, "': "),
                           "DeleteObject", client_lock.Move()->DeleteObject(req));
  }

  arrow::Status CopyObject(const S3Path& src_path, const S3Path& dest_path) {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    S3Model::CopyObjectRequest req;
    req.SetBucket(ToAwsString(dest_path.bucket));
    req.SetKey(ToAwsString(dest_path.key));
    // ARROW-13048: Copy source "Must be URL-encoded" according to AWS SDK docs.
    // However at least in 1.8 and 1.9 the SDK URL-encodes the path for you
    req.SetCopySource(src_path.ToAwsString());
    return OutcomeToStatus(std::forward_as_tuple("When copying key '", src_path.key, "' in bucket '", src_path.bucket,
                                                 "' to key '", dest_path.key, "' in bucket '", dest_path.bucket, "': "),
                           "CopyObject", client_lock.Move()->CopyObject(req));
  }

  // On Minio, an empty "directory" doesn't satisfy the same API requests as
  // a non-empty "directory".  This is a Minio-specific quirk, but we need
  // to handle it for unit testing.

  // If this method is called after HEAD on "bucket/key" already returned a 404,
  // can pass the given outcome to spare a spurious HEAD call.
  arrow::Result<bool> IsEmptyDirectory(const std::string& bucket,
                                       const std::string& key,
                                       const S3Model::HeadObjectOutcome* previous_outcome = nullptr) {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    if (previous_outcome) {
      // Fetch the backend from the previous error
      DCHECK(!previous_outcome->IsSuccess());
      if (!backend_) {
        SaveBackend(previous_outcome->GetError());
        DCHECK(backend_);
      }
      if (backend_ != S3Backend::Minio) {
        // HEAD already returned a 404, nothing more to do
        return false;
      }
    }

    // We come here in one of two situations:
    // - we don't know the backend and there is no previous outcome
    // - the backend is Minio
    S3Model::HeadObjectRequest req;
    req.SetBucket(ToAwsString(bucket));
    if (backend_ && *backend_ == S3Backend::Minio) {
      // Minio wants a slash at the end, Amazon doesn't
      req.SetKey(ToAwsString(key) + kSep);
    } else {
      req.SetKey(ToAwsString(key));
    }

    auto outcome = client_lock.Move()->HeadObject(req);
    if (outcome.IsSuccess()) {
      return true;
    }
    if (!backend_) {
      SaveBackend(outcome.GetError());
      DCHECK(backend_);
      if (*backend_ == S3Backend::Minio) {
        // Try again with separator-terminated key (see above)
        return IsEmptyDirectory(bucket, key);
      }
    }
    if (IsNotFound(outcome.GetError())) {
      return false;
    }
    return ErrorToStatus(
        std::forward_as_tuple("When reading information for key '", key, "' in bucket '", bucket, "': "), "HeadObject",
        outcome.GetError());
  }

  arrow::Result<bool> IsEmptyDirectory(const S3Path& path,
                                       const S3Model::HeadObjectOutcome* previous_outcome = nullptr) {
    return IsEmptyDirectory(path.bucket, path.key, previous_outcome);
  }

  arrow::Result<bool> IsNonEmptyDirectory(const S3Path& path) {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    S3Model::ListObjectsV2Request req;
    req.SetBucket(ToAwsString(path.bucket));
    req.SetPrefix(ToAwsString(path.key) + kSep);
    req.SetDelimiter(Aws::String() + kSep);
    req.SetMaxKeys(1);
    auto outcome = client_lock.Move()->ListObjectsV2(req);
    if (outcome.IsSuccess()) {
      const S3Model::ListObjectsV2Result& r = outcome.GetResult();
      // In some cases, there may be 0 keys but some prefixes
      return r.GetKeyCount() > 0 || !r.GetCommonPrefixes().empty();
    }
    if (IsNotFound(outcome.GetError())) {
      return false;
    }
    return ErrorToStatus(
        std::forward_as_tuple("When listing objects under key '", path.key, "' in bucket '", path.bucket, "': "),
        "ListObjectsV2", outcome.GetError());
  }

  static FileInfo MakeDirectoryInfo(std::string dirname) {
    FileInfo dir;
    dir.set_type(FileType::Directory);
    dir.set_path(std::move(dirname));
    return dir;
  }

  static std::vector<FileInfo> MakeDirectoryInfos(std::vector<std::string> dirnames) {
    std::vector<FileInfo> dir_infos;
    for (auto& dirname : dirnames) {
      dir_infos.push_back(MakeDirectoryInfo(std::move(dirname)));
    }
    return dir_infos;
  }

  using FileInfoSink = arrow::PushGenerator<std::vector<FileInfo>>::Producer;

  struct FileListerState {
    FileInfoSink files_queue;
    const bool allow_not_found;
    const int max_recursion;
    const bool include_implicit_dirs;
    const arrow::io::IOContext io_context;
    S3ClientHolder* const holder;

    S3Model::ListObjectsV2Request req;
    std::unordered_set<std::string> directories;
    bool empty = true;

    FileListerState(arrow::PushGenerator<std::vector<FileInfo>>::Producer files_queue,
                    FileSelector select,
                    const std::string& bucket,
                    const std::string& key,
                    bool include_implicit_dirs,
                    arrow::io::IOContext io_context,
                    S3ClientHolder* holder)
        : files_queue(std::move(files_queue)),
          allow_not_found(select.allow_not_found),
          max_recursion(select.max_recursion),
          include_implicit_dirs(include_implicit_dirs),
          io_context(std::move(io_context)),
          holder(holder) {
      req.SetBucket(bucket);
      req.SetMaxKeys(kListObjectsMaxKeys);
      if (!key.empty()) {
        req.SetPrefix(key + kSep);
      }
      if (!select.recursive) {
        req.SetDelimiter(Aws::String() + kSep);
      }
    }

    void Finish() {
      // `empty` means that we didn't get a single file info back from S3.  This may be
      // a situation that we should consider as PathNotFound.
      //
      // * If the prefix is empty then we were querying the contents of an entire bucket
      //   and this is not a PathNotFound case because if the bucket didn't exist then
      //   we would have received an error and not an empty set of results.
      //
      // * If the prefix is not empty then we asked for all files under a particular
      //   directory.  S3 will also return the directory itself, if it exists.  So if
      //   we get zero results then we know that there are no files under the directory
      //   and the directory itself doesn't exist.  This should be considered PathNotFound
      if (empty && !allow_not_found && !req.GetPrefix().empty()) {
        files_queue.Push(PathNotFound(req.GetBucket(), req.GetPrefix()));
      }
    }

    // Given a path, iterate through all possible sub-paths and, if we haven't
    // seen that sub-path before, return it.
    //
    // For example, given A/B/C we might return A/B and A if we have not seen
    // those paths before.  This allows us to consider "implicit" directories which
    // don't exist as objects in S3 but can be inferred.
    std::vector<std::string> GetNewDirectories(const std::string_view& path) {
      std::string current(path);
      std::string base = req.GetBucket();
      if (!req.GetPrefix().empty()) {
        base = base + kSep + std::string(RemoveTrailingSlash(req.GetPrefix()));
      }
      std::vector<std::string> new_directories;
      while (true) {
        const std::string parent_dir = GetAbstractPathParent(current).first;
        if (parent_dir.empty()) {
          break;
        }
        current = parent_dir;
        if (current == base) {
          break;
        }
        if (directories.insert(parent_dir).second) {
          new_directories.push_back(std::move(parent_dir));
        }
      }
      return new_directories;
    }
  };

  struct FileListerTask : public arrow::util::AsyncTaskScheduler::Task {
    std::shared_ptr<FileListerState> state;
    arrow::util::AsyncTaskScheduler* scheduler;

    FileListerTask(std::shared_ptr<FileListerState> state, arrow::util::AsyncTaskScheduler* scheduler)
        : state(std::move(state)), scheduler(scheduler) {}

    std::vector<FileInfo> ToFileInfos(const std::string& bucket,
                                      const std::string& prefix,
                                      const S3Model::ListObjectsV2Result& result) {
      std::vector<FileInfo> file_infos;
      // If this is a non-recursive listing we may see "common prefixes" which represent
      // directories we did not recurse into.  We will add those as directories.
      for (const auto& child_prefix : result.GetCommonPrefixes()) {
        const auto child_key = RemoveTrailingSlash(FromAwsString(child_prefix.GetPrefix()));
        std::stringstream child_path_ss;
        child_path_ss << bucket << kSep << child_key;
        FileInfo info;
        info.set_path(child_path_ss.str());
        info.set_type(FileType::Directory);
        file_infos.push_back(std::move(info));
      }
      // S3 doesn't have any concept of "max depth" and so we emulate it by counting the
      // number of '/' characters.  E.g. if the user is searching bucket/subdirA/subdirB
      // then the starting depth is 2.
      // A file subdirA/subdirB/somefile will have a child depth of 2 and a "depth" of 0.
      // A file subdirA/subdirB/subdirC/somefile will have a child depth of 3 and a
      //   "depth" of 1
      int base_depth = arrow::fs::internal::GetAbstractPathDepth(prefix);
      for (const auto& obj : result.GetContents()) {
        if (obj.GetKey() == prefix) {
          // S3 will return the basedir itself (if it is a file / empty file).  We don't
          // want that.  But this is still considered "finding the basedir" and so we mark
          // it "not empty".
          state->empty = false;
          continue;
        }
        std::string child_key = std::string(RemoveTrailingSlash(FromAwsString(obj.GetKey())));
        bool had_trailing_slash = child_key.size() != obj.GetKey().size();
        int child_depth = arrow::fs::internal::GetAbstractPathDepth(child_key);
        // Recursion depth is 1 smaller because a path with depth 1 (e.g. foo) is
        // considered to have a "recursion" of 0
        int recursion_depth = child_depth - base_depth - 1;
        if (recursion_depth > state->max_recursion) {
          // If we have A/B/C/D and max_recursion is 2 then we ignore this (don't add it
          // to file_infos) but we still want to potentially add A and A/B as directories.
          // So we "pretend" like we have a file A/B/C for the call to GetNewDirectories
          // below
          int to_trim = recursion_depth - state->max_recursion - 1;
          if (to_trim > 0) {
            child_key = bucket + kSep + arrow::fs::internal::SliceAbstractPath(child_key, 0, child_depth - to_trim);
          } else {
            child_key = bucket + kSep + child_key;
          }
        } else {
          // If the file isn't beyond our max recursion then count it as a file
          // unless it's empty and then it depends on whether or not the file ends
          // with a trailing slash
          std::stringstream child_path_ss;
          child_path_ss << bucket << kSep << child_key;
          child_key = child_path_ss.str();
          if (obj.GetSize() > 0 || !had_trailing_slash) {
            // We found a real file.
            // XXX Ideally, for 0-sized files we would also check the Content-Type
            // against kAwsDirectoryContentType, but ListObjectsV2 does not give
            // that information.
            FileInfo info;
            info.set_path(child_key);
            FileObjectToInfo(obj, &info);
            file_infos.push_back(std::move(info));
          } else {
            // We found an empty file and we want to treat it like a directory.  Only
            // add it if we haven't seen this directory before.
            if (state->directories.insert(child_key).second) {
              file_infos.push_back(MakeDirectoryInfo(child_key));
            }
          }
        }

        if (state->include_implicit_dirs) {
          // Now that we've dealt with the file itself we need to look at each of the
          // parent paths and potentially add them as directories.  For example, after
          // finding a file A/B/C/D we want to consider adding directories A, A/B, and
          // A/B/C.
          for (const auto& newdir : state->GetNewDirectories(child_key)) {
            file_infos.push_back(MakeDirectoryInfo(newdir));
          }
        }
      }
      if (file_infos.size() > 0) {
        state->empty = false;
      }
      return file_infos;
    }

    void Run() {
      // We are on an I/O thread now so just synchronously make the call and interpret the
      // results.
      arrow::Result<S3ClientLock> client_lock = state->holder->Lock();
      if (!client_lock.ok()) {
        state->files_queue.Push(client_lock.status());
        return;
      }
      S3Model::ListObjectsV2Outcome outcome = client_lock->Move()->ListObjectsV2(state->req);
      if (!outcome.IsSuccess()) {
        const auto& err = outcome.GetError();
        if (state->allow_not_found && IsNotFound(err)) {
          return;
        }
        state->files_queue.Push(
            ErrorToStatus(std::forward_as_tuple("When listing objects under key '", state->req.GetPrefix(),
                                                "' in bucket '", state->req.GetBucket(), "': "),
                          "ListObjectsV2", err));
        return;
      }
      const S3Model::ListObjectsV2Result& result = outcome.GetResult();
      // We could immediately schedule the continuation (if there are enough results to
      // trigger paging) but that would introduce race condition complexity for arguably
      // little benefit.
      std::vector<FileInfo> file_infos = ToFileInfos(state->req.GetBucket(), state->req.GetPrefix(), result);
      if (file_infos.size() > 0) {
        state->files_queue.Push(std::move(file_infos));
      }

      // If there are enough files to warrant a continuation then go ahead and schedule
      // that now.
      if (result.GetIsTruncated()) {
        DCHECK(!result.GetNextContinuationToken().empty());
        state->req.SetContinuationToken(result.GetNextContinuationToken());
        scheduler->AddTask(std::make_unique<FileListerTask>(state, scheduler));
      } else {
        // Otherwise, we have finished listing all the files
        state->Finish();
      }
    }

    arrow::Result<Future<>> operator()() override {
      return state->io_context.executor()->Submit([this] {
        Run();
        return arrow::Status::OK();
      });
    }
    std::string_view name() const override { return "S3ListFiles"; }
  };

  // Lists all file, potentially recursively, in a bucket
  //
  // include_implicit_dirs controls whether or not implicit directories should be
  // included. These are directories that are not actually file objects but instead are
  // inferred from other objects.
  //
  // For example, if a file exists with path A/B/C then implicit directories A/ and A/B/
  // will exist even if there are no file objects with these paths.
  void ListAsync(const FileSelector& select,
                 const std::string& bucket,
                 const std::string& key,
                 bool include_implicit_dirs,
                 arrow::util::AsyncTaskScheduler* scheduler,
                 FileInfoSink sink) {
    // We can only fetch kListObjectsMaxKeys files at a time and so we create a
    // scheduler and schedule a task to grab the first batch.  Once that's done we
    // schedule a new task for the next batch.  All of these tasks share the same
    // FileListerState object but none of these tasks run in parallel so there is
    // no need to worry about mutexes
    auto state = std::make_shared<FileListerState>(sink, select, bucket, key, include_implicit_dirs, io_context_,
                                                   this->holder_.get());

    // Create the first file lister task (it may spawn more)
    auto file_lister_task = std::make_unique<FileListerTask>(state, scheduler);
    scheduler->AddTask(std::move(file_lister_task));
  }

  // Fully list all files from all buckets
  void FullListAsync(bool include_implicit_dirs,
                     arrow::util::AsyncTaskScheduler* scheduler,
                     FileInfoSink sink,
                     bool recursive) {
    scheduler->AddSimpleTask(
        [this, scheduler, sink, include_implicit_dirs, recursive]() mutable {
          return ListBucketsAsync().Then([this, scheduler, sink, include_implicit_dirs,
                                          recursive](const std::vector<std::string>& buckets) mutable {
            // Return the buckets themselves as directories
            std::vector<FileInfo> buckets_as_directories = MakeDirectoryInfos(buckets);
            sink.Push(std::move(buckets_as_directories));

            if (recursive) {
              // Recursively list each bucket (these will run in parallel but sink
              // should be thread safe and so this is ok)
              for (const auto& bucket : buckets) {
                FileSelector select;
                select.allow_not_found = true;
                select.recursive = true;
                select.base_dir = bucket;
                ListAsync(select, bucket, "", include_implicit_dirs, scheduler, sink);
              }
            }
          });
        },
        std::string_view("FullListBucketScan"));
  }

  // Delete multiple objects at once
  Future<> DeleteObjectsAsync(const std::string& bucket, const std::vector<std::string>& keys) {
    struct DeleteCallback {
      std::string bucket;

      arrow::Status operator()(const S3Model::DeleteObjectsOutcome& outcome) const {
        if (!outcome.IsSuccess()) {
          return ErrorToStatus("DeleteObjects", outcome.GetError());
        }
        // Also need to check per-key errors, even on successful outcome
        // See
        // https://docs.aws.amazon.com/fr_fr/AmazonS3/latest/API/multiobjectdeleteapi.html
        const auto& errors = outcome.GetResult().GetErrors();
        if (!errors.empty()) {
          std::stringstream ss;
          ss << "Got the following " << errors.size() << " errors when deleting objects in S3 bucket '" << bucket
             << "':\n";
          for (const auto& error : errors) {
            ss << "- key '" << error.GetKey() << "': " << error.GetMessage() << "\n";
          }
          return arrow::Status::IOError(ss.str());
        }
        return arrow::Status::OK();
      }
    };

    const auto chunk_size = static_cast<size_t>(kMultipleDeleteMaxKeys);
    const DeleteCallback delete_cb{bucket};

    std::vector<Future<>> futures;
    futures.reserve(arrow::bit_util::CeilDiv(keys.size(), chunk_size));

    for (size_t start = 0; start < keys.size(); start += chunk_size) {
      S3Model::DeleteObjectsRequest req;
      S3Model::Delete del;
      size_t remaining = keys.size() - start;
      size_t next_chunk_size = std::min(remaining, chunk_size);
      for (size_t i = start; i < start + next_chunk_size; ++i) {
        del.AddObjects(S3Model::ObjectIdentifier().WithKey(ToAwsString(keys[i])));
      }
      req.SetBucket(ToAwsString(bucket));
      req.SetDelete(std::move(del));
      ARROW_ASSIGN_OR_RAISE(
          auto fut, SubmitIO(io_context_, [holder = holder_, req = std::move(req), delete_cb]() -> arrow::Status {
            ARROW_ASSIGN_OR_RAISE(auto client_lock, holder->Lock());
            return delete_cb(client_lock.Move()->DeleteObjects(req));
          }));
      futures.push_back(std::move(fut));
    }

    return AllFinished(futures);
  }

  arrow::Status DeleteObjects(const std::string& bucket, const std::vector<std::string>& keys) {
    return DeleteObjectsAsync(bucket, keys).status();
  }

  // Check to make sure the given path is not a file
  //
  // Returns true if the path seems to be a directory, false if it is a file
  Future<bool> EnsureIsDirAsync(const std::string& bucket, const std::string& key) {
    if (key.empty()) {
      // There is no way for a bucket to be a file
      return Future<bool>::MakeFinished(true);
    }
    auto self = shared_from_this();
    return DeferNotOk(SubmitIO(io_context_, [self, bucket, key]() mutable -> arrow::Result<bool> {
      S3Model::HeadObjectRequest req;
      req.SetBucket(ToAwsString(bucket));
      req.SetKey(ToAwsString(key));

      ARROW_ASSIGN_OR_RAISE(auto client_lock, self->holder_->Lock());
      auto outcome = client_lock.Move()->HeadObject(req);
      if (outcome.IsSuccess()) {
        return IsDirectory(key, outcome.GetResult());
      }
      if (IsNotFound(outcome.GetError())) {
        // If we can't find it then it isn't a file.
        return true;
      } else {
        return ErrorToStatus(
            std::forward_as_tuple("When getting information for key '", key, "' in bucket '", bucket, "': "),
            "HeadObject", outcome.GetError());
      }
    }));
  }

  // Some operations require running multiple S3 calls, either in parallel or serially. We
  // need to ensure that the S3 filesystem instance stays valid and that S3 isn't
  // finalized.  We do this by wrapping all the tasks in a scheduler which keeps the
  // resources alive
  Future<> RunInScheduler(
      std::function<Status(arrow::util::AsyncTaskScheduler*, MultiPartUploadS3FS::Impl*)> callable) {
    auto self = shared_from_this();
    arrow::FnOnce<Status(arrow::util::AsyncTaskScheduler*)> initial_task =
        [callable = std::move(callable), this](arrow::util::AsyncTaskScheduler* scheduler) mutable {
          return callable(scheduler, this);
        };
    Future<> scheduler_fut = arrow::util::AsyncTaskScheduler::Make(
        std::move(initial_task),
        /*abort_callback=*/
        [](const Status& st) {
          // No need for special abort logic.
        },
        io_context_.stop_token());
    // Keep self alive until all tasks finish
    return scheduler_fut.Then([self]() { return arrow::Status::OK(); });
  }

  Future<> DoDeleteDirContentsAsync(const std::string& bucket, const std::string& key) {
    return RunInScheduler([bucket, key](arrow::util::AsyncTaskScheduler* scheduler, MultiPartUploadS3FS::Impl* self) {
      scheduler->AddSimpleTask(
          [=] {
            FileSelector select;
            select.base_dir = bucket + kSep + key;
            select.recursive = true;
            select.allow_not_found = false;

            FileInfoGenerator file_infos = self->GetFileInfoGenerator(select);

            auto handle_file_infos = [=](const std::vector<FileInfo>& file_infos) {
              std::vector<std::string> file_paths;
              for (const auto& file_info : file_infos) {
                DCHECK_GT(file_info.path().size(), bucket.size());
                auto file_path = file_info.path().substr(bucket.size() + 1);
                if (file_info.IsDirectory()) {
                  // The selector returns FileInfo objects for directories with a
                  // a path that never ends in a trailing slash, but for AWS the file
                  // needs to have a trailing slash to recognize it as directory
                  // (https://github.com/apache/arrow/issues/38618)
                  DCHECK_OK(arrow::fs::internal::AssertNoTrailingSlash(file_path));
                  file_path = file_path + kSep;
                }
                file_paths.push_back(std::move(file_path));
              }
              scheduler->AddSimpleTask(
                  [=, file_paths = std::move(file_paths)] { return self->DeleteObjectsAsync(bucket, file_paths); },
                  std::string_view("DeleteDirContentsDeleteTask"));
              return arrow::Status::OK();
            };

            return VisitAsyncGenerator(arrow::AsyncGenerator<std::vector<FileInfo>>(std::move(file_infos)),
                                       std::move(handle_file_infos));
          },
          std::string_view("ListFilesForDelete"));
      return arrow::Status::OK();
    });
  }

  Future<> DeleteDirContentsAsync(const std::string& bucket, const std::string& key) {
    auto self = shared_from_this();
    return EnsureIsDirAsync(bucket, key).Then([self, bucket, key](bool is_dir) -> Future<> {
      if (!is_dir) {
        return arrow::Status::IOError("Cannot delete directory contents at ", bucket, kSep, key,
                                      " because it is a file");
      }
      return self->DoDeleteDirContentsAsync(bucket, key);
    });
  }

  FileInfoGenerator GetFileInfoGenerator(const FileSelector& select) {
    auto maybe_base_path = S3Path::FromString(select.base_dir);
    if (!maybe_base_path.ok()) {
      return arrow::MakeFailingGenerator<FileInfoVector>(maybe_base_path.status());
    }
    auto base_path = *std::move(maybe_base_path);

    arrow::PushGenerator<std::vector<FileInfo>> generator;
    Future<> scheduler_fut =
        RunInScheduler([select, base_path, sink = generator.producer()](arrow::util::AsyncTaskScheduler* scheduler,
                                                                        MultiPartUploadS3FS::Impl* self) {
          if (base_path.empty()) {
            bool should_recurse = select.recursive && select.max_recursion > 0;
            self->FullListAsync(/*include_implicit_dirs=*/true, scheduler, sink, should_recurse);
          } else {
            self->ListAsync(select, base_path.bucket, base_path.key,
                            /*include_implicit_dirs=*/true, scheduler, sink);
          }
          return arrow::Status::OK();
        });

    // Mark the generator done once all tasks are finished
    scheduler_fut.AddCallback([sink = generator.producer()](const Status& st) mutable {
      if (!st.ok()) {
        sink.Push(st);
      }
      sink.Close();
    });

    return generator;
  }

  arrow::Status EnsureDirectoryExists(const S3Path& path) {
    if (!path.key.empty()) {
      return CreateEmptyDir(path.bucket, path.key);
    }
    return arrow::Status::OK();
  }

  arrow::Status EnsureParentExists(const S3Path& path) {
    if (path.has_parent()) {
      return EnsureDirectoryExists(path.parent());
    }
    return arrow::Status::OK();
  }

  static arrow::Result<std::vector<std::string>> ProcessListBuckets(const S3Model::ListBucketsOutcome& outcome) {
    if (!outcome.IsSuccess()) {
      return ErrorToStatus(std::forward_as_tuple("When listing buckets: "), "ListBuckets", outcome.GetError());
    }
    std::vector<std::string> buckets;
    buckets.reserve(outcome.GetResult().GetBuckets().size());
    for (const auto& bucket : outcome.GetResult().GetBuckets()) {
      buckets.emplace_back(FromAwsString(bucket.GetName()));
    }
    return buckets;
  }

  arrow::Result<std::vector<std::string>> ListBuckets() {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());
    return ProcessListBuckets(client_lock.Move()->ListBuckets());
  }

  Future<std::vector<std::string>> ListBucketsAsync() {
    auto deferred = [self = shared_from_this()]() mutable -> arrow::Result<std::vector<std::string>> {
      ARROW_ASSIGN_OR_RAISE(auto client_lock, self->holder_->Lock());
      return self->ProcessListBuckets(client_lock.Move()->ListBuckets());
    };
    return DeferNotOk(SubmitIO(io_context_, std::move(deferred)));
  }

  arrow::Result<std::shared_ptr<ObjectInputFile>> OpenInputFile(const std::string& s, S3FileSystem* fs) {
    ARROW_RETURN_NOT_OK(arrow::fs::internal::AssertNoTrailingSlash(s));
    ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));
    RETURN_NOT_OK(ValidateFilePath(path));

    RETURN_NOT_OK(CheckS3Initialized());

    auto ptr = std::make_shared<ObjectInputFile>(holder_, fs->io_context(), path);
    RETURN_NOT_OK(ptr->Init());
    return ptr;
  }

  arrow::Result<std::shared_ptr<ObjectInputFile>> OpenInputFile(const FileInfo& info, S3FileSystem* fs) {
    ARROW_RETURN_NOT_OK(arrow::fs::internal::AssertNoTrailingSlash(info.path()));
    if (info.type() == FileType::NotFound) {
      return ::arrow::fs::internal::PathNotFound(info.path());
    }
    if (info.type() != FileType::File && info.type() != FileType::Unknown) {
      return ::arrow::fs::internal::NotAFile(info.path());
    }

    ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(info.path()));
    RETURN_NOT_OK(ValidateFilePath(path));

    RETURN_NOT_OK(CheckS3Initialized());

    auto ptr = std::make_shared<ObjectInputFile>(holder_, fs->io_context(), path, info.size());
    RETURN_NOT_OK(ptr->Init());
    return ptr;
  }

  arrow::Result<std::shared_ptr<S3ClientMetrics>> GetMetrics() {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());
    return {client_lock.Move()->GetMetrics()};
  }
};

MultiPartUploadS3FS::~MultiPartUploadS3FS() {}

arrow::Result<std::shared_ptr<MultiPartUploadS3FS>> MultiPartUploadS3FS::Make(const ExtendedS3Options& options,
                                                                              const arrow::io::IOContext& io_context) {
  RETURN_NOT_OK(CheckS3Initialized());

  std::shared_ptr<MultiPartUploadS3FS> ptr(new MultiPartUploadS3FS(options, io_context));
  RETURN_NOT_OK(ptr->impl_->Init());
  return ptr;
}

bool MultiPartUploadS3FS::Equals(const FileSystem& other) const {
  if (this == &other) {
    return true;
  }
  if (other.type_name() != type_name()) {
    return false;
  }
  const auto& s3fs = ::arrow::fs::internal::checked_cast<const S3FileSystem&>(other);
  return options().Equals(s3fs.options());
}

arrow::Result<std::string> MultiPartUploadS3FS::PathFromUri(const std::string& uri_string) const {
  return arrow::fs::internal::PathFromUriHelper(uri_string, {"multiPartUploadS3"}, /*accept_local_paths=*/false,
                                                arrow::fs::internal::AuthorityHandlingBehavior::kPrepend);
}

arrow::Result<FileInfo> MultiPartUploadS3FS::GetFileInfo(const std::string& s) {
  ARROW_ASSIGN_OR_RAISE(auto client_lock, impl_->holder_->Lock());

  ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));
  FileInfo info;
  info.set_path(s);

  if (path.empty()) {
    // It's the root path ""
    info.set_type(FileType::Directory);
    return info;
  } else if (path.key.empty()) {
    // It's a bucket
    S3Model::HeadBucketRequest req;
    req.SetBucket(ToAwsString(path.bucket));

    auto outcome = client_lock.Move()->HeadBucket(req);
    if (!outcome.IsSuccess()) {
      if (!IsNotFound(outcome.GetError())) {
        const auto msg = "When getting information for bucket '" + path.bucket + "': ";
        return ErrorToStatus(msg, "HeadBucket", outcome.GetError(), impl_->options().region);
      }
      info.set_type(FileType::NotFound);
      return info;
    }
    // NOTE: S3 doesn't have a bucket modification time.  Only a creation
    // time is available, and you have to list all buckets to get it.
    info.set_type(FileType::Directory);
    return info;
  } else {
    // It's an object
    S3Model::HeadObjectRequest req;
    req.SetBucket(ToAwsString(path.bucket));
    req.SetKey(ToAwsString(path.key));

    auto outcome = client_lock.Move()->HeadObject(req);
    if (outcome.IsSuccess()) {
      // "File" object found
      FileObjectToInfo(path.key, outcome.GetResult(), &info);
      return info;
    }
    if (!IsNotFound(outcome.GetError())) {
      const auto msg = "When getting information for key '" + path.key + "' in bucket '" + path.bucket + "': ";
      return ErrorToStatus(msg, "HeadObject", outcome.GetError(), impl_->options().region);
    }
    // Not found => perhaps it's an empty "directory"
    ARROW_ASSIGN_OR_RAISE(bool is_dir, impl_->IsEmptyDirectory(path, &outcome));
    if (is_dir) {
      info.set_type(FileType::Directory);
      return info;
    }
    // Not found => perhaps it's a non-empty "directory"
    ARROW_ASSIGN_OR_RAISE(is_dir, impl_->IsNonEmptyDirectory(path));
    if (is_dir) {
      info.set_type(FileType::Directory);
    } else {
      info.set_type(FileType::NotFound);
    }
    return info;
  }
}

arrow::Result<FileInfoVector> MultiPartUploadS3FS::GetFileInfo(const FileSelector& select) {
  Future<std::vector<FileInfoVector>> file_infos_fut = CollectAsyncGenerator(GetFileInfoGenerator(select));
  ARROW_ASSIGN_OR_RAISE(std::vector<FileInfoVector> file_infos, file_infos_fut.result());
  FileInfoVector combined_file_infos;
  for (const auto& file_info_vec : file_infos) {
    combined_file_infos.insert(combined_file_infos.end(), file_info_vec.begin(), file_info_vec.end());
  }
  return combined_file_infos;
}

FileInfoGenerator MultiPartUploadS3FS::GetFileInfoGenerator(const FileSelector& select) {
  return impl_->GetFileInfoGenerator(select);
}

arrow::Status MultiPartUploadS3FS::CreateDir(const std::string& s, bool recursive) {
  ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));

  if (path.key.empty()) {
    // Create bucket
    return impl_->CreateBucket(path.bucket);
  }

  FileInfo file_info;
  // Create object
  if (recursive) {
    // Ensure bucket exists
    ARROW_ASSIGN_OR_RAISE(bool bucket_exists, impl_->BucketExists(path.bucket));
    if (!bucket_exists) {
      RETURN_NOT_OK(impl_->CreateBucket(path.bucket));
    }

    auto key_i = path.key_parts.begin();
    std::string parent_key{};
    if (options().check_directory_existence_before_creation) {
      // Walk up the directory first to find the first existing parent
      for (const auto& part : path.key_parts) {
        parent_key += part;
        parent_key += kSep;
      }
      for (key_i = path.key_parts.end(); key_i-- != path.key_parts.begin();) {
        ARROW_ASSIGN_OR_RAISE(file_info, this->GetFileInfo(path.bucket + kSep + parent_key));
        if (file_info.type() != FileType::NotFound) {
          // Found!
          break;
        } else {
          // remove the kSep and the part
          parent_key.pop_back();
          parent_key.erase(parent_key.end() - key_i->size(), parent_key.end());
        }
      }
      key_i++;  // Above for loop moves one extra iterator at the end
    }
    // Ensure that all parents exist, then the directory itself
    // Create all missing directories
    for (; key_i < path.key_parts.end(); ++key_i) {
      parent_key += *key_i;
      parent_key += kSep;
      RETURN_NOT_OK(impl_->CreateEmptyDir(path.bucket, parent_key));
    }
    return arrow::Status::OK();
  } else {
    // Check parent dir exists
    if (path.has_parent()) {
      S3Path parent_path = path.parent();
      ARROW_ASSIGN_OR_RAISE(bool exists, impl_->IsNonEmptyDirectory(parent_path));
      if (!exists) {
        ARROW_ASSIGN_OR_RAISE(exists, impl_->IsEmptyDirectory(parent_path));
      }
      if (!exists) {
        return arrow::Status::IOError("Cannot create directory '", path.full_path,
                                      "': parent directory does not exist");
      }
    }
  }

  // Check if the directory exists already
  if (options().check_directory_existence_before_creation) {
    ARROW_ASSIGN_OR_RAISE(file_info, this->GetFileInfo(path.full_path));
    if (file_info.type() != FileType::NotFound) {
      return arrow::Status::OK();
    }
  }
  // XXX Should we check that no non-directory entry exists?
  // Minio does it for us, not sure about other S3 implementations.
  return impl_->CreateEmptyDir(path.bucket, path.key);
}

arrow::Status MultiPartUploadS3FS::DeleteDir(const std::string& s) {
  ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));
  if (path.empty()) {
    return arrow::Status::NotImplemented("Cannot delete all S3 buckets");
  }
  RETURN_NOT_OK(impl_->DeleteDirContentsAsync(path.bucket, path.key).status());
  if (path.key.empty() && options().allow_bucket_deletion) {
    // Delete bucket
    ARROW_ASSIGN_OR_RAISE(auto client_lock, impl_->holder_->Lock());
    S3Model::DeleteBucketRequest req;
    req.SetBucket(ToAwsString(path.bucket));
    return OutcomeToStatus(std::forward_as_tuple("When deleting bucket '", path.bucket, "': "), "DeleteBucket",
                           client_lock.Move()->DeleteBucket(req));
  } else if (path.key.empty()) {
    return arrow::Status::IOError("Would delete bucket '", path.bucket, "'. ",
                                  "To delete buckets, enable the allow_bucket_deletion option.");
  } else {
    // Delete "directory"
    RETURN_NOT_OK(impl_->DeleteObject(path.bucket, path.key + kSep));
    // Parent may be implicitly deleted if it became empty, recreate it
    return impl_->EnsureParentExists(path);
  }
}

arrow::Status MultiPartUploadS3FS::DeleteDirContents(const std::string& s, bool missing_dir_ok) {
  return DeleteDirContentsAsync(s, missing_dir_ok).status();
}

arrow::Future<> MultiPartUploadS3FS::DeleteDirContentsAsync(const std::string& s, bool missing_dir_ok) {
  ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));

  if (path.empty()) {
    return arrow::Status::NotImplemented("Cannot delete all S3 buckets");
  }
  auto self = impl_;
  return impl_->DeleteDirContentsAsync(path.bucket, path.key)
      .Then(
          [path, self]() {
            // Directory may be implicitly deleted, recreate it
            return self->EnsureDirectoryExists(path);
          },
          [missing_dir_ok](const Status& err) {
            if (missing_dir_ok && ::arrow::internal::ErrnoFromStatus(err) == ENOENT) {
              return arrow::Status::OK();
            }
            return err;
          });
}

arrow::Status MultiPartUploadS3FS::DeleteRootDirContents() {
  return arrow::Status::NotImplemented("Cannot delete all S3 buckets");
}

arrow::Status MultiPartUploadS3FS::DeleteFile(const std::string& s) {
  ARROW_ASSIGN_OR_RAISE(auto client_lock, impl_->holder_->Lock());

  ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));
  RETURN_NOT_OK(ValidateFilePath(path));

  // Check the object exists
  S3Model::HeadObjectRequest req;
  req.SetBucket(ToAwsString(path.bucket));
  req.SetKey(ToAwsString(path.key));

  auto outcome = client_lock.Move()->HeadObject(req);
  if (!outcome.IsSuccess()) {
    if (IsNotFound(outcome.GetError())) {
      return PathNotFound(path);
    } else {
      return ErrorToStatus(
          std::forward_as_tuple("When getting information for key '", path.key, "' in bucket '", path.bucket, "': "),
          "HeadObject", outcome.GetError());
    }
  }
  // Object found, delete it
  RETURN_NOT_OK(impl_->DeleteObject(path.bucket, path.key));
  // Parent may be implicitly deleted if it became empty, recreate it
  return impl_->EnsureParentExists(path);
}

arrow::Status MultiPartUploadS3FS::Move(const std::string& src, const std::string& dest) {
  // XXX We don't implement moving directories as it would be too expensive:
  // one must copy all directory contents one by one (including object data),
  // then delete the original contents.

  ARROW_ASSIGN_OR_RAISE(auto src_path, S3Path::FromString(src));
  RETURN_NOT_OK(ValidateFilePath(src_path));
  ARROW_ASSIGN_OR_RAISE(auto dest_path, S3Path::FromString(dest));
  RETURN_NOT_OK(ValidateFilePath(dest_path));

  if (src_path == dest_path) {
    return arrow::Status::OK();
  }
  RETURN_NOT_OK(impl_->CopyObject(src_path, dest_path));
  RETURN_NOT_OK(impl_->DeleteObject(src_path.bucket, src_path.key));
  // Source parent may be implicitly deleted if it became empty, recreate it
  return impl_->EnsureParentExists(src_path);
}

arrow::Status MultiPartUploadS3FS::CopyFile(const std::string& src, const std::string& dest) {
  ARROW_ASSIGN_OR_RAISE(auto src_path, S3Path::FromString(src));
  RETURN_NOT_OK(ValidateFilePath(src_path));
  ARROW_ASSIGN_OR_RAISE(auto dest_path, S3Path::FromString(dest));
  RETURN_NOT_OK(ValidateFilePath(dest_path));

  if (src_path == dest_path) {
    return arrow::Status::OK();
  }
  return impl_->CopyObject(src_path, dest_path);
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> MultiPartUploadS3FS::OpenOutputStreamWithUploadSize(
    const std::string& s, int64_t upload_size) {
  return OpenOutputStreamWithUploadSize(s, std::shared_ptr<const arrow::KeyValueMetadata>{}, upload_size);
};

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> MultiPartUploadS3FS::OpenOutputStreamWithUploadSize(
    const std::string& s, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata, int64_t upload_size) {
  ARROW_RETURN_NOT_OK(arrow::fs::internal::AssertNoTrailingSlash(s));
  ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));
  RETURN_NOT_OK(ValidateFilePath(path));

  RETURN_NOT_OK(CheckS3Initialized());

  auto ptr =
      std::make_shared<CustomOutputStream>(impl_->holder_, io_context(), path, impl_->options(), metadata, upload_size);
  RETURN_NOT_OK(ptr->Init());
  return ptr;
};

MultiPartUploadS3FS::MultiPartUploadS3FS(const ExtendedS3Options& options, const arrow::io::IOContext& io_context)
    : arrow::fs::S3FileSystem(options, io_context), impl_(std::make_shared<Impl>(options, io_context)) {
  default_async_is_sync_ = false;
}

arrow::Result<std::shared_ptr<arrow::io::InputStream>> MultiPartUploadS3FS::OpenInputStream(const std::string& s) {
  return impl_->OpenInputFile(s, this);
}

arrow::Result<std::shared_ptr<arrow::io::InputStream>> MultiPartUploadS3FS::OpenInputStream(const FileInfo& info) {
  return impl_->OpenInputFile(info, this);
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> MultiPartUploadS3FS::OpenInputFile(const std::string& s) {
  return impl_->OpenInputFile(s, this);
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> MultiPartUploadS3FS::OpenInputFile(const FileInfo& info) {
  return impl_->OpenInputFile(info, this);
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> MultiPartUploadS3FS::OpenOutputStream(
    const std::string& s, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) {
  return OpenOutputStreamWithUploadSize(s, std::shared_ptr<const arrow::KeyValueMetadata>{}, 10 * 1024 * 1024);
};

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> MultiPartUploadS3FS::OpenAppendStream(
    const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) {
  return arrow::Status::NotImplemented("It is not possible to append efficiently to S3 objects");
}

arrow::Result<std::shared_ptr<S3ClientMetrics>> MultiPartUploadS3FS::GetMetrics() { return impl_->GetMetrics(); }

// -----------------------------------------------------------------------
// AWS SDK Initialization and finalization

namespace {

struct AwsInstance {
  AwsInstance() : is_initialized_(false), is_finalized_(false) {}
  ~AwsInstance() { Finalize(/*from_destructor=*/true); }

  // Returns true iff the instance was newly initialized with `options`
  arrow::Result<bool> EnsureInitialized(const ExtendS3GlobalOptions& options) {
    // NOTE: The individual accesses are atomic but the entire sequence below is not.
    // The application should serialize calls to InitializeS3() and FinalizeS3()
    // (see docstrings).
    if (is_finalized_.load()) {
      return arrow::Status::Invalid("Attempt to initialize S3 after it has been finalized");
    }
    bool newly_initialized = false;
    // EnsureInitialized() can be called concurrently by FileSystemFromUri,
    // therefore we need to serialize initialization (GH-39897).
    std::call_once(initialize_flag_, [&]() {
      bool was_initialized = is_initialized_.exchange(true);
      DCHECK(!was_initialized);
      DoInitialize(options);
      newly_initialized = true;
    });
    return newly_initialized;
  }

  bool IsInitialized() { return !is_finalized_ && is_initialized_; }

  bool IsFinalized() { return is_finalized_; }

  void Finalize(bool from_destructor = false) {
    if (is_finalized_.exchange(true)) {
      // Already finalized
      return;
    }
    auto client_finalizer = GetClientFinalizer();
    if (is_initialized_.exchange(false)) {
      // Was initialized
      if (from_destructor) {
        ARROW_LOG(WARNING) << " arrow::fs::FinalizeS3 was not called even though S3 was initialized.  "
                              "This could lead to a segmentation fault at exit";
        auto* leaked_shared_ptr = new std::shared_ptr<S3ClientFinalizer>(client_finalizer);
        ARROW_UNUSED(leaked_shared_ptr);
        return;
      }
      client_finalizer->Finalize();
#ifdef ARROW_S3_HAS_S3CLIENT_CONFIGURATION
      EndpointProviderCache::Instance()->Reset();
#endif
      Aws::ShutdownAPI(aws_options_);
    }
  }

  private:
  void DoInitialize(const ExtendS3GlobalOptions& options) {
    Aws::Utils::Logging::LogLevel aws_log_level;

#define LOG_LEVEL_CASE(level_name)                             \
  case S3LogLevel::level_name:                                 \
    aws_log_level = Aws::Utils::Logging::LogLevel::level_name; \
    break;

    switch (options.log_level) {
      LOG_LEVEL_CASE(Fatal)
      LOG_LEVEL_CASE(Error)
      LOG_LEVEL_CASE(Warn)
      LOG_LEVEL_CASE(Info)
      LOG_LEVEL_CASE(Debug)
      LOG_LEVEL_CASE(Trace)
      default:
        aws_log_level = Aws::Utils::Logging::LogLevel::Off;
    }

#undef LOG_LEVEL_CASE

    aws_options_.loggingOptions.logLevel = aws_log_level;
    // By default the AWS SDK logs to files, log to console instead
    aws_options_.loggingOptions.logger_create_fn = [this] {
      return std::make_shared<Aws::Utils::Logging::ConsoleLogSystem>(aws_options_.loggingOptions.logLevel);
    };
    if (options.override_default_http_options) {
      aws_options_.httpOptions = options.http_options;
    }
    Aws::InitAPI(aws_options_);
  }

  Aws::SDKOptions aws_options_;
  std::atomic<bool> is_initialized_;
  std::atomic<bool> is_finalized_;
  std::once_flag initialize_flag_;
};

AwsInstance* GetAwsInstance() {
  // make sure ClientFinializer is initialized before the AwsInstance
  // so that the static object destructor is called later
  GetClientFinalizer();
  static auto instance = std::make_unique<AwsInstance>();
  return instance.get();
}

arrow::Result<bool> EnsureAwsInstanceInitialized(const ExtendS3GlobalOptions& options) {
  return GetAwsInstance()->EnsureInitialized(options);
}

}  // namespace

arrow::Status InitializeS3(const ExtendS3GlobalOptions& options) {
  ARROW_ASSIGN_OR_RAISE(bool successfully_initialized, EnsureAwsInstanceInitialized(options));
  if (!successfully_initialized) {
    return arrow::Status::Invalid(
        "S3 was already initialized.  It is safe to use but the options passed in this "
        "call have been ignored.");
  }
  return arrow::Status::OK();
}

arrow::Status EnsureS3Initialized() { return EnsureAwsInstanceInitialized(ExtendS3GlobalOptions::Defaults()).status(); }

arrow::Status FinalizeS3() {
  // GetAwsInstance()->Finalize();
  auto instance = GetAwsInstance();
  // The AWS instance might already be destroyed in case FinalizeS3
  // is called from an atexit handler (which is a bad idea anyway as the
  // AWS SDK is not safe anymore to shutdown by this time). See GH-44071.
  if (instance == nullptr) {
    return arrow::Status::Invalid("FinalizeS3 called too late");
  }
  instance->Finalize();
  return arrow::Status::OK();
}

arrow::Status EnsureS3Finalized() { return FinalizeS3(); }

bool IsS3Initialized() { return GetAwsInstance()->IsInitialized(); }

bool IsS3Finalized() { return GetAwsInstance()->IsFinalized(); }

ExtendS3GlobalOptions ExtendS3GlobalOptions::Defaults() {
  auto log_level = S3LogLevel::Fatal;

  return ExtendS3GlobalOptions{log_level};
}

}  // namespace milvus_storage
