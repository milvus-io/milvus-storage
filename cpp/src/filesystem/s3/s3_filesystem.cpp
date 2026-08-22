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

#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/fs.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <chrono>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <optional>
#include <shared_mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <arrow/util/async_generator.h>
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/log.h"
#include <arrow/util/logging.h>
#include <arrow/buffer.h>
#include <arrow/result.h>
#include <arrow/io/memory.h>
#include <arrow/util/future.h>
#include <arrow/util/thread_pool.h>
#include <arrow/filesystem/path_util.h>
#include <arrow/io/interfaces.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/util/string.h>

#include <aws/core/Aws.h>
#include <aws/core/Region.h>
#include <aws/core/VersionConfig.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/auth/STSCredentialsProvider.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/client/RetryStrategy.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/core/utils/xml/XmlSerializer.h>
#include <aws/identity-management/auth/STSAssumeRoleCredentialsProvider.h>
#include <aws/s3/S3Client.h>
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
#ifdef WITH_CRT
#include <aws/s3-crt/model/GetObjectRequest.h>
#include <aws/s3-crt/model/HeadBucketRequest.h>
#include <aws/s3-crt/model/HeadObjectRequest.h>
#endif

#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/common/path_util.h"
#include "milvus-storage/common/writer_status.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/util_internal.h"
#include "milvus-storage/filesystem/s3/s3_client.h"
#include "milvus-storage/filesystem/s3/s3_client_builder.h"
#ifdef WITH_CRT
#include "milvus-storage/filesystem/s3/s3_crt_client.h"
#endif

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
using ::arrow::fs::internal::RemoveTrailingSlash;
using ::Aws::Client::AWSError;
using ::milvus_storage::S3Options;
using ::milvus_storage::fs::internal::BoundedDetail;
using ::milvus_storage::fs::internal::ConnectRetryStrategy;
using ::milvus_storage::fs::internal::DetectS3Backend;
using ::milvus_storage::fs::internal::ErrorToStatus;
using ::milvus_storage::fs::internal::S3ErrorProvenance;
using ::milvus_storage::fs::internal::S3ResourceKind;

namespace {
template <typename Holder>
S3ErrorProvenance ProvenanceOf(const Holder&, S3ResourceKind resource_kind = S3ResourceKind::Unknown) {
  return S3ErrorProvenance{resource_kind};
}
}  // namespace
using ::milvus_storage::fs::internal::FromAwsDatetime;
using ::milvus_storage::fs::internal::FromAwsString;
using ::milvus_storage::fs::internal::IsAlreadyExists;
using ::milvus_storage::fs::internal::IsBucketNotFound;
using ::milvus_storage::fs::internal::IsExplicitBucketNotFound;
using ::milvus_storage::fs::internal::IsObjectNotFound;
using ::milvus_storage::fs::internal::OutcomeToResult;
using ::milvus_storage::fs::internal::OutcomeToStatus;
using ::milvus_storage::fs::internal::S3Backend;
using ::milvus_storage::fs::internal::ToAwsString;

namespace S3Model = Aws::S3::Model;
#ifdef WITH_CRT
namespace S3CrtModel = Aws::S3Crt::Model;
#endif

namespace milvus_storage {

using arrow::io::internal::SubmitIO;
using arrow::io::internal::SubmitIOWithCompletion;

// -----------------------------------------------------------------------
// S3FileSystem implementation

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

template <typename T, typename = void>
struct HasSetACL : std::false_type {};

template <typename T>
struct HasSetACL<T, std::void_t<decltype(std::declval<T>().SetACL(std::declval<Aws::S3::Model::ObjectCannedACL>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSetContentType : std::false_type {};

template <typename T>
struct HasSetContentType<T, std::void_t<decltype(std::declval<T>().SetContentType(std::declval<Aws::String>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSetCacheControl : std::false_type {};

template <typename T>
struct HasSetCacheControl<T, std::void_t<decltype(std::declval<T>().SetCacheControl(std::declval<Aws::String>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSetContentLanguage : std::false_type {};

template <typename T>
struct HasSetContentLanguage<T,
                             std::void_t<decltype(std::declval<T>().SetContentLanguage(std::declval<Aws::String>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSetExpires : std::false_type {};

template <typename T>
struct HasSetExpires<T, std::void_t<decltype(std::declval<T>().SetExpires(std::declval<Aws::Utils::DateTime>()))>>
    : std::true_type {};

template <typename ObjectRequest>
struct ObjectMetadataSetter {
  using Setter = std::function<Status(const std::string& value, ObjectRequest* req)>;

  static std::unordered_map<std::string, Setter> GetSetters() {
    std::unordered_map<std::string, Setter> setters;
    if constexpr (HasSetACL<ObjectRequest>::value) {
      setters.emplace("ACL", CannedACLSetter());
    }

    if constexpr (HasSetContentType<ObjectRequest>::value) {
      setters.emplace("Content-Type", ContentTypeSetter());
    }

    if constexpr (HasSetCacheControl<ObjectRequest>::value) {
      setters.emplace("Cache-Control", StringSetter(&ObjectRequest::SetCacheControl));
    }
    if constexpr (HasSetContentLanguage<ObjectRequest>::value) {
      setters.emplace("Content-Language", StringSetter(&ObjectRequest::SetContentLanguage));
    }
    if constexpr (HasSetExpires<ObjectRequest>::value) {
      setters.emplace("Expires", DateTimeSetter(&ObjectRequest::SetExpires));
    }
    return setters;
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

  [[nodiscard]] Aws::String ToAwsString() const {
    Aws::String res(bucket.begin(), bucket.end());
    res.reserve(bucket.size() + key.size() + 1);
    res += kSep;
    res.append(key.begin(), key.end());
    return res;
  }

  [[nodiscard]] S3Path parent() const {
    DCHECK(!key_parts.empty());
    auto parent = S3Path{"", bucket, "", key_parts};
    parent.key_parts.pop_back();
    parent.key = arrow::fs::internal::JoinAbstractPath(parent.key_parts);
    parent.full_path = parent.bucket + kSep + parent.key;
    return parent;
  }

  [[nodiscard]] bool has_parent() const { return !key.empty(); }

  [[nodiscard]] bool empty() const { return bucket.empty() && key.empty(); }

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

static std::unordered_set<std::string> condition_write_key = {"If-None-Match", "x-goog-if-generation-match",
                                                              "x-cos-forbid-overwrite", "x-oss-forbid-overwrite"};

static std::unordered_map<std::string, std::pair<std::string, std::string>> condition_write_map = {
    {kCloudProviderAWS, {"If-None-Match", "*"}},
    {kCloudProviderGCP, {"x-goog-if-generation-match", "0"}},
    {kCloudProviderTencent, {"x-cos-forbid-overwrite", "true"}},
    {kCloudProviderAliyun, {"x-oss-forbid-overwrite", "true"}},
    {kAzureFileSystemName, {"If-None-Match", "*"}}};

bool IsConditionWriteKey(const std::string& key) { return condition_write_key.find(key) != condition_write_key.end(); }

/// use the SFINAE to check if the type has the member functions
template <typename T, typename = void>
struct HasAddMetadata : std::false_type {};

template <typename T>
struct HasAddMetadata<
    T,
    std::void_t<decltype(std::declval<T>().AddMetadata(std::declval<Aws::String>(), std::declval<Aws::String>()))>>
    : std::true_type {};

template <typename ObjectRequest>
arrow::Status SetObjectMetadata(const std::shared_ptr<const arrow::KeyValueMetadata>& metadata, ObjectRequest* req) {
  static auto setters = ObjectMetadataSetter<ObjectRequest>::GetSetters();

  DCHECK(metadata != nullptr);
  const auto& keys = metadata->keys();
  const auto& values = metadata->values();

  for (size_t i = 0; i < keys.size(); ++i) {
    auto it = setters.find(keys[i]);
    if (it != setters.end()) {
      ARROW_RETURN_NOT_OK(it->second(values[i], req));
    } else if (IsConditionWriteKey(keys[i])) {
      // condition write header
      req->SetAdditionalCustomHeaderValue(ToAwsString(keys[i]), ToAwsString(values[i]));
    } else if constexpr (HasAddMetadata<ObjectRequest>::value) {
      // custom metadata
      req->AddMetadata(ToAwsString(keys[i]), ToAwsString(values[i]));
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

template <typename ErrorType>
arrow::Status ObjectReadErrorToStatus(const S3Path& path,
                                      int64_t start,
                                      int64_t length,
                                      const Aws::Client::AWSError<ErrorType>& error,
                                      S3ErrorProvenance provenance,
                                      S3ResourceKind resource_kind = S3ResourceKind::Object) {
  provenance.resource_kind = resource_kind;
  return ErrorToStatus(fmt::format("When reading {} bytes at offset {} for key '{}' in bucket '{}': ", length, start,
                                   BoundedDetail(path.key, 256), BoundedDetail(path.bucket, 128)),
                       "GetObject", error, provenance);
}

std::optional<int64_t> ParseContentRangeSize(const Aws::String& content_range) {
  int64_t first = 0;
  int64_t last = 0;
  int64_t size = 0;
  int parsed = 0;

  // Expected format: "bytes {first}-{last}/{size}".
  if (std::sscanf(content_range.c_str(), "bytes %" SCNd64 "-%" SCNd64 "/%" SCNd64 "%n", &first, &last, &size,
                  &parsed) == 3 &&
      parsed == static_cast<int>(content_range.size()) && first >= 0 && last >= first && size >= 0) {
    return size;
  }

  // Expected format: "bytes */{size}".
  parsed = 0;
  if (std::sscanf(content_range.c_str(), "bytes */%" SCNd64 "%n", &size, &parsed) == 1 &&
      parsed == static_cast<int>(content_range.size()) && size >= 0) {
    return size;
  }

  return std::nullopt;
}

template <typename ObjectResult>
std::optional<int64_t> GetObjectSizeFromReadResult(const ObjectResult& result, int64_t position) {
  auto content_length = ParseContentRangeSize(result.GetContentRange());
  if (!content_length && position == 0 && result.GetContentRange().empty()) {
    content_length = result.GetContentLength();
  }
  return content_length;
}

Aws::IOStreamFactory AwsWriteableStreamFactory(void* data, int64_t nbytes) {
  return [=]() { return Aws::New<StringViewStream>("", data, nbytes); };
}

arrow::Result<S3Model::GetObjectResult> GetObjectRange(Aws::S3::S3Client* client,
                                                       const S3Path& path,
                                                       int64_t start,
                                                       int64_t length,
                                                       void* out,
                                                       S3ErrorProvenance provenance) {
  S3Model::GetObjectRequest req;
  req.SetBucket(ToAwsString(path.bucket));
  req.SetKey(ToAwsString(path.key));
  req.SetRange(ToAwsString(FormatRange(start, length)));
  req.SetResponseStreamFactory(AwsWriteableStreamFactory(out, length));
  auto outcome = client->GetObject(req);
  if (outcome.IsSuccess()) {
    return std::move(outcome).GetResultWithOwnership();
  }
  return ObjectReadErrorToStatus(path, start, length, outcome.GetError(), provenance);
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

  // Get custom metadata
  const auto& metadata_map = result.GetMetadata();
  for (const auto& [key, val] : metadata_map) {
    if (!val.empty()) {
      push(std::string(FromAwsString(key)), val);
    }
  }

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
    const auto content_length = GetCachedContentLength();
    if (content_length != kNoSize) {
      DCHECK_GE(content_length, 0);
    }
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
    const auto content_length = GetCachedContentLength();
    if (content_length != kNoSize && position > content_length) {
      return arrow::Status::IOError("Cannot ", action, " past end of file");
    }
    return arrow::Status::OK();
  }

  // RandomAccessFile APIs

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    ARROW_RETURN_NOT_OK(CheckClosed());
    ARROW_RETURN_NOT_OK(EnsureHeadObject(/*need_metadata=*/true));
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    return metadata_;
  }

  Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    return Future<std::shared_ptr<const arrow::KeyValueMetadata>>::MakeFinished(ReadMetadata());
  }

  arrow::Status Close() override {
    holder_ = nullptr;
    closed_ = true;
    return arrow::Status::OK();
  }

  bool closed() const override { return closed_; }

  arrow::Result<int64_t> Tell() const override {
    ARROW_RETURN_NOT_OK(CheckClosed());
    return pos_;
  }

  arrow::Result<int64_t> GetSize() override {
    ARROW_RETURN_NOT_OK(CheckClosed());
    ARROW_RETURN_NOT_OK(EnsureHeadObject(/*need_metadata=*/false));
    return GetCachedContentLength();
  }

  arrow::Status Seek(int64_t position) override {
    ARROW_RETURN_NOT_OK(CheckClosed());
    ARROW_RETURN_NOT_OK(CheckPosition(position, "seek"));

    pos_ = position;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    FIU_RETURN_ON(FIUKEY_S3FS_READER_READAT_FAIL,
                  MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL),
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL)));
    ARROW_RETURN_NOT_OK(CheckClosed());
    ARROW_ASSIGN_OR_RAISE(nbytes, GetReadSize(position, nbytes));
    if (nbytes == 0) {
      return 0;
    }

    // Read the desired range of bytes
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());
    ARROW_ASSIGN_OR_RAISE(S3Model::GetObjectResult result,
                          GetObjectRange(client_lock.get(), path_, position, nbytes, out, ProvenanceOf(holder_)));

    // The response body has already been written into the caller-provided
    // PreallocatedStreamBuf. If the requested range extends past EOF, the
    // buffer capacity is still `nbytes`, but the server returns a shorter 206
    // body. Use the response Content-Length as the actual bytes read instead
    // of consuming the stream and trusting gcount().
    const auto response_content_length = result.GetContentLength();
    if (response_content_length < 0 || response_content_length > nbytes) {
      return arrow::Status::IOError("Unexpected GetObject Content-Length ", response_content_length,
                                    " for range read of ", nbytes, " bytes");
    }

    const int64_t bytes_read = response_content_length;
    CacheContentLengthFromRead(result, position, bytes_read);
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    FIU_RETURN_ON(FIUKEY_S3FS_READER_READAT_FAIL,
                  MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL),
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL)));
    ARROW_RETURN_NOT_OK(CheckClosed());
    ARROW_ASSIGN_OR_RAISE(nbytes, GetReadSize(position, nbytes));

    ARROW_ASSIGN_OR_RAISE(auto buf, AllocateResizableBuffer(nbytes, io_context_.pool()));
    if (nbytes > 0) {
      ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(position, nbytes, buf->mutable_data()));
      DCHECK_LE(bytes_read, nbytes);
      ARROW_RETURN_NOT_OK(buf->Resize(bytes_read));
    }
    // R build with openSUSE155 requires an explicit shared_ptr construction
    return std::shared_ptr<Buffer>(std::move(buf));
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    FIU_RETURN_ON(FIUKEY_S3FS_READER_READ_FAIL,
                  MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READ_FAIL),
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READ_FAIL)));
    ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(pos_, nbytes, out));
    pos_ += bytes_read;
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) override {
    FIU_RETURN_ON(FIUKEY_S3FS_READER_READ_FAIL,
                  MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READ_FAIL),
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READ_FAIL)));
    ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
    pos_ += buffer->size();
    return buffer;
  }

  arrow::Result<int64_t> GetReadSize(int64_t position, int64_t nbytes) const {
    if (position < 0) {
      return arrow::Status::Invalid("Cannot read from negative position");
    }
    if (nbytes < 0) {
      return arrow::Status::Invalid("Cannot read negative number of bytes");
    }
    const auto content_length = GetCachedContentLength();
    if (content_length != kNoSize) {
      if (position > content_length) {
        return arrow::Status::IOError("Cannot read past end of file");
      }
      nbytes = std::min(nbytes, content_length - position);
    }
    return nbytes;
  }

  arrow::Status EnsureHeadObject(bool need_metadata) {
    if (GetCachedContentLength() != kNoSize && (!need_metadata || HasCachedMetadata())) {
      return arrow::Status::OK();
    }

    S3Model::HeadObjectRequest req;
    req.SetBucket(ToAwsString(path_.bucket));
    req.SetKey(ToAwsString(path_.key));

    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());
    auto outcome = client_lock.Move()->HeadObject(req);
    if (!outcome.IsSuccess()) {
      // HEAD responses carry no body, so a missing bucket and a missing key
      // both arrive as a generic 404 here. Report key-level ENOENT without
      // probing further: disambiguation probes cost an extra RPC on every
      // miss and are deliberately not done.
      if (IsObjectNotFound(outcome.GetError())) {
        return PathNotFound(path_);
      }
      return ErrorToStatus(
          std::forward_as_tuple("When reading information for key '", path_.key, "' in bucket '", path_.bucket, "': "),
          "HeadObject", outcome.GetError(), ProvenanceOf(holder_));
    }

    auto content_length = outcome.GetResult().GetContentLength();
    DCHECK_GE(content_length, 0);
    auto metadata = GetObjectMetadata(outcome.GetResult());
    {
      std::lock_guard<std::mutex> lock(metadata_mutex_);
      metadata_ = std::move(metadata);
    }
    SetCachedContentLength(content_length);
    return arrow::Status::OK();
  }

  template <typename ObjectResult>
  void CacheContentLengthFromRead(const ObjectResult& result, int64_t position, int64_t bytes_read) {
    auto content_length = GetObjectSizeFromReadResult(result, position);
    if (!content_length) {
      return;
    }

    DCHECK_LE(position + bytes_read, *content_length);
    SetCachedContentLengthIfAbsent(*content_length);
  }

  int64_t GetCachedContentLength() const { return content_length_.load(std::memory_order_acquire); }

  void SetCachedContentLength(int64_t content_length) {
    content_length_.store(content_length, std::memory_order_release);
  }

  void SetCachedContentLengthIfAbsent(int64_t content_length) {
    int64_t expected = kNoSize;
    content_length_.compare_exchange_strong(expected, content_length, std::memory_order_acq_rel,
                                            std::memory_order_acquire);
  }

  bool HasCachedMetadata() const {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    return metadata_ != nullptr;
  }

  protected:
  std::shared_ptr<S3ClientHolder> holder_;
  const arrow::io::IOContext io_context_;
  S3Path path_;

  bool closed_ = false;
  int64_t pos_ = 0;
  mutable std::mutex metadata_mutex_;
  std::atomic<int64_t> content_length_{kNoSize};
  std::shared_ptr<const arrow::KeyValueMetadata> metadata_;
};

#ifdef WITH_CRT
class ObjectCrtInputFile final : public arrow::io::RandomAccessFile, public NonBlockingReadAtFile {
  protected:
  struct ReadState {
    explicit ReadState(int64_t size) : content_length(size) {}

    std::atomic<int64_t> content_length;
  };

  public:
  ObjectCrtInputFile(std::shared_ptr<S3CrtClientHolder> holder,
                     const arrow::io::IOContext& io_context,
                     const S3Path& path,
                     int64_t size = kNoSize)
      : holder_(std::move(holder)),
        io_context_(io_context),
        path_(path),
        read_state_(std::make_shared<ReadState>(size)) {}

  arrow::Status Init() {
    const auto content_length = GetCachedContentLength();
    if (content_length != kNoSize) {
      DCHECK_GE(content_length, 0);
    }
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
    const auto content_length = GetCachedContentLength();
    if (content_length != kNoSize && position > content_length) {
      return arrow::Status::IOError("Cannot ", action, " past end of file");
    }
    return arrow::Status::OK();
  }

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    ARROW_RETURN_NOT_OK(CheckClosed());
    ARROW_RETURN_NOT_OK(EnsureHeadObject(/*need_metadata=*/true));
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    return metadata_;
  }

  Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    return Future<std::shared_ptr<const arrow::KeyValueMetadata>>::MakeFinished(ReadMetadata());
  }

  arrow::Status Close() override {
    holder_ = nullptr;
    closed_ = true;
    return arrow::Status::OK();
  }

  bool closed() const override { return closed_; }

  arrow::Result<int64_t> Tell() const override {
    ARROW_RETURN_NOT_OK(CheckClosed());
    return pos_;
  }

  arrow::Result<int64_t> GetSize() override {
    ARROW_RETURN_NOT_OK(CheckClosed());
    ARROW_RETURN_NOT_OK(EnsureHeadObject(/*need_metadata=*/false));
    return GetCachedContentLength();
  }

  arrow::Status Seek(int64_t position) override {
    ARROW_RETURN_NOT_OK(CheckClosed());
    ARROW_RETURN_NOT_OK(CheckPosition(position, "seek"));

    pos_ = position;
    return arrow::Status::OK();
  }

  Future<int64_t> ReadAtAsyncInto(int64_t position, int64_t nbytes, uint8_t* out) override {
    FIU_RETURN_ON(FIUKEY_S3FS_READER_READAT_FAIL,
                  FailedReadFuture(MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                                   fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL),
                                                   fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL))));
    auto status = CheckClosed();
    if (!status.ok()) {
      return FailedReadFuture(status);
    }

    auto maybe_read_size = GetReadSize(position, nbytes);
    if (!maybe_read_size.ok()) {
      return FailedReadFuture(maybe_read_size.status());
    }
    nbytes = maybe_read_size.ValueOrDie();
    if (nbytes == 0) {
      return Future<int64_t>::MakeFinished(0);
    }

    auto maybe_client_lease = holder_->Acquire();
    if (!maybe_client_lease.ok()) {
      return FailedReadFuture(maybe_client_lease.status());
    }

    auto ctx = std::make_shared<AsyncReadContext>();
    ctx->future = Future<int64_t>::Make();
    ctx->client_lease = std::move(maybe_client_lease).ValueOrDie();
    ctx->read_state = read_state_;
    ctx->metrics = holder_->GetMetrics();
    ctx->position = position;
    ctx->nbytes = nbytes;
    ctx->request.SetBucket(ToAwsString(path_.bucket));
    ctx->request.SetKey(ToAwsString(path_.key));
    ctx->request.SetRange(ToAwsString(FormatRange(position, nbytes)));
    ctx->request.SetResponseStreamFactory(AwsWriteableStreamFactory(out, nbytes));

    ctx->metrics->IncrementReadCount();
    ctx->client_lease->GetObjectAsync(
        ctx->request,
        [ctx](const Aws::S3Crt::S3CrtClient*, const S3CrtModel::GetObjectRequest&, S3CrtModel::GetObjectOutcome outcome,
              const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) mutable {
          if (!outcome.IsSuccess()) {
            ctx->metrics->IncrementFailedCount();
            const auto provenance = ProvenanceOf(ctx->self->holder_);
            ctx->future.MarkFinished(arrow::Result<int64_t>(
                ObjectReadErrorToStatus(ctx->self->path_, ctx->position, ctx->nbytes, outcome.GetError(), provenance)));
            return;
          }

          const auto& result = outcome.GetResult();
          const auto response_content_length = result.GetContentLength();
          if (response_content_length < 0 || response_content_length > ctx->nbytes) {
            ctx->metrics->IncrementFailedCount();
            ctx->future.MarkFinished(arrow::Result<int64_t>(
                arrow::Status::IOError("Unexpected GetObject Content-Length ", response_content_length,
                                       " for range read of ", ctx->nbytes, " bytes")));
            return;
          }

          const int64_t bytes_read = response_content_length;
          auto content_length = GetObjectSizeFromReadResult(result, ctx->position);
          if (content_length) {
            DCHECK_LE(ctx->position + bytes_read, *content_length);
            int64_t expected = kNoSize;
            ctx->read_state->content_length.compare_exchange_strong(
                expected, *content_length, std::memory_order_acq_rel, std::memory_order_acquire);
          }
          if (bytes_read > 0) {
            ctx->metrics->IncrementReadBytes(bytes_read);
          }
          ctx->future.MarkFinished(bytes_read);
        });
    // Keep executor selection at the caller-owned continuation boundary.
    return ctx->future;
  }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    FIU_RETURN_ON(FIUKEY_S3FS_READER_READAT_FAIL,
                  MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL),
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL)));
    return ReadAtAsyncInto(position, nbytes, reinterpret_cast<uint8_t*>(out)).result();
  }

  arrow::Result<std::shared_ptr<Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    FIU_RETURN_ON(FIUKEY_S3FS_READER_READAT_FAIL,
                  MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL),
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL)));
    ARROW_RETURN_NOT_OK(CheckClosed());
    ARROW_ASSIGN_OR_RAISE(nbytes, GetReadSize(position, nbytes));

    ARROW_ASSIGN_OR_RAISE(auto buf, AllocateResizableBuffer(nbytes, io_context_.pool()));
    if (nbytes > 0) {
      ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(position, nbytes, buf->mutable_data()));
      DCHECK_LE(bytes_read, nbytes);
      ARROW_RETURN_NOT_OK(buf->Resize(bytes_read));
    }
    return std::shared_ptr<Buffer>(std::move(buf));
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    FIU_RETURN_ON(FIUKEY_S3FS_READER_READ_FAIL,
                  MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READ_FAIL),
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READ_FAIL)));
    ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(pos_, nbytes, out));
    pos_ += bytes_read;
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) override {
    FIU_RETURN_ON(FIUKEY_S3FS_READER_READ_FAIL,
                  MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READ_FAIL),
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READ_FAIL)));
    ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
    pos_ += buffer->size();
    return buffer;
  }

  Future<std::shared_ptr<Buffer>> ReadAsync(const arrow::io::IOContext& io_context,
                                            int64_t position,
                                            int64_t nbytes) override {
    FIU_RETURN_ON(
        FIUKEY_S3FS_READER_READAT_FAIL,
        FailedBufferFuture(MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                           fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL),
                                           fmt::format("Injected fault: {}", FIUKEY_S3FS_READER_READAT_FAIL))));
    auto status = CheckClosed();
    if (!status.ok()) {
      return FailedBufferFuture(status);
    }

    auto maybe_read_size = GetReadSize(position, nbytes);
    if (!maybe_read_size.ok()) {
      return FailedBufferFuture(maybe_read_size.status());
    }
    nbytes = maybe_read_size.ValueOrDie();

    auto maybe_buf = AllocateResizableBuffer(nbytes, io_context.pool());
    if (!maybe_buf.ok()) {
      return FailedBufferFuture(maybe_buf.status());
    }
    auto buf = std::move(maybe_buf).ValueOrDie();
    auto* out = buf->mutable_data();

    return ReadAtAsyncInto(position, nbytes, out)
        .Then([buf = std::move(buf), nbytes](int64_t bytes_read) mutable -> arrow::Result<std::shared_ptr<Buffer>> {
          DCHECK_LE(bytes_read, nbytes);
          ARROW_RETURN_NOT_OK(buf->Resize(bytes_read));
          return std::shared_ptr<Buffer>(std::move(buf));
        });
  }

  std::vector<Future<std::shared_ptr<Buffer>>> ReadManyAsync(const arrow::io::IOContext& io_context,
                                                             const std::vector<arrow::io::ReadRange>& ranges) override {
    std::vector<Future<std::shared_ptr<Buffer>>> futures;
    futures.reserve(ranges.size());
    for (const auto& range : ranges) {
      futures.push_back(ReadAsync(io_context, range.offset, range.length));
    }
    return futures;
  }

  arrow::Result<int64_t> GetReadSize(int64_t position, int64_t nbytes) const {
    if (position < 0) {
      return arrow::Status::Invalid("Cannot read from negative position");
    }
    if (nbytes < 0) {
      return arrow::Status::Invalid("Cannot read negative number of bytes");
    }
    const auto content_length = GetCachedContentLength();
    if (content_length != kNoSize) {
      if (position > content_length) {
        return arrow::Status::IOError("Cannot read past end of file");
      }
      nbytes = std::min(nbytes, content_length - position);
    }
    return nbytes;
  }

  arrow::Status EnsureHeadObject(bool need_metadata) {
    if (GetCachedContentLength() != kNoSize && (!need_metadata || HasCachedMetadata())) {
      return arrow::Status::OK();
    }

    S3CrtModel::HeadObjectRequest req;
    req.SetBucket(ToAwsString(path_.bucket));
    req.SetKey(ToAwsString(path_.key));

    ARROW_ASSIGN_OR_RAISE(auto client_lease, holder_->Acquire());
    auto outcome = client_lease->HeadObject(req);
    if (!outcome.IsSuccess()) {
      // Same bucket/key split as the non-CRT read path: a 404 that is not a
      // typed NO_SUCH_BUCKET is reported as key-level ENOENT, with no extra
      // disambiguation RPC.
      if (outcome.GetError().GetResponseCode() == Aws::Http::HttpResponseCode::NOT_FOUND &&
          static_cast<Aws::S3::S3Errors>(outcome.GetError().GetErrorType()) != Aws::S3::S3Errors::NO_SUCH_BUCKET) {
        return PathNotFound(path_);
      }
      return ErrorToStatus(
          std::forward_as_tuple("When reading information for key '", path_.key, "' in bucket '", path_.bucket, "': "),
          "HeadObject", outcome.GetError(), ProvenanceOf(holder_));
    }

    auto content_length = outcome.GetResult().GetContentLength();
    DCHECK_GE(content_length, 0);
    auto metadata = GetObjectMetadata(outcome.GetResult());
    {
      std::lock_guard<std::mutex> lock(metadata_mutex_);
      metadata_ = std::move(metadata);
    }
    SetCachedContentLength(content_length);
    return arrow::Status::OK();
  }

  int64_t GetCachedContentLength() const { return read_state_->content_length.load(std::memory_order_acquire); }

  void SetCachedContentLength(int64_t content_length) {
    read_state_->content_length.store(content_length, std::memory_order_release);
  }

  bool HasCachedMetadata() const {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    return metadata_ != nullptr;
  }

  protected:
  static Future<int64_t> FailedReadFuture(const arrow::Status& status) {
    return Future<int64_t>::MakeFinished(arrow::Result<int64_t>(status));
  }

  static Future<std::shared_ptr<Buffer>> FailedBufferFuture(const arrow::Status& status) {
    return Future<std::shared_ptr<Buffer>>::MakeFinished(arrow::Result<std::shared_ptr<Buffer>>(status));
  }

  struct AsyncReadContext {
    // AWS CRT retains this context through the callback. Never add owning
    // references to S3CrtClient, S3CrtClientHolder, or ObjectCrtInputFile here.
    // The lease owns only operation state and a non-owning client pointer.
    Future<int64_t> future;
    S3CrtClientLease client_lease;
    S3CrtModel::GetObjectRequest request;
    std::shared_ptr<ReadState> read_state;
    std::shared_ptr<FilesystemMetrics> metrics;
    int64_t position = 0;
    int64_t nbytes = 0;
  };

  std::shared_ptr<S3CrtClientHolder> holder_;
  const arrow::io::IOContext io_context_;
  S3Path path_;

  bool closed_ = false;
  int64_t pos_ = 0;
  mutable std::mutex metadata_mutex_;
  std::shared_ptr<ReadState> read_state_;
  std::shared_ptr<const arrow::KeyValueMetadata> metadata_;
};
#endif  // WITH_CRT

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
        use_crc32c_checksum_(options.use_crc32c_checksum),
        part_upload_size_(part_size),
        writer_status_(std::make_shared<WriterStatus>()) {}

  ~CustomOutputStream() override {
    if (!closed_) {
      DiscardLocal();
    }
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
      ARROW_RETURN_NOT_OK(SetObjectMetadata(metadata, request));

      is_content_type_set = metadata->Contains("Content-Type");
    }

    if (!is_content_type_set) {
      // If we do not set anything then the SDK will default to application/xml
      // which confuses some tools (https://github.com/apache/arrow/issues/11934)
      // So we instead default to application/octet-stream which is less misleading
      if constexpr (HasSetContentType<ObjectRequest>::value) {
        request->SetContentType("application/octet-stream");
      }
    }

    return arrow::Status::OK();
  }

  std::shared_ptr<CustomOutputStream> Self() {
    return std::dynamic_pointer_cast<CustomOutputStream>(shared_from_this());
  }

  arrow::Status CreateMultipartUpload() {
    FIU_RETURN_ON(FIUKEY_S3FS_CREATE_UPLOAD_FAIL,
                  arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_S3FS_CREATE_UPLOAD_FAIL)));
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    // Initiate the multi-part upload
    S3Model::CreateMultipartUploadRequest req;
    req.SetBucket(ToAwsString(path_.bucket));
    req.SetKey(ToAwsString(path_.key));
    if (use_crc32c_checksum_) {
      req.SetChecksumAlgorithm(S3Model::ChecksumAlgorithm::CRC32C);
    }
    ARROW_RETURN_NOT_OK(SetMetadataInRequest(&req));

    auto outcome = client_lock.Move()->CreateMultipartUpload(req);
    if (!outcome.IsSuccess()) {
      return ErrorToStatus(std::forward_as_tuple("When initiating multiple part upload for key '", path_.key,
                                                 "' in bucket '", path_.bucket, "': "),
                           "CreateMultipartUpload", outcome.GetError(), ProvenanceOf(holder_, S3ResourceKind::Bucket));
    }
    multipart_upload_id_ = outcome.GetResult().GetUploadId();
    PublishUploadForAbort();

    return arrow::Status::OK();
  }

  /// Hand the upload's identity to UploadState so a completion that outlives
  /// this stream can still cancel it. Called wherever the id or the state is
  /// created, because either can come first: a delayed-open stream creates the
  /// upload long after Init(), while a non-delayed one creates it before.
  void PublishUploadForAbort() {
    if (upload_state_ == nullptr || multipart_upload_id_.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lock(upload_state_->mutex);
    upload_state_->holder = holder_;
    upload_state_->bucket = path_.bucket;
    upload_state_->key = path_.key;
    upload_state_->multipart_upload_id = multipart_upload_id_;
  }

  arrow::Status Init() {
    // If we are allowed to do delayed I/O, we can use a single request to upload the
    // data. If not, we use a multi-part upload and initiate it here to
    // sanitize that writing to the bucket is possible.
    if (!allow_delayed_open_) {
      ARROW_RETURN_NOT_OK(CreateMultipartUpload());
    }

    upload_state_ = std::make_shared<UploadState>();
    PublishUploadForAbort();
    closed_ = false;
    return arrow::Status::OK();
  }

  arrow::Status Abort() override {
    // Abort after a successful Close is an idempotent no-op. Check the healthy
    // closed state before BeginDiscard(), otherwise a completed stream becomes
    // Cancelled and a later idempotent Close returns the wrong result.
    if (closed_ && writer_status_->ok()) {
      return arrow::Status::OK();
    }
    writer_status_->BeginDiscard();
    // Release the server-side upload before dropping the local handle.
    //
    // This is the one piece of I/O a giving-up path still owes the store, and
    // it is not diagnosis: the parts we already uploaded are a resource THIS
    // stream created and nobody else can name. They do not appear in
    // ListObjectsV2, so neither a bucket listing, a usage report, nor any GC
    // that walks object keys will ever find them again -- only
    // ListMultipartUploads or an AbortIncompleteMultipartUpload lifecycle rule
    // will, and the second one is not something this library can assume a
    // deployment configured.
    //
    // Best effort by construction: the outcome is logged and thrown away, so
    // the caller's own failure is what gets reported, never this one. Abort()
    // must stay usable from a path that is already handling an error.
    // Mark first, so a part completing while this runs knows a cancel is owed.
    // Whether one is still in flight decides who finishes the job, not whether
    // it gets finished.
    bool uploads_still_in_flight = false;
    if (upload_state_ != nullptr) {
      std::lock_guard<std::mutex> lock(upload_state_->mutex);
      upload_state_->abort_requested = true;
      uploads_still_in_flight = upload_state_->uploads_in_progress > 0;
    }

    if (!uploads_still_in_flight) {
      // Nothing can land behind us, so this attempt is the whole job. Consume
      // the shared identity through the same exactly-once path used by the last
      // completion. A repeated Abort() may race that completion after it drops
      // the state lock, and must not clear the identity before either caller
      // has issued the remote abort.
      if (upload_state_ != nullptr) {
        AbortRecordedUpload(upload_state_);
      } else {
        (void)AbortMultipartUploadIfCreated();
      }
    }
    // Otherwise the cancel is deliberately deferred to the completion that
    // takes uploads_in_progress to zero (AbortRecordedUpload). Aborting now as
    // well would be wasted: the parts still on the wire would land afterwards
    // and orphan themselves anyway, which is exactly the leak this exists to
    // close. Waiting for them here is not an option either -- R2.6 forbids a
    // failure path from waiting on children.
    DiscardLocal();
    return arrow::Status::OK();
  }

  /// Cancel an upload using only what UploadState holds, so it works from a
  /// completion callback running after the stream is gone.
  ///
  /// Takes the identity out of the state under the lock and clears it, which
  /// makes the cancel happen at most once, then issues the request with the
  /// lock released. Best effort: the outcome is logged and dropped, because
  /// whoever triggered this is already handling a failure of their own.
  static void AbortRecordedUpload(const std::shared_ptr<UploadState>& state) noexcept {
    std::shared_ptr<S3ClientHolder> holder;
    std::string bucket;
    std::string key;
    Aws::String upload_id;
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      if (!state->abort_requested || state->holder == nullptr || state->multipart_upload_id.empty()) {
        return;
      }
      holder = std::move(state->holder);
      bucket = std::move(state->bucket);
      key = std::move(state->key);
      upload_id = std::move(state->multipart_upload_id);
      state->holder = nullptr;
      state->multipart_upload_id.clear();
    }

    bool issued = false;
    try {
      auto client_lock = holder->Lock();
      if (!client_lock.ok()) {
        LOG_STORAGE_WARNING_ << "Cannot abort multipart upload for key '" << key << "' in bucket '" << bucket
                             << "' right now: " << client_lock.status().ToString();
      } else {
        S3Model::AbortMultipartUploadRequest req;
        req.SetBucket(ToAwsString(bucket));
        req.SetKey(ToAwsString(key));
        req.SetUploadId(upload_id);

        auto outcome = std::move(client_lock).ValueOrDie().Move()->AbortMultipartUpload(req);
        issued = true;
        if (!outcome.IsSuccess()) {
          LOG_STORAGE_WARNING_ << "Failed to abort multipart upload for key '" << key << "' in bucket '" << bucket
                               << "', parts may remain until a lifecycle rule removes them";
        }
      }
    } catch (...) {
      // Falls through to the hand-back below.
    }

    if (issued) {
      return;
    }

    // The request never went out, so this call is not the one that released the
    // upload. Taking the identity is what makes the cancel happen at most once,
    // but keeping it after failing to use it is what turned a transient failure
    // into a permanent leak: every later Abort() early-returns on an empty
    // identity, so nothing could ever name the upload again. Hand it back. A
    // duplicate abort, should one ever race, is harmless -- S3 answers
    // NoSuchUpload.
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->holder == nullptr && state->multipart_upload_id.empty()) {
      state->holder = std::move(holder);
      state->bucket = std::move(bucket);
      state->key = std::move(key);
      state->multipart_upload_id = std::move(upload_id);
    }
  }

  /// Cancel the multipart upload this stream created, if it created one.
  ///
  /// Returns the failure for logging and tests; Abort() deliberately discards
  /// it. Like arrow's own ObjectOutputStream::Abort, this does not wait for
  /// background part uploads that are still in flight -- one that lands after
  /// the abort is orphaned again, and only a second abort or a lifecycle rule
  /// clears it. Waiting here would mean issuing more I/O while handling a
  /// failure, which is exactly what R2.7 forbids; releasing what we already
  /// hold is not the same thing as waiting on it.
  arrow::Status AbortMultipartUploadIfCreated() {
    // Deliberately NOT gated on closed_. A failed Write/Flush/Close marks the
    // stream closed and drops its buffers, and that is precisely the case an
    // abort has to still work for. What proves there is nothing to release is
    // holder_ == nullptr, which only a SUCCESSFUL close produces
    // (CleanupAfterClose) -- a completed upload has no parts left to cancel.
    if (!IsMultipartCreated() || holder_ == nullptr) {
      return arrow::Status::OK();
    }
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    S3Model::AbortMultipartUploadRequest req;
    req.SetBucket(ToAwsString(path_.bucket));
    req.SetKey(ToAwsString(path_.key));
    req.SetUploadId(multipart_upload_id_);

    auto outcome = client_lock.Move()->AbortMultipartUpload(req);
    if (!outcome.IsSuccess()) {
      auto status = ErrorToStatus(std::forward_as_tuple("When aborting multiple part upload for key '", path_.key,
                                                        "' in bucket '", path_.bucket, "': "),
                                  "AbortMultipartUpload", outcome.GetError(),
                                  ProvenanceOf(holder_, S3ResourceKind::MultipartUpload));
      LOG_STORAGE_WARNING_ << "Failed to abort multipart upload, parts may remain until a lifecycle rule "
                              "removes them: "
                           << status.ToString();
      return status;
    }
    return arrow::Status::OK();
  }

  /// Drop the buffered part and refuse further work, but KEEP the upload id and
  /// the client.
  ///
  /// Used by the failure paths. Forgetting the upload id here is what made a
  /// failed Close leak: the writer was terminal, the caller went on to destroy
  /// it, and by then nothing knew which upload to cancel. R2.7 says failure
  /// handling must not issue I/O -- it does not say failure handling may throw
  /// away the only handle to a resource we still own. The release happens later,
  /// in Abort(), where abandoning is explicit.
  void DiscardBuffersAfterFailure() {
    current_part_.reset();
    current_part_buffer_.reset();
    current_part_size_ = 0;
    closed_ = true;
  }

  /// Full local release: also forgets the upload. Only correct once the upload
  /// has been cancelled (Abort) or completed, or when nothing can act on it any
  /// more (destruction).
  void DiscardLocal() {
    DiscardBuffersAfterFailure();
    multipart_upload_id_.clear();
    holder_ = nullptr;
  }

  // OutputStream interface

  bool ShouldBeMultipartUpload() const { return pos_ > part_upload_size_ - 1 || !allow_delayed_open_; }

  bool IsMultipartCreated() const { return !multipart_upload_id_.empty(); }

  arrow::Status EnsureReadyToFlushFromClose() {
    if (ShouldBeMultipartUpload()) {
      if (current_part_) {
        // Upload last part
        ARROW_RETURN_NOT_OK(CommitCurrentPart());
      }

      // S3 mandates at least one part, upload an empty one if necessary
      if (part_number_ == 1) {
        ARROW_RETURN_NOT_OK(UploadPart("", 0));
      }
    } else {
      ARROW_RETURN_NOT_OK(UploadUsingSingleRequest());
    }

    return arrow::Status::OK();
  }

  arrow::Status CleanupAfterClose() {
    // A successful close has completed the multipart upload. Forget the
    // shared abort identity as well as the stream-local one, so a later
    // idempotent Abort() cannot try to cancel an already committed upload.
    if (upload_state_ != nullptr) {
      std::lock_guard<std::mutex> lock(upload_state_->mutex);
      upload_state_->holder = nullptr;
      upload_state_->multipart_upload_id.clear();
    }
    holder_ = nullptr;
    closed_ = true;
    return arrow::Status::OK();
  }

  arrow::Status FinishPartUploadAfterFlush() {
    FIU_RETURN_ON(FIUKEY_S3FS_COMPLETE_UPLOAD_FAIL,
                  arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_S3FS_COMPLETE_UPLOAD_FAIL)));
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

    ARROW_RETURN_NOT_OK(SetMetadataInRequest(&req));

    auto outcome = client_lock.get()->CompleteMultipartUploadWithErrorFixup(std::move(req));
    if (!outcome.IsSuccess()) {
      return ErrorToStatus(fmt::format("When completing multiple part upload for key '{}' in bucket '{}': ",
                                       BoundedDetail(path_.key, 256), BoundedDetail(path_.bucket, 128)),
                           "CompleteMultipartUpload", outcome.GetError(),
                           ProvenanceOf(holder_, S3ResourceKind::MultipartUpload));
    }

    return arrow::Status::OK();
  }

  arrow::Status CleanupIfFailed(Status status) {
    if (!status.ok()) {
      writer_status_->ObserveFailure(status);
      DiscardBuffersAfterFailure();
      return writer_status_->status();
    }
    return arrow::Status::OK();
  }

  arrow::Status Close() override {
    if (!writer_status_->status().ok()) {
      return CloseAfterFailure();
    }
    ARROW_RETURN_NOT_OK(writer_status_->Check());
    return writer_status_->Fail(CloseImpl());
  }

  arrow::Status CloseAfterFailure(arrow::Status primary_status = arrow::Status::OK()) {
    writer_status_->ObserveFailure(primary_status);
    auto first_status = writer_status_->status();
    DiscardBuffersAfterFailure();
    return first_status;
  }

  arrow::Status CloseImpl() {
    if (!writer_status_->status().ok()) {
      return CloseAfterFailure();
    }
    if (closed_) {
      return arrow::Status::OK();
    }

    FIU_RETURN_ON(FIUKEY_S3FS_WRITER_CLOSE_FAIL,
                  CleanupIfFailed(MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_WRITER_CLOSE_FAIL),
                                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_WRITER_CLOSE_FAIL))));

    ARROW_RETURN_NOT_OK(CleanupIfFailed(EnsureReadyToFlushFromClose()));
    ARROW_RETURN_NOT_OK(CleanupIfFailed(FlushImpl()));

    if (IsMultipartCreated()) {
      ARROW_RETURN_NOT_OK(CleanupIfFailed(FinishPartUploadAfterFlush()));
    }

    return CleanupAfterClose();
  }

  Future<> CloseAsync() override {
    if (!writer_status_->status().ok()) {
      return CloseAfterFailureAsync();
    }
    auto check_status = writer_status_->Check();
    if (!check_status.ok()) {
      return check_status;
    }
    auto future = CloseAsyncImpl();
    return future.Then(
        []() { return arrow::Status::OK(); },
        [writer_status = writer_status_](const arrow::Status& status) { return writer_status->Fail(status); });
  }

  Future<> CloseAsyncImpl() {
    if (!writer_status_->status().ok()) {
      return CloseAfterFailureAsync();
    }
    if (closed_) {
      return arrow::Status::OK();
    }

    FIU_RETURN_ON(
        FIUKEY_S3FS_WRITER_CLOSE_FAIL,
        CloseAfterFailureAsync(MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                               fmt::format("Injected fault: {}", FIUKEY_S3FS_WRITER_CLOSE_FAIL),
                                               fmt::format("Injected fault: {}", FIUKEY_S3FS_WRITER_CLOSE_FAIL))));

    auto ready_status = EnsureReadyToFlushFromClose();
    if (!ready_status.ok()) {
      return CloseAfterFailureAsync(std::move(ready_status));
    }

    return FlushAsyncImpl().Then(
        [self = Self()]() {
          if (self->IsMultipartCreated()) {
            ARROW_RETURN_NOT_OK(self->CleanupIfFailed(self->FinishPartUploadAfterFlush()));
          }
          return self->CleanupAfterClose();
        },
        [self = Self()](const arrow::Status& status) { return self->CloseAfterFailureAsync(status); });
  }

  bool closed() const override { return closed_; }

  arrow::Result<int64_t> Tell() const override {
    ARROW_RETURN_NOT_OK(writer_status_->Check());
    if (closed_) {
      return writer_status_->Fail(arrow::Status::Invalid("Operation on closed stream"));
    }
    return pos_;
  }

  arrow::Status Write(const std::shared_ptr<Buffer>& buffer) override {
    return DoWrite(buffer->data(), buffer->size(), buffer);
  }

  arrow::Status Write(const void* data, int64_t nbytes) override { return DoWrite(data, nbytes); }

  arrow::Status DoWrite(const void* data, int64_t nbytes, const std::shared_ptr<Buffer>& owned_buffer = nullptr) {
    ARROW_RETURN_NOT_OK(writer_status_->Check());
    auto status = DoWriteImpl(data, nbytes, owned_buffer);
    if (!status.ok()) {
      return writer_status_->Fail(std::move(status));
    }
    // A background part may have failed while DoWriteImpl was preparing or
    // submitting this write. Never report success after that failure has
    // already made the writer terminal.
    return writer_status_->Check();
  }

  arrow::Status DoWriteImpl(const void* data, int64_t nbytes, const std::shared_ptr<Buffer>& owned_buffer = nullptr) {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed stream");
    }

    FIU_RETURN_ON(FIUKEY_S3FS_WRITER_WRITE_FAIL,
                  MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_WRITER_WRITE_FAIL),
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_WRITER_WRITE_FAIL)));

    const auto* data_ptr = reinterpret_cast<const int8_t*>(data);
    auto advance_ptr = [&data_ptr, &nbytes](const int64_t offset) {
      data_ptr += offset;
      nbytes -= offset;
    };

    // Handle case where we have some bytes buffered from prior calls.
    if (current_part_size_ > 0) {
      // Try to fill current buffer
      const int64_t to_copy = std::min(nbytes, part_upload_size_ - current_part_size_);
      ARROW_RETURN_NOT_OK(current_part_->Write(data_ptr, to_copy));
      current_part_size_ += to_copy;
      advance_ptr(to_copy);
      pos_ += to_copy;

      // If buffer isn't full, break
      if (current_part_size_ < part_upload_size_) {
        return arrow::Status::OK();
      }

      ARROW_RETURN_NOT_OK(CommitCurrentPart());
    }

    // We can upload chunks without copying them into a buffer
    while (nbytes >= part_upload_size_) {
      ARROW_RETURN_NOT_OK(UploadPart(data_ptr, part_upload_size_));
      advance_ptr(part_upload_size_);
      pos_ += part_upload_size_;
    }

    // Buffer remaining bytes
    if (nbytes > 0) {
      current_part_size_ = nbytes;
      ARROW_ASSIGN_OR_RAISE(current_part_buffer_,
                            arrow::AllocateResizableBuffer(part_upload_size_, io_context_.pool()));
      current_part_ = std::make_shared<arrow::io::BufferOutputStream>(current_part_buffer_);
      ARROW_RETURN_NOT_OK(current_part_->Write(data_ptr, current_part_size_));
      pos_ += current_part_size_;
    }

    return arrow::Status::OK();
  }

  arrow::Status Flush() override {
    if (!writer_status_->status().ok()) {
      return writer_status_->status();
    }
    ARROW_RETURN_NOT_OK(writer_status_->Check());
    return writer_status_->Fail(FlushImpl());
  }

  arrow::Status FlushImpl() {
    if (!writer_status_->status().ok()) {
      return writer_status_->status();
    }
    auto fut = FlushAsyncImpl();
    return fut.status();
  }

  Future<> FlushAsync() {
    if (!writer_status_->status().ok()) {
      return writer_status_->status();
    }
    auto check_status = writer_status_->Check();
    if (!check_status.ok()) {
      return check_status;
    }
    auto future = FlushAsyncImpl();
    future.AddCallback(
        [writer_status = writer_status_](const arrow::Status& status) { (void)writer_status->Fail(status); });
    return future;
  }

  Future<> FlushAsyncImpl() {
    if (!writer_status_->status().ok()) {
      return writer_status_->status();
    }
    FIU_RETURN_ON(FIUKEY_S3FS_WRITER_FLUSH_FAIL,
                  MakeExtendError(ExtendStatusCode::StorageTransientNetwork,
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_WRITER_FLUSH_FAIL),
                                  fmt::format("Injected fault: {}", FIUKEY_S3FS_WRITER_FLUSH_FAIL)));
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed stream");
    }
    // Wait for background writes to finish
    std::unique_lock<std::mutex> lock(upload_state_->mutex);
    return upload_state_->pending_uploads_completed;
  }

  Future<> CloseAfterFailureAsync(arrow::Status primary_status = arrow::Status::OK()) {
    return CloseAfterFailure(std::move(primary_status));
  }

  // Upload-related helpers

  // Get the buffered data as a zero-copy slice, avoiding BufferOutputStream::Finish()
  // which calls ZeroPadding() and memsets (capacity - size) bytes to zero.
  arrow::Result<std::shared_ptr<Buffer>> FinishCurrentPart() {
    DCHECK(current_part_);
    DCHECK(current_part_buffer_);
    ARROW_ASSIGN_OR_RAISE(auto stream_pos, current_part_->Tell());
    if (stream_pos != current_part_size_) {
      return arrow::Status::Invalid("Buffer size mismatch: current_part_size_=", current_part_size_,
                                    " stream position=", stream_pos);
    }
    auto buf = arrow::SliceBuffer(current_part_buffer_, 0, current_part_size_);
    current_part_.reset();
    current_part_buffer_.reset();
    current_part_size_ = 0;
    return buf;
  }

  arrow::Status CommitCurrentPart() {
    if (!IsMultipartCreated()) {
      ARROW_RETURN_NOT_OK(CreateMultipartUpload());
    }

    ARROW_ASSIGN_OR_RAISE(auto buf, FinishCurrentPart());
    return UploadPart(buf);
  }

  arrow::Status UploadUsingSingleRequest() {
    std::shared_ptr<Buffer> buf;
    if (current_part_ == nullptr) {
      // In case the stream is closed directly after it has been opened without writing
      // anything, we'll have to create an empty buffer.
      buf = std::make_shared<Buffer>("");
    } else {
      ARROW_ASSIGN_OR_RAISE(buf, FinishCurrentPart());
    }

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
    req.SetContentLength(nbytes);
    if (use_crc32c_checksum_) {
      req.SetChecksumAlgorithm(S3Model::ChecksumAlgorithm::CRC32C);
    }

    if (!background_writes_) {
      // GH-45304: avoid setting a body stream if length is 0.
      // This workaround can be removed once we require AWS SDK 1.11.489 or later.
      if (nbytes != 0) {
        req.SetBody(std::make_shared<StringViewStream>(data, nbytes));
      }

      ARROW_ASSIGN_OR_RAISE(auto outcome, TriggerUploadRequest(req, holder_));

      ARROW_RETURN_NOT_OK(sync_result_callback(req, upload_state_, part_number_, outcome));
    } else {
      // (GH-45304: avoid setting a body stream if length is 0, see above)
      if (nbytes != 0) {
        // If the data isn't owned, make an immutable copy for the lifetime of the closure
        if (owned_buffer == nullptr) {
          ARROW_ASSIGN_OR_RAISE(owned_buffer, AllocateBuffer(nbytes, io_context_.pool()));
          memcpy(owned_buffer->mutable_data(), data, nbytes);
        } else {
          DCHECK_EQ(data, owned_buffer->data());
          DCHECK_EQ(nbytes, owned_buffer->size());
        }
        req.SetBody(std::make_shared<StringViewStream>(owned_buffer->data(), owned_buffer->size()));
      }

      // A previous part can complete while one large Write() is still walking
      // its input. Stop before accepting another request once that failure has
      // made the writer terminal.
      ARROW_RETURN_NOT_OK(writer_status_->Check());

      auto make_task = [&]() {
        return [owned_buffer, holder = holder_, req = std::move(req), state = upload_state_, async_result_callback,
                part_number = part_number_]() mutable -> arrow::Status {
          auto outcome_result = TriggerUploadRequest(req, holder);
          if (!outcome_result.ok()) {
            return outcome_result.status();
          }
          return async_result_callback(req, state, part_number, std::move(outcome_result).ValueOrDie());
        };
      };
      auto make_completion = [&]() {
        return [state = upload_state_, writer_status = writer_status_](const arrow::Status& status) {
          HandleUploadOutcome(state, writer_status, status);
        };
      };

      using Task = decltype(make_task());
      using Completion = decltype(make_completion());
      std::optional<Task> task;
      std::optional<Completion> completion;
      // Construct every caller-owned object that may allocate before this
      // upload owns a counter slot. Once registered, every remaining exit is
      // settled by SubmitIOWithCompletion; before it, an exception unwinds
      // without leaving a phantom upload behind, so it is left to propagate.
      task.emplace(make_task());
      completion.emplace(make_completion());

      {
        std::unique_lock<std::mutex> lock(upload_state_->mutex);
        // Serialize admission with completion failure observation. If the
        // completion got this lock first, no later part may register.
        ARROW_RETURN_NOT_OK(writer_status_->Check());
        if (upload_state_->uploads_in_progress == 0) {
          upload_state_->pending_uploads_completed = Future<>::Make();
          upload_state_->pending_completion_published = false;
        }
        ++upload_state_->uploads_in_progress;
      }

      // SubmitIOWithCompletion keeps the buffer and upload state alive and
      // settles the registered counter/future for task rejection,
      // pre-request failures, exceptions, and normal SDK outcomes.
      ARROW_RETURN_NOT_OK(SubmitIOWithCompletion(io_context_, std::move(*task), std::move(*completion)));
    }

    ++part_number_;

    return arrow::Status::OK();
  }

  static arrow::Status UploadUsingSingleRequestError(const Aws::S3::Model::PutObjectRequest& request,
                                                     const Aws::S3::Model::PutObjectOutcome& outcome,
                                                     S3ErrorProvenance provenance) {
    return ErrorToStatus(std::forward_as_tuple("When uploading object with key '", request.GetKey(), "' in bucket '",
                                               request.GetBucket(), "': "),
                         "PutObject", outcome.GetError(), provenance);
  }

  arrow::Status UploadUsingSingleRequest(const std::shared_ptr<Buffer>& buffer) {
    return UploadUsingSingleRequest(buffer->data(), buffer->size(), buffer);
  }

  arrow::Status UploadUsingSingleRequest(const void* data,
                                         int64_t nbytes,
                                         std::shared_ptr<Buffer> owned_buffer = nullptr) {
    // PutObject creates or replaces a key. A request-level 404 therefore cannot
    // mean "that key is absent"; it names the destination bucket.
    const auto provenance = ProvenanceOf(holder_, S3ResourceKind::Bucket);
    auto sync_result_callback = [provenance](const Aws::S3::Model::PutObjectRequest& request,
                                             const std::shared_ptr<UploadState>& state, int32_t part_number,
                                             const Aws::S3::Model::PutObjectOutcome& outcome) {
      if (!outcome.IsSuccess()) {
        return UploadUsingSingleRequestError(request, outcome, provenance);
      }
      return arrow::Status::OK();
    };

    auto async_result_callback = [provenance](const Aws::S3::Model::PutObjectRequest& request,
                                              const std::shared_ptr<UploadState>& state, int32_t part_number,
                                              const Aws::S3::Model::PutObjectOutcome& outcome) {
      if (!outcome.IsSuccess()) {
        return UploadUsingSingleRequestError(request, outcome, provenance);
      }
      return arrow::Status::OK();
    };

    Aws::S3::Model::PutObjectRequest req{};
    ARROW_RETURN_NOT_OK(SetMetadataInRequest(&req));

    return Upload<Aws::S3::Model::PutObjectRequest, Aws::S3::Model::PutObjectOutcome>(
        std::move(req), std::move(sync_result_callback), std::move(async_result_callback), data, nbytes,
        std::move(owned_buffer));
  }

  arrow::Status UploadPart(const std::shared_ptr<Buffer>& buffer) {
    return UploadPart(buffer->data(), buffer->size(), buffer);
  }

  static arrow::Status UploadPartError(const Aws::S3::Model::UploadPartRequest& request,
                                       const Aws::S3::Model::UploadPartOutcome& outcome,
                                       const std::shared_ptr<S3ClientHolder>& holder) {
    const std::string bucket(FromAwsString(request.GetBucket()));
    return ErrorToStatus(
        fmt::format("When uploading part for key '{}' in bucket '{}': ",
                    BoundedDetail(std::string(FromAwsString(request.GetKey())), 256), BoundedDetail(bucket, 128)),
        "UploadPart", outcome.GetError(), ProvenanceOf(holder, S3ResourceKind::MultipartUpload));
  }

  arrow::Status UploadPart(const void* data, int64_t nbytes, std::shared_ptr<Buffer> owned_buffer = nullptr) {
    if (!IsMultipartCreated()) {
      ARROW_RETURN_NOT_OK(CreateMultipartUpload());
    }

    Aws::S3::Model::UploadPartRequest req{};
    req.SetPartNumber(part_number_);
    req.SetUploadId(multipart_upload_id_);

    auto sync_result_callback = [holder = holder_](const Aws::S3::Model::UploadPartRequest& request,
                                                   const std::shared_ptr<UploadState>& state, int32_t part_number,
                                                   Aws::S3::Model::UploadPartOutcome outcome) {
      if (!outcome.IsSuccess()) {
        return UploadPartError(request, outcome, holder);
      } else {
        AddCompletedPart(state, part_number, outcome.GetResult());
      }

      return arrow::Status::OK();
    };

    auto async_result_callback = [holder = holder_](const Aws::S3::Model::UploadPartRequest& request,
                                                    const std::shared_ptr<UploadState>& state, int32_t part_number,
                                                    const Aws::S3::Model::UploadPartOutcome& outcome) {
      if (!outcome.IsSuccess()) {
        return UploadPartError(request, outcome, holder);
      }
      std::unique_lock<std::mutex> lock(state->mutex);
      AddCompletedPart(state, part_number, outcome.GetResult());
      return arrow::Status::OK();
    };

    FIU_RETURN_ON(FIUKEY_S3FS_PART_UPLOAD_FAIL,
                  arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_S3FS_PART_UPLOAD_FAIL)));

    return Upload<Aws::S3::Model::UploadPartRequest, Aws::S3::Model::UploadPartOutcome>(
        std::move(req), std::move(sync_result_callback), std::move(async_result_callback), data, nbytes,
        std::move(owned_buffer));
  }

  static void HandleUploadOutcome(const std::shared_ptr<UploadState>& state,
                                  const std::shared_ptr<WriterStatus>& writer_status,
                                  const arrow::Status& status) {
    Future<> future;
    bool publish = false;
    std::unique_lock<std::mutex> lock(state->mutex);
    DCHECK_GT(state->uploads_in_progress, 0);
    --state->uploads_in_progress;
    if (!status.ok()) {
      // Counter settlement comes first. Observe while holding the admission
      // lock so a foreground Write cannot register a new part after this
      // failure has won the ordering race.
      writer_status->ObserveFailure(status);
    }
    if (!state->pending_completion_published && (!status.ok() || state->uploads_in_progress == 0)) {
      state->pending_completion_published = true;
      future = state->pending_uploads_completed;
      publish = true;
    }
    const bool owes_abort = state->abort_requested && state->uploads_in_progress == 0;
    lock.unlock();

    // Publishing may allocate. If it fails the exception propagates and the
    // process dies. That is deliberate: the publish slot above has already been
    // consumed, so anything that swallows the failure here leaves
    // pending_uploads_completed unfinished with no later completion able to
    // claim it, and every FlushAsync/CloseAsync waiter blocks forever. A crash
    // is restartable; that hang is not.
    auto completion_status = arrow::io::internal::ObserveBackgroundIOStatus(writer_status.get(), status);
    if (publish) {
      future.MarkFinished(std::move(completion_status));
    }
    if (owes_abort) {
      // Last one out cancels the upload. By now nothing else can add a part, so
      // this attempt cannot be raced by another in-flight one.
      try {
        AbortRecordedUpload(state);
      } catch (...) {
        // Remote cleanup is best effort and must not escape the completion.
      }
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
    if (!result.GetChecksumCRC32C().empty()) {
      part.SetChecksumCRC32C(result.GetChecksumCRC32C());
    }
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
  const bool use_crc32c_checksum_;
  const bool allow_delayed_open_{true};

  int64_t part_upload_size_;

  Aws::String multipart_upload_id_;
  bool closed_ = true;
  int64_t pos_ = 0;
  int32_t part_number_ = 1;
  std::shared_ptr<arrow::ResizableBuffer> current_part_buffer_;
  std::shared_ptr<arrow::io::BufferOutputStream> current_part_;
  int64_t current_part_size_ = 0;

  // This struct is kept alive through background writes to avoid problems
  // in the completion handler.
  struct UploadState {
    std::mutex mutex;
    // Only populated for multi-part uploads.
    Aws::Vector<S3Model::CompletedPart> completed_parts;
    int64_t uploads_in_progress = 0;
    bool pending_completion_published = true;
    arrow::Future<> pending_uploads_completed = arrow::Future<>::MakeFinished(arrow::Status::OK());

    // Everything the LAST completing part needs to cancel the upload, kept here
    // rather than on the stream because this struct outlives it.
    //
    // Abort() cannot finish the job on its own: background part uploads capture
    // their client by value at submit time, so a part already in flight will
    // land successfully after the abort and orphan itself. AWS documents the
    // same thing -- an abort raced by in-flight parts has to be repeated. The
    // repeat happens here, driven by the completion that takes the counter to
    // zero, so no failure path ever waits on a child (R2.6).
    bool abort_requested = false;
    std::shared_ptr<S3ClientHolder> holder;
    std::string bucket;
    std::string key;
    Aws::String multipart_upload_id;
  };
  std::shared_ptr<UploadState> upload_state_;
  std::shared_ptr<WriterStatus> writer_status_;
};

class S3FileSystem::Impl : public std::enable_shared_from_this<S3FileSystem::Impl> {
  public:
  ClientBuilder<S3Client> builder_;
  const arrow::io::IOContext io_context_;
  std::shared_ptr<S3ClientHolder> holder_;
#ifdef WITH_CRT
  bool use_crt_async_reads_ = false;
  ClientBuilder<Aws::S3Crt::S3CrtClient> crt_builder_;
  std::shared_ptr<S3CrtClientHolder> crt_holder_;
#endif
  std::optional<S3Backend> backend_;

  static constexpr int32_t kListObjectsMaxKeys = 1000;
  // At most 1000 keys per multiple-delete request
  static constexpr int32_t kMultipleDeleteMaxKeys = 1000;

  explicit Impl(S3Options options, arrow::io::IOContext io_context)
      : builder_(options),
        io_context_(std::move(io_context))
#ifdef WITH_CRT
        ,
        use_crt_async_reads_(options.use_crt_async_reads),
        crt_builder_(std::move(options))
#endif
  {
  }

  arrow::Status Init() {
    auto result = builder_.BuildClient(io_context_);
    if (!result.ok()) {
      return result.status().WithMessage("Failed to build S3 client: ", result.status().message());
    }
    ARROW_RETURN_NOT_OK(std::move(result).Value(&holder_));
#ifdef WITH_CRT
    if (UseCrtReadPath()) {
      std::shared_ptr<FilesystemMetrics> metrics;
      {
        ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());
        metrics = client_lock.Move()->GetMetrics();
      }
      auto crt_result = crt_builder_.BuildClient(io_context_, std::move(metrics));
      if (!crt_result.ok()) {
        return crt_result.status().WithMessage("Failed to build S3 CRT client: ", crt_result.status().message());
      }
      ARROW_RETURN_NOT_OK(std::move(crt_result).Value(&crt_holder_));
    }
#endif
    return arrow::Status::OK();
  }

  template <typename Error>
  void SaveBackend(const Aws::Client::AWSError<Error>& error) {
    if (!backend_ || *backend_ == S3Backend::Other) {
      backend_ = DetectS3Backend(error);
    }
  }

  const S3Options& options() const { return builder_.options(); }

#ifdef WITH_CRT
  bool UseCrtReadPath() const { return use_crt_async_reads_ && options().cloud_provider != kCloudProviderGCP; }
#endif

  std::string region() const { return std::string(FromAwsString(builder_.config().region)); }

  // Tests to see if a bucket exists
  arrow::Result<bool> BucketExists(const std::string& bucket) {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    S3Model::HeadBucketRequest req;
    req.SetBucket(ToAwsString(bucket));

    auto outcome = client_lock.Move()->HeadBucket(req);
    if (!outcome.IsSuccess()) {
      if (!IsBucketNotFound(outcome.GetError())) {
        return ErrorToStatus(std::forward_as_tuple("When testing for existence of bucket '", bucket, "': "),
                             "HeadBucket", outcome.GetError(), ProvenanceOf(holder_, S3ResourceKind::Bucket));
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
      } else if (!IsBucketNotFound(outcome.GetError())) {
        return ErrorToStatus(std::forward_as_tuple("When creating bucket '", bucket, "': "), "HeadBucket",
                             outcome.GetError(), ProvenanceOf(holder_, S3ResourceKind::Bucket));
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
                           outcome.GetError(), ProvenanceOf(holder_, S3ResourceKind::Bucket));
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
    if (options().use_crc32c_checksum) {
      req.SetChecksumAlgorithm(S3Model::ChecksumAlgorithm::CRC32C);
    }
    return OutcomeToStatus(std::forward_as_tuple("When creating key '", key, "' in bucket '", bucket, "': "),
                           "PutObject", client_lock.Move()->PutObject(req),
                           ProvenanceOf(holder_, S3ResourceKind::Bucket));
  }

  arrow::Status DeleteObject(const std::string& bucket, const std::string& key) {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    S3Model::DeleteObjectRequest req;
    req.SetBucket(ToAwsString(bucket));
    req.SetKey(ToAwsString(key));
    return OutcomeToStatus(std::forward_as_tuple("When delete key '", key, "' in bucket '", bucket, "': "),
                           "DeleteObject", client_lock.Move()->DeleteObject(req),
                           ProvenanceOf(holder_, S3ResourceKind::Bucket));
  }

  arrow::Status CopyObject(const S3Path& src_path, const S3Path& dest_path) {
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());

    S3Model::CopyObjectRequest req;
    req.SetBucket(ToAwsString(dest_path.bucket));
    req.SetKey(ToAwsString(dest_path.key));
    // ARROW-13048: Copy source "Must be URL-encoded" according to AWS SDK docs.
    // However at least in 1.8 and 1.9 the SDK URL-encodes the path for you
    req.SetCopySource(src_path.ToAwsString());
    if (options().use_crc32c_checksum) {
      req.SetChecksumAlgorithm(S3Model::ChecksumAlgorithm::CRC32C);
    }
    auto outcome = client_lock.get()->CopyObject(req);
    if (outcome.IsSuccess()) {
      return arrow::Status::OK();
    }
    return ErrorToStatus(fmt::format("When copying key '{}' in bucket '{}' to key '{}' in bucket '{}': ",
                                     BoundedDetail(src_path.key, 256), BoundedDetail(src_path.bucket, 128),
                                     BoundedDetail(dest_path.key, 256), BoundedDetail(dest_path.bucket, 128)),
                         "CopyObject", outcome.GetError(), ProvenanceOf(holder_, S3ResourceKind::Object));
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
    if (IsExplicitBucketNotFound(outcome.GetError())) {
      return ErrorToStatus(
          std::forward_as_tuple("When reading information for key '", key, "' in bucket '", bucket, "': "),
          "HeadObject", outcome.GetError(), ProvenanceOf(holder_, S3ResourceKind::Bucket));
    }
    if (IsObjectNotFound(outcome.GetError())) {
      return false;
    }
    return ErrorToStatus(
        std::forward_as_tuple("When reading information for key '", key, "' in bucket '", bucket, "': "), "HeadObject",
        outcome.GetError(), ProvenanceOf(holder_));
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
    if (IsBucketNotFound(outcome.GetError())) {
      return ErrorToStatus(
          std::forward_as_tuple("When listing objects under key '", path.key, "' in bucket '", path.bucket, "': "),
          "ListObjectsV2", outcome.GetError(), ProvenanceOf(holder_, S3ResourceKind::Bucket));
    }
    return ErrorToStatus(
        std::forward_as_tuple("When listing objects under key '", path.key, "' in bucket '", path.bucket, "': "),
        "ListObjectsV2", outcome.GetError(), ProvenanceOf(holder_));
  }

  static FileInfo MakeDirectoryInfo(std::string dirname) {
    FileInfo dir;
    dir.set_type(FileType::Directory);
    dir.set_path(std::move(dirname));
    return dir;
  }

  static std::vector<FileInfo> MakeDirectoryInfos(std::vector<std::string> dirnames) {
    std::vector<FileInfo> dir_infos;
    dir_infos.reserve(dirnames.size());
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
                    const FileSelector& select,
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
          new_directories.push_back(parent_dir);
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
        state->files_queue.Push(
            ErrorToStatus(std::forward_as_tuple("When listing objects under key '", state->req.GetPrefix(),
                                                "' in bucket '", state->req.GetBucket(), "': "),
                          "ListObjectsV2", err, ProvenanceOf(state->holder, S3ResourceKind::Bucket)));
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
    [[nodiscard]] std::string_view name() const override { return "S3ListFiles"; }
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
                 const FileInfoSink& sink) {
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

  static arrow::Status DeleteKeyErrorToStatus(const std::string& code, const std::string& message) {
    const auto lower_message = code + ": " + message;
    if (fs::internal::IsAccessDeniedErrorName(code)) {
      return MakeExtendError(ExtendStatusCode::StorageAccessDenied, lower_message);
    }
    if (code == "SlowDown" || code == "RequestLimitExceeded" || code == "Throttling") {
      return MakeExtendError(ExtendStatusCode::StorageTransientThrottling, lower_message);
    }
    if (code == "InternalError" || code == "ServiceUnavailable") {
      return MakeExtendError(ExtendStatusCode::StorageTransientService, lower_message);
    }
    if (code == "RequestTimeout") {
      return MakeExtendError(ExtendStatusCode::StorageTransientTimeout, lower_message);
    }
    if (code == "OperationAborted") {
      return MakeExtendError(ExtendStatusCode::StorageConflict, lower_message);
    }
    return arrow::Status::IOError(lower_message);
  }

  // Delete multiple objects concurrently. Return the first observed
  // lower-layer failure without waiting for sibling requests to finish.
  Future<> DeleteObjectsAsync(const std::string& bucket, const std::vector<std::string>& keys) {
    struct DeleteCallback {
      std::string bucket;
      S3ErrorProvenance provenance;

      arrow::Status operator()(const S3Model::DeleteObjectsOutcome& outcome) const {
        if (!outcome.IsSuccess()) {
          return ErrorToStatus(std::forward_as_tuple("When deleting objects in bucket '", bucket, "': "),
                               "DeleteObjects", outcome.GetError(), provenance);
        }
        // Also need to check per-key errors, even on successful outcome
        // See
        // https://docs.aws.amazon.com/fr_fr/AmazonS3/latest/API/multiobjectdeleteapi.html
        for (const auto& error : outcome.GetResult().GetErrors()) {
          return DeleteKeyErrorToStatus(error.GetCode(), error.GetMessage());
        }
        return arrow::Status::OK();
      }
    };

    const auto chunk_size = static_cast<size_t>(kMultipleDeleteMaxKeys);
    // A request-level failure says nothing about any individual key: S3 did
    // not process the batch. In particular, a generic 404 here can only name
    // the bucket, so do not turn it into object-missing.
    const DeleteCallback delete_cb{bucket, ProvenanceOf(holder_, S3ResourceKind::Bucket)};

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
      if (options().use_crc32c_checksum) {
        req.SetChecksumAlgorithm(S3Model::ChecksumAlgorithm::CRC32C);
      } else if (options().cloud_provider == kCloudProviderGCP) {
        // GCP's S3-compatible API accepts x-amz-checksum-crc32; AWS SDK >=
        // 1.11.x no longer adds one by default, so set it explicitly.
        req.SetChecksumAlgorithm(S3Model::ChecksumAlgorithm::CRC32);
      } else if (options().cloud_provider == kCloudProviderAliyun ||
                 options().cloud_provider == kCloudProviderTencent ||
                 options().cloud_provider == kCloudProviderHuawei) {
        // Aliyun OSS / Tencent COS / Huawei OBS only honor Content-MD5 on
        // DeleteObjects and silently ignore x-amz-checksum-*. AWS SDK >= 1.11.x
        // no longer auto-computes Content-MD5, so compute it from the
        // serialized payload and inject it as a custom header.
        req.SetAdditionalCustomHeaderValue(
            "Content-MD5",
            Aws::Utils::HashingUtils::Base64Encode(Aws::Utils::HashingUtils::CalculateMD5(req.SerializePayload())));
      }
      auto maybe_fut = SubmitIO(io_context_, [holder = holder_, req = std::move(req), delete_cb]() -> arrow::Status {
        ARROW_ASSIGN_OR_RAISE(auto client_lock, holder->Lock());
        return delete_cb(client_lock.Move()->DeleteObjects(req));
      });
      if (!maybe_fut.ok()) {
        return Future<>::MakeFinished(maybe_fut.status());
      }
      futures.push_back(std::move(maybe_fut).ValueOrDie());
    }
    return AllComplete(futures);
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
      if (IsExplicitBucketNotFound(outcome.GetError())) {
        return ErrorToStatus(
            std::forward_as_tuple("When getting information for key '", key, "' in bucket '", bucket, "': "),
            "HeadObject", outcome.GetError(), ProvenanceOf(self->holder_, S3ResourceKind::Bucket));
      }
      if (IsObjectNotFound(outcome.GetError())) {
        // If we can't find the key then it isn't a file. A generic 404 may still
        // be a missing bucket; the ListObjects step that follows this guard is
        // bucket-level and will return 118 rather than swallowing it.
        return true;
      } else {
        return ErrorToStatus(
            std::forward_as_tuple("When getting information for key '", key, "' in bucket '", bucket, "': "),
            "HeadObject", outcome.GetError(), ProvenanceOf(self->holder_));
      }
    }));
  }

  // Some operations require running multiple S3 calls, either in parallel or serially. We
  // need to ensure that the S3 filesystem instance stays valid and that S3 isn't
  // finalized.  We do this by wrapping all the tasks in a scheduler which keeps the
  // resources alive
  Future<> RunInScheduler(std::function<Status(arrow::util::AsyncTaskScheduler*, S3FileSystem::Impl*)> callable) {
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
    return RunInScheduler([bucket, key](arrow::util::AsyncTaskScheduler* scheduler, S3FileSystem::Impl* self) {
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
    Future<> scheduler_fut = RunInScheduler([select, base_path, sink = generator.producer()](
                                                arrow::util::AsyncTaskScheduler* scheduler, S3FileSystem::Impl* self) {
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

  static arrow::Result<std::vector<std::string>> ProcessListBuckets(const S3Model::ListBucketsOutcome& outcome,
                                                                    S3ErrorProvenance provenance) {
    if (!outcome.IsSuccess()) {
      return ErrorToStatus(std::forward_as_tuple("When listing buckets: "), "ListBuckets", outcome.GetError(),
                           provenance);
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
    return ProcessListBuckets(client_lock.Move()->ListBuckets(), ProvenanceOf(holder_));
  }

  Future<std::vector<std::string>> ListBucketsAsync() {
    auto deferred = [self = shared_from_this()]() mutable -> arrow::Result<std::vector<std::string>> {
      ARROW_ASSIGN_OR_RAISE(auto client_lock, self->holder_->Lock());
      return self->ProcessListBuckets(client_lock.Move()->ListBuckets(), ProvenanceOf(self->holder_));
    };
    return DeferNotOk(SubmitIO(io_context_, std::move(deferred)));
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& s, S3FileSystem* fs) {
    ARROW_RETURN_NOT_OK(arrow::fs::internal::AssertNoTrailingSlash(s));
    ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));
    ARROW_RETURN_NOT_OK(ValidateFilePath(path));

    ARROW_RETURN_NOT_OK(CheckS3Initialized());

#ifdef WITH_CRT
    if (UseCrtReadPath()) {
      auto ptr = std::make_shared<ObjectCrtInputFile>(crt_holder_, fs->io_context(), path);
      ARROW_RETURN_NOT_OK(ptr->Init());
      return std::static_pointer_cast<arrow::io::RandomAccessFile>(ptr);
    }
#endif
    auto ptr = std::make_shared<ObjectInputFile>(holder_, fs->io_context(), path);
    ARROW_RETURN_NOT_OK(ptr->Init());
    return std::static_pointer_cast<arrow::io::RandomAccessFile>(ptr);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const FileInfo& info, S3FileSystem* fs) {
    ARROW_RETURN_NOT_OK(arrow::fs::internal::AssertNoTrailingSlash(info.path()));
    if (info.type() == FileType::NotFound) {
      return ::arrow::fs::internal::PathNotFound(info.path());
    }
    if (info.type() != FileType::File && info.type() != FileType::Unknown) {
      return ::arrow::fs::internal::NotAFile(info.path());
    }

    ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(info.path()));
    ARROW_RETURN_NOT_OK(ValidateFilePath(path));

    ARROW_RETURN_NOT_OK(CheckS3Initialized());

#ifdef WITH_CRT
    if (UseCrtReadPath()) {
      auto ptr = std::make_shared<ObjectCrtInputFile>(crt_holder_, fs->io_context(), path, info.size());
      ARROW_RETURN_NOT_OK(ptr->Init());
      return std::static_pointer_cast<arrow::io::RandomAccessFile>(ptr);
    }
#endif
    auto ptr = std::make_shared<ObjectInputFile>(holder_, fs->io_context(), path, info.size());
    ARROW_RETURN_NOT_OK(ptr->Init());
    return std::static_pointer_cast<arrow::io::RandomAccessFile>(ptr);
  }

  arrow::Result<std::shared_ptr<FilesystemMetrics>> GetMetrics() {
#ifdef WITH_CRT
    if (crt_holder_) {
      return {crt_holder_->GetMetrics()};
    }
#endif
    ARROW_ASSIGN_OR_RAISE(auto client_lock, holder_->Lock());
    return {client_lock.Move()->GetMetrics()};
  }
};

S3FileSystem::~S3FileSystem() {}

std::string S3FileSystem::type_name() const { return impl_->options().cloud_provider; }

arrow::Result<std::shared_ptr<S3FileSystem>> S3FileSystem::Make(const S3Options& options,
                                                                const arrow::io::IOContext& io_context) {
  ARROW_RETURN_NOT_OK(CheckS3Initialized());

  std::shared_ptr<S3FileSystem> ptr(new S3FileSystem(options, io_context));
  ARROW_RETURN_NOT_OK(ptr->impl_->Init());
  return ptr;
}

bool S3FileSystem::Equals(const FileSystem& other) const {
  if (this == &other) {
    return true;
  }
  if (other.type_name() != type_name()) {
    return false;
  }
  const auto& s3fs = ::arrow::fs::internal::checked_cast<const S3FileSystem&>(other);
  return options().Equals(s3fs.options());
}

arrow::Result<std::string> S3FileSystem::PathFromUri(const std::string& uri_string) const {
  return arrow::fs::internal::PathFromUriHelper(uri_string, {"multiPartUploadS3"}, /*accept_local_paths=*/false,
                                                arrow::fs::internal::AuthorityHandlingBehavior::kPrepend);
}

arrow::Result<FileInfo> S3FileSystem::GetFileInfo(const std::string& s) {
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
      const auto msg = "When getting information for bucket '" + path.bucket + "': ";
      return ErrorToStatus(msg, "HeadBucket", outcome.GetError(), S3ErrorProvenance{S3ResourceKind::Bucket},
                           impl_->options().region);
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
    if (IsExplicitBucketNotFound(outcome.GetError())) {
      const auto msg = "When getting information for key '" + path.key + "' in bucket '" + path.bucket + "': ";
      return ErrorToStatus(msg, "HeadObject", outcome.GetError(), S3ErrorProvenance{S3ResourceKind::Bucket},
                           impl_->options().region);
    }
    if (!IsObjectNotFound(outcome.GetError())) {
      const auto msg = "When getting information for key '" + path.key + "' in bucket '" + path.bucket + "': ";
      return ErrorToStatus(msg, "HeadObject", outcome.GetError(), S3ErrorProvenance{S3ResourceKind::Object},
                           impl_->options().region);
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

arrow::Result<FileInfoVector> S3FileSystem::GetFileInfo(const FileSelector& select) {
  Future<std::vector<FileInfoVector>> file_infos_fut = CollectAsyncGenerator(GetFileInfoGenerator(select));
  ARROW_ASSIGN_OR_RAISE(std::vector<FileInfoVector> file_infos, file_infos_fut.result());
  FileInfoVector combined_file_infos;
  for (const auto& file_info_vec : file_infos) {
    combined_file_infos.insert(combined_file_infos.end(), file_info_vec.begin(), file_info_vec.end());
  }
  return combined_file_infos;
}

FileInfoGenerator S3FileSystem::GetFileInfoGenerator(const FileSelector& select) {
  return impl_->GetFileInfoGenerator(select);
}

arrow::Status S3FileSystem::CreateDir(const std::string& s, bool recursive) {
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
      ARROW_RETURN_NOT_OK(impl_->CreateBucket(path.bucket));
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
      ARROW_RETURN_NOT_OK(impl_->CreateEmptyDir(path.bucket, parent_key));
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

arrow::Status S3FileSystem::DeleteDir(const std::string& s) {
  ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));
  if (path.empty()) {
    return arrow::Status::NotImplemented("Cannot delete all S3 buckets");
  }
  ARROW_RETURN_NOT_OK(impl_->DeleteDirContentsAsync(path.bucket, path.key).status());
  if (path.key.empty() && options().allow_bucket_deletion) {
    // Delete bucket
    ARROW_ASSIGN_OR_RAISE(auto client_lock, impl_->holder_->Lock());
    S3Model::DeleteBucketRequest req;
    req.SetBucket(ToAwsString(path.bucket));
    return OutcomeToStatus(std::forward_as_tuple("When deleting bucket '", path.bucket, "': "), "DeleteBucket",
                           client_lock.Move()->DeleteBucket(req), S3ErrorProvenance{S3ResourceKind::Bucket});
  } else if (path.key.empty()) {
    return arrow::Status::IOError("Would delete bucket '", path.bucket, "'. ",
                                  "To delete buckets, enable the allow_bucket_deletion option.");
  } else {
    // Delete "directory"
    ARROW_RETURN_NOT_OK(impl_->DeleteObject(path.bucket, path.key + kSep));
    // Parent may be implicitly deleted if it became empty, recreate it
    return impl_->EnsureParentExists(path);
  }
}

arrow::Status S3FileSystem::DeleteDirContents(const std::string& s, bool missing_dir_ok) {
  return DeleteDirContentsAsync(s, missing_dir_ok).status();
}

arrow::Future<> S3FileSystem::DeleteDirContentsAsync(const std::string& s, bool missing_dir_ok) {
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

arrow::Status S3FileSystem::DeleteRootDirContents() {
  return arrow::Status::NotImplemented("Cannot delete all S3 buckets");
}

arrow::Status S3FileSystem::DeleteFile(const std::string& s) {
  ARROW_ASSIGN_OR_RAISE(auto client_lock, impl_->holder_->Lock());

  ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));
  ARROW_RETURN_NOT_OK(ValidateFilePath(path));

  // Check the object exists
  S3Model::HeadObjectRequest req;
  req.SetBucket(ToAwsString(path.bucket));
  req.SetKey(ToAwsString(path.key));

  auto outcome = client_lock.Move()->HeadObject(req);
  if (!outcome.IsSuccess()) {
    if (IsExplicitBucketNotFound(outcome.GetError())) {
      return ErrorToStatus(
          std::forward_as_tuple("When getting information for key '", path.key, "' in bucket '", path.bucket, "': "),
          "HeadObject", outcome.GetError(), S3ErrorProvenance{S3ResourceKind::Bucket});
    }
    if (IsObjectNotFound(outcome.GetError())) {
      // A bodyless object HEAD cannot distinguish a missing key from a missing
      // bucket. Keep the operation-level missing-object result; diagnosing it
      // with HeadBucket would add I/O only after the original operation failed.
      return PathNotFound(path);
    } else {
      return ErrorToStatus(
          std::forward_as_tuple("When getting information for key '", path.key, "' in bucket '", path.bucket, "': "),
          "HeadObject", outcome.GetError(), S3ErrorProvenance{});
    }
  }
  // Object found, delete it
  ARROW_RETURN_NOT_OK(impl_->DeleteObject(path.bucket, path.key));
  // Parent may be implicitly deleted if it became empty, recreate it
  return impl_->EnsureParentExists(path);
}

arrow::Status S3FileSystem::Move(const std::string& src, const std::string& dest) {
  // XXX We don't implement moving directories as it would be too expensive:
  // one must copy all directory contents one by one (including object data),
  // then delete the original contents.

  ARROW_ASSIGN_OR_RAISE(auto src_path, S3Path::FromString(src));
  ARROW_RETURN_NOT_OK(ValidateFilePath(src_path));
  ARROW_ASSIGN_OR_RAISE(auto dest_path, S3Path::FromString(dest));
  ARROW_RETURN_NOT_OK(ValidateFilePath(dest_path));

  if (src_path == dest_path) {
    return arrow::Status::OK();
  }
  ARROW_RETURN_NOT_OK(impl_->CopyObject(src_path, dest_path));
  ARROW_RETURN_NOT_OK(impl_->DeleteObject(src_path.bucket, src_path.key));
  // Source parent may be implicitly deleted if it became empty, recreate it
  return impl_->EnsureParentExists(src_path);
}

arrow::Status S3FileSystem::CopyFile(const std::string& src, const std::string& dest) {
  ARROW_ASSIGN_OR_RAISE(auto src_path, S3Path::FromString(src));
  ARROW_RETURN_NOT_OK(ValidateFilePath(src_path));
  ARROW_ASSIGN_OR_RAISE(auto dest_path, S3Path::FromString(dest));
  ARROW_RETURN_NOT_OK(ValidateFilePath(dest_path));

  if (src_path == dest_path) {
    return arrow::Status::OK();
  }
  return impl_->CopyObject(src_path, dest_path);
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> S3FileSystem::OpenOutputStreamWithUploadSize(
    const std::string& s, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata, int64_t upload_size) {
  ARROW_RETURN_NOT_OK(arrow::fs::internal::AssertNoTrailingSlash(s));
  ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));
  ARROW_RETURN_NOT_OK(ValidateFilePath(path));

  ARROW_RETURN_NOT_OK(CheckS3Initialized());

  auto ptr =
      std::make_shared<CustomOutputStream>(impl_->holder_, io_context(), path, impl_->options(), metadata, upload_size);
  ARROW_RETURN_NOT_OK(ptr->Init());
  return ptr;
};

S3FileSystem::S3FileSystem(const S3Options& options, const arrow::io::IOContext& io_context)
    : FileSystem(io_context), impl_(std::make_shared<Impl>(options, io_context)) {
  default_async_is_sync_ = false;
}

const S3Options& S3FileSystem::options() const { return impl_->options(); }

std::string S3FileSystem::region() const { return impl_->region(); }

arrow::Result<std::shared_ptr<arrow::io::InputStream>> S3FileSystem::OpenInputStream(const std::string& s) {
  return impl_->OpenInputFile(s, this);
}

arrow::Result<std::shared_ptr<arrow::io::InputStream>> S3FileSystem::OpenInputStream(const FileInfo& info) {
  return impl_->OpenInputFile(info, this);
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> S3FileSystem::OpenInputFile(const std::string& s) {
  return impl_->OpenInputFile(s, this);
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> S3FileSystem::OpenInputFile(const FileInfo& info) {
  return impl_->OpenInputFile(info, this);
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> S3FileSystem::OpenOutputStream(
    const std::string& s, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) {
  // safe to cast multi_part_upload_size to int64_t, the range is 5MB to 5GB
  return OpenOutputStreamWithUploadSize(s, metadata, impl_->options().multi_part_upload_size);
};

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> S3FileSystem::OpenAppendStream(
    const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) {
  return arrow::Status::NotImplemented("It is not possible to append efficiently to S3 objects");
}

std::shared_ptr<FilesystemMetrics> S3FileSystem::GetMetrics() const {
  auto result = const_cast<S3FileSystem*>(this)->impl_->GetMetrics();
  if (result.ok()) {
    return result.ValueOrDie();
  }
  return nullptr;
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> S3FileSystem::OpenConditionalOutputStream(
    const std::string& path, std::shared_ptr<arrow::KeyValueMetadata> metadata) {
  if (!metadata) {
    metadata = std::make_shared<arrow::KeyValueMetadata>();
  }

  // Get the type name from this filesystem
  std::string type_name = this->type_name();

  if (type_name == kCloudProviderAWS) {
    metadata->Append("If-None-Match", "*");
  } else if (type_name == kCloudProviderGCP) {
    metadata->Append("x-goog-if-generation-match", "0");
  } else if (type_name == kCloudProviderTencent) {
    metadata->Append("x-cos-forbid-overwrite", "true");
  } else if (type_name == kCloudProviderAliyun) {
    metadata->Append("x-oss-forbid-overwrite", "true");
  } else if (type_name == kAzureFileSystemName) {
    metadata->Append("If-None-Match", "*");
  } else {  // Unsupported fs type
    return arrow::Status::NotImplemented("Conditional uploads are not supported for current fs type: ", type_name);
  }

  return OpenOutputStream(path, metadata);
}

}  // namespace milvus_storage
