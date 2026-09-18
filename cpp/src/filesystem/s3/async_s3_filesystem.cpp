// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#include "filesystem/s3/async_s3_filesystem.h"
#ifdef WITH_CRT
#include <charconv>
#include <cctype>
#include <limits>
#include <set>
#include <arrow/filesystem/path_util.h>
#include <aws/core/utils/xml/XmlSerializer.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/HeadObjectResult.h>
#include "milvus-storage/filesystem/s3/s3_internal.h"
#include "milvus-storage/filesystem/util_internal.h"
#include <aws/s3/model/ListBucketsRequest.h>
#include <aws/s3/model/ListBucketsResult.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/ListObjectsV2Result.h>

namespace milvus_storage {
namespace {
using arrow::Future;
using arrow::Result;
using arrow::Status;
using arrow::fs::FileInfo;
using arrow::fs::FileInfoVector;
using arrow::fs::FileSelector;
using arrow::fs::FileType;
using Aws::Http::HttpMethod;
namespace S3 = Aws::S3::Model;

struct Path {
  std::string bucket;
  std::string key;
  static Result<Path> Parse(const std::string& path) {
    if (arrow::fs::internal::IsLikelyUri(path) || (!path.empty() && path.front() == '/')) {
      return Status::Invalid("Expected bucket/key path: ", path);
    }
    if (path.find('\0') != std::string::npos)
      return Status::Invalid("NUL in S3 path");
    auto clean = std::string(arrow::fs::internal::RemoveTrailingSlash(path));
    for (const auto& component : arrow::fs::internal::SplitAbstractPath(clean)) {
      if (component == "." || component == "..")
        return Status::Invalid("Dot components in S3 path");
    }
    ARROW_RETURN_NOT_OK(arrow::fs::internal::ValidateAbstractPath(clean));
    auto slash = clean.find('/');
    return Path{clean.substr(0, slash), slash == std::string::npos ? "" : clean.substr(slash + 1)};
  }
};
template <class T>
Future<T> Failed(Status status) {
  return Future<T>::MakeFinished(std::move(status));
}

template <class T>
Result<T> XmlResult(const NativeS3Response& response) {
  ARROW_RETURN_NOT_OK(response.ToStatus());
  auto xml = Aws::Utils::Xml::XmlDocument::CreateFromXmlString(response.body.c_str());
  if (!xml.WasParseSuccessful() || xml.GetRootElement().IsNull())
    return Status::IOError("Invalid S3 XML response");
  Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument> result(std::move(xml), response.headers);
  return T(result);
}
Result<int64_t> ContentLength(const NativeS3Response& response) {
  const auto it = response.headers.find("content-length");
  if (it == response.headers.end())
    return Status::IOError("S3 HEAD has no Content-Length");
  int64_t size = -1;
  const auto& value = it->second;
  auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), size);
  if (error != std::errc() || end != value.data() + value.size() || size < 0) {
    return Status::IOError("Invalid S3 Content-Length");
  }
  return size;
}
FileInfo Info(const std::string& path, FileType type) {
  FileInfo info(path, type);
  return info;
}

Result<Path> ObjectPath(const std::string& path) {
  ARROW_RETURN_NOT_OK(arrow::fs::internal::AssertNoTrailingSlash(path));
  ARROW_ASSIGN_OR_RAISE(auto parsed, Path::Parse(path));
  if (parsed.key.empty())
    return Status::Invalid("Expected S3 object path");
  return parsed;
}
class AsyncS3FileSystem final : public NativeS3Operations, public std::enable_shared_from_this<AsyncS3FileSystem> {
  public:
  AsyncS3FileSystem(std::shared_ptr<NativeS3Transport> transport, S3Options options, arrow::io::IOContext io)
      : transport_(std::move(transport)), options_(std::move(options)), io_(std::move(io)) {}

  Future<NativeS3ObjectMetadata> ReadMetadataAsync(const std::string& path,
                                                   const arrow::io::IOContext& io_context) override {
    ARROW_RETURN_NOT_OK(arrow::fs::internal::AssertNoTrailingSlash(path));
    ARROW_ASSIGN_OR_RAISE(auto parsed, Path::Parse(path));
    if (parsed.key.empty())
      return Status::Invalid("Expected S3 object path");
    S3::HeadObjectRequest request;
    request.SetBucket(parsed.bucket.c_str());
    request.SetKey(parsed.key.c_str());
    return transport_->Send(request, parsed.key, HttpMethod::HTTP_HEAD, "", io_context)
        .Then([path](const NativeS3Response& response) -> Result<NativeS3ObjectMetadata> {
          if (response.HasHttpStatus(404))
            return arrow::fs::internal::PathNotFound(path);
          auto status = response.ToStatus();
          if (!status.ok())
            return status.WithMessage("HeadObject for '", path, "': ", status.message());
          if (response.headers.find("content-length") == response.headers.end())
            return Status::IOError("HEAD has no Content-Length");
          Aws::Utils::Xml::XmlDocument xml;
          Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument> result(std::move(xml), response.headers);
          S3::HeadObjectResult head(result);
          if (head.GetContentLength() < 0)
            return Status::IOError("Invalid HEAD Content-Length");
          return NativeS3ObjectMetadata{head.GetContentLength(), fs::internal::GetObjectMetadata(head)};
        });
  }

  Future<NativeS3Response> Head(const Path& path, bool marker = false) {
    if (path.key.empty()) {
      S3::HeadBucketRequest request;
      request.SetBucket(path.bucket.c_str());
      return transport_->Send(request, "", HttpMethod::HTTP_HEAD, "", io_);
    }
    S3::HeadObjectRequest request;
    request.SetBucket(path.bucket.c_str());
    auto key = path.key + (marker ? "/" : "");
    request.SetKey(key.c_str());
    return transport_->Send(request, key, HttpMethod::HTTP_HEAD, "", io_);
  }
  Future<NativeS3Response> List(const Path& path, const std::string& token, bool recursive, int max_keys = 1000) {
    S3::ListObjectsV2Request request;
    request.SetBucket(path.bucket.c_str());
    request.SetPrefix((path.key.empty() ? "" : path.key + "/").c_str());
    if (!recursive)
      request.SetDelimiter("/");
    request.SetMaxKeys(max_keys);
    if (!token.empty())
      request.SetContinuationToken(token.c_str());
    return transport_->Send(request, "", HttpMethod::HTTP_GET, "?list-type=2", io_);
  }
  Future<FileInfo> GetFileInfoAsync(const std::string& path) override {
    auto parsed = Path::Parse(path);
    if (!parsed.ok())
      return Failed<FileInfo>(parsed.status());
    if (parsed->bucket.empty())
      return Future<FileInfo>::MakeFinished(Info(path, FileType::Directory));
    auto p = *parsed;
    return Head(p).Then([self = shared_from_this(), p, path](const NativeS3Response& response) -> Future<FileInfo> {
      if (!response.HasHttpStatus(404)) {
        auto status = response.ToStatus();
        if (!status.ok())
          return Failed<FileInfo>(status);
        auto info = Info(path, p.key.empty() ? FileType::Directory : FileType::File);
        if (!p.key.empty()) {
          auto size = ContentLength(response);
          if (!size.ok())
            return Failed<FileInfo>(size.status());
          info.set_size(*size);
          const auto modified = response.headers.find("last-modified");
          if (modified != response.headers.end()) {
            Aws::Utils::DateTime time(modified->second, Aws::Utils::DateFormat::RFC822);
            if (time.WasParseSuccessful())
              info.set_mtime(time.UnderlyingTimestamp());
          }
          auto type = response.headers.find("content-type");
          if (*size == 0 && type != response.headers.end() && type->second == "application/x-directory") {
            info.set_type(FileType::Directory);
          }
        }
        return Future<FileInfo>::MakeFinished(std::move(info));
      }
      if (p.key.empty())
        return Future<FileInfo>::MakeFinished(Info(path, FileType::NotFound));
      // A directory marker is an object whose key ends in '/'. Prefix listing
      // also discovers implicit directories and avoids an extra marker HEAD.
      return self->List(p, "", false, 1).Then([path](const NativeS3Response& listed) -> Result<FileInfo> {
        if (listed.HasHttpStatus(404))
          return Info(path, FileType::NotFound);
        ARROW_ASSIGN_OR_RAISE(auto result, XmlResult<S3::ListObjectsV2Result>(listed));
        return Info(path, result.GetContents().empty() && result.GetCommonPrefixes().empty() ? FileType::NotFound
                                                                                             : FileType::Directory);
      });
    });
  }

  arrow::fs::FileInfoGenerator GetFileInfoGenerator(const FileSelector& selector) override {
    struct Listing {
      std::shared_ptr<AsyncS3FileSystem> fs;
      FileSelector selector;
      Path path;
      std::string token;
      std::string previous_key;
      bool done = false;
    };
    auto parsed = Path::Parse(selector.base_dir);
    if (!parsed.ok())
      return [status = parsed.status()] { return Failed<FileInfoVector>(status); };
    auto state = std::make_shared<Listing>(Listing{shared_from_this(), selector, *parsed});
    if (state->path.bucket.empty()) {
      // Root listing enumerates buckets. Recursive root traversal is composed
      // lazily from each bucket's generator, without collecting object listings.
      struct Root {
        std::shared_ptr<Listing> state;
        FileInfoVector buckets;
        size_t next = 0;
        bool loaded = false;
        std::string token;
        arrow::fs::FileInfoGenerator child;
      };
      auto root = std::make_shared<Root>();
      root->state = state;
      auto next = std::make_shared<std::function<Future<FileInfoVector>()>>();
      // Weak recursion prevents a generator/function ownership cycle.
      std::weak_ptr<std::function<Future<FileInfoVector>()>> weak = next;
      *next = [root, weak]() -> Future<FileInfoVector> {
        if (!root->loaded || (!root->child && root->next == root->buckets.size() && !root->token.empty())) {
          root->loaded = true;
          S3::ListBucketsRequest request;
          request.SetMaxBuckets(1000);
          if (!root->token.empty())
            request.SetContinuationToken(root->token.c_str());
          return root->state->fs->transport_->Send(request, "", HttpMethod::HTTP_GET, "", root->state->fs->io_)
              .Then([root](const NativeS3Response& response) -> Result<FileInfoVector> {
                ARROW_ASSIGN_OR_RAISE(auto result, XmlResult<S3::ListBucketsResult>(response));
                const std::string token = result.GetContinuationToken().c_str();
                if (!token.empty() && token == root->token)
                  return Status::IOError("S3 ListBuckets repeated continuation token");
                root->token = token;
                root->buckets.clear();
                root->next = 0;
                for (const auto& bucket : result.GetBuckets())
                  root->buckets.push_back(Info(bucket.GetName().c_str(), FileType::Directory));
                if (!root->state->selector.recursive)
                  root->next = root->buckets.size();
                return root->buckets;
              })
              .Then([root, weak](FileInfoVector page) -> Future<FileInfoVector> {
                if (page.empty() && !root->token.empty()) {
                  if (auto next = weak.lock())
                    return (*next)();
                  return Failed<FileInfoVector>(Status::Cancelled("S3 listing abandoned"));
                }
                return Future<FileInfoVector>::MakeFinished(std::move(page));
              });
        }
        if (!root->state->selector.recursive)
          return Future<FileInfoVector>::MakeFinished(FileInfoVector{});
        if (!root->child) {
          if (root->next == root->buckets.size())
            return Future<FileInfoVector>::MakeFinished(FileInfoVector{});
          auto child_selector = root->state->selector;
          child_selector.base_dir = root->buckets[root->next++].path();
          if (child_selector.max_recursion <= 0)
            return Future<FileInfoVector>::MakeFinished(FileInfoVector{});
          --child_selector.max_recursion;
          root->child = root->state->fs->GetFileInfoGenerator(child_selector);
        }
        return root->child().Then([root, weak](const FileInfoVector& page) -> Future<FileInfoVector> {
          if (!page.empty())
            return Future<FileInfoVector>::MakeFinished(page);
          root->child = {};
          if (auto next = weak.lock())
            return (*next)();
          return Failed<FileInfoVector>(Status::Cancelled("S3 listing abandoned"));
        });
      };
      return [next] { return (*next)(); };
    }
    auto next = std::make_shared<std::function<Future<FileInfoVector>()>>();
    std::weak_ptr<std::function<Future<FileInfoVector>()>> weak = next;
    *next = [state, weak]() -> Future<FileInfoVector> {
      if (state->done)
        return Future<FileInfoVector>::MakeFinished(FileInfoVector{});
      return state->fs->List(state->path, state->token, state->selector.recursive)
          .Then([state, weak](const NativeS3Response& response) -> Future<FileInfoVector> {
            if (response.HasHttpStatus(404) && state->selector.allow_not_found) {
              state->done = true;
              return Future<FileInfoVector>::MakeFinished(FileInfoVector{});
            }
            auto decoded = XmlResult<S3::ListObjectsV2Result>(response);
            if (!decoded.ok()) {
              state->done = true;
              return Failed<FileInfoVector>(decoded.status());
            }
            const auto& result = *decoded;
            const std::string token = result.GetNextContinuationToken().c_str();
            if (result.GetIsTruncated() && (token.empty() || token == state->token)) {
              state->done = true;
              return Failed<FileInfoVector>(Status::IOError("S3 LIST returned a missing/repeated continuation token"));
            }
            state->token = token;
            state->done = !result.GetIsTruncated();
            if (state->done && result.GetContents().empty() && result.GetCommonPrefixes().empty() &&
                state->previous_key.empty() && !state->path.key.empty() && !state->selector.allow_not_found) {
              return state->fs->GetFileInfoAsync(state->selector.base_dir)
                  .Then([](const FileInfo& info) -> Result<FileInfoVector> {
                    if (info.type() != FileType::Directory)
                      return Status::IOError("S3 listing base is not a directory: ", info.path());
                    return FileInfoVector{};
                  });
            }
            FileInfoVector page;
            const auto prefix = state->path.key.empty() ? "" : state->path.key + "/";
            for (const auto& object : result.GetContents()) {
              const std::string key = object.GetKey().c_str();
              if (key.compare(0, prefix.size(), prefix) != 0)
                return Failed<FileInfoVector>(Status::IOError("S3 LIST escaped prefix"));
              if (key == prefix)
                continue;
              const auto relative = key.substr(prefix.size());
              const auto depth = std::count(relative.begin(), relative.end(), '/');
              if (state->selector.recursive) {
                size_t slash = key.find('/', prefix.size());
                int level = 0;
                while (slash != std::string::npos && level++ <= state->selector.max_recursion) {
                  const auto dir = key.substr(0, slash + 1);
                  if (state->previous_key.compare(0, dir.size(), dir) != 0) {
                    page.push_back(Info(state->path.bucket + "/" + key.substr(0, slash), FileType::Directory));
                  }
                  slash = key.find('/', slash + 1);
                }
              }
              if (key.back() != '/' && depth <= state->selector.max_recursion) {
                auto info = Info(state->path.bucket + "/" + key, FileType::File);
                info.set_size(object.GetSize());
                info.set_mtime(object.GetLastModified().UnderlyingTimestamp());
                page.push_back(std::move(info));
              }
              state->previous_key = key;
            }
            for (const auto& common : result.GetCommonPrefixes()) {
              auto key = std::string(common.GetPrefix().c_str());
              if (key.compare(0, prefix.size(), prefix) != 0 || key.size() <= prefix.size() || key.back() != '/')
                return Failed<FileInfoVector>(Status::IOError("S3 common prefix escaped listing"));
              if (!key.empty() && key.back() == '/')
                key.pop_back();
              page.push_back(Info(state->path.bucket + "/" + key, FileType::Directory));
            }
            if (page.empty() && !state->done) {
              if (auto next = weak.lock())
                return (*next)();
              return Failed<FileInfoVector>(Status::Cancelled("S3 listing abandoned"));
            }
            return Future<FileInfoVector>::MakeFinished(std::move(page));
          });
    };
    return [next] { return (*next)(); };
  }

  Future<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStreamAsync(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    auto parsed = ObjectPath(path);
    if (!parsed.ok())
      return Failed<std::shared_ptr<arrow::io::OutputStream>>(parsed.status());
    return Future<std::shared_ptr<arrow::io::OutputStream>>::MakeFinished(
        OpenNativeS3OutputStream(options_, transport_, io_, path, metadata));
  }

  private:
  std::shared_ptr<NativeS3Transport> transport_;
  S3Options options_;
  arrow::io::IOContext io_;
};
}  // namespace

Result<std::shared_ptr<NativeS3Operations>> MakeNativeS3Operations(const S3Options& options,
                                                                   const arrow::io::IOContext& io_context,
                                                                   std::shared_ptr<NativeS3Transport> transport) {
  if (!io_context.executor())
    return Status::Invalid("Native S3 requires a caller executor");
  if (!transport)
    return Status::Invalid("Native S3 requires the filesystem transport");
  return std::make_shared<AsyncS3FileSystem>(std::move(transport), options, io_context);
}
}  // namespace milvus_storage
#endif
