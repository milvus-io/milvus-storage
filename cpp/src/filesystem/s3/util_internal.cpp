
#include "milvus-storage/filesystem/s3/util_internal.h"

#include <algorithm>
#include <cerrno>

#include <arrow/buffer.h>
#include <arrow/filesystem/path_util.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/util/io_util.h>
#include <arrow/util/string.h>

namespace arrow {

using internal::StatusDetailFromErrno;
using util::Uri;

namespace fs {
namespace internal {

TimePoint CurrentTimePoint() {
  auto now = std::chrono::system_clock::now();
  return TimePoint(std::chrono::duration_cast<TimePoint::duration>(now.time_since_epoch()));
}

arrow::Status CopyStream(const std::shared_ptr<io::InputStream>& src,
                         const std::shared_ptr<io::OutputStream>& dest,
                         int64_t chunk_size,
                         const io::IOContext& io_context) {
  ARROW_ASSIGN_OR_RAISE(auto chunk, AllocateBuffer(chunk_size, io_context.pool()));

  while (true) {
    ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, src->Read(chunk_size, chunk->mutable_data()));
    if (bytes_read == 0) {
      // EOF
      break;
    }
    RETURN_NOT_OK(dest->Write(chunk->data(), bytes_read));
  }

  return arrow::Status::OK();
}

arrow::Status PathNotFound(std::string_view path) {
  return arrow::Status::IOError("Path does not exist '", path, "'")
      .WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
}

arrow::Status IsADir(std::string_view path) {
  return arrow::Status::IOError("Is a directory: '", path, "'").WithDetail(StatusDetailFromErrno(EISDIR));
}

arrow::Status NotADir(std::string_view path) {
  return arrow::Status::IOError("Not a directory: '", path, "'").WithDetail(StatusDetailFromErrno(ENOTDIR));
}

arrow::Status NotEmpty(std::string_view path) {
  return arrow::Status::IOError("Directory not empty: '", path, "'").WithDetail(StatusDetailFromErrno(ENOTEMPTY));
}

arrow::Status NotAFile(std::string_view path) { return arrow::Status::IOError("Not a regular file: '", path, "'"); }

arrow::Status InvalidDeleteDirContents(std::string_view path) {
  return arrow::Status::Invalid("DeleteDirContents called on invalid path '", path, "'. ",
                                "If you wish to delete the root directory's contents, call DeleteRootDirContents.");
}

arrow::Result<Uri> ParseFileSystemUri(const std::string& uri_string) {
  Uri uri;
  auto status = uri.Parse(uri_string);
  if (!status.ok()) {
#ifdef _WIN32
    // Could be a "file:..." URI with backslashes instead of regular slashes.
    RETURN_NOT_OK(uri.Parse(ToSlashes(uri_string)));
    if (uri.scheme() != "file") {
      return arrow::status;
    }
#else
    return status;
#endif
  }
  return uri;
}

#ifdef _WIN32
static bool IsDriveLetter(char c) {
  // Can't use locale-dependent functions from the C/C++ stdlib
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
#endif

bool DetectAbsolutePath(const std::string& s) {
  // Is it a /-prefixed local path?
  if (s.length() >= 1 && s[0] == '/') {
    return true;
  }
#ifdef _WIN32
  // Is it a \-prefixed local path?
  if (s.length() >= 1 && s[0] == '\\') {
    return true;
  }
  // Does it start with a drive letter in addition to being /- or \-prefixed,
  // e.g. "C:\..."?
  if (s.length() >= 3 && s[1] == ':' && (s[2] == '/' || s[2] == '\\') && IsDriveLetter(s[0])) {
    return true;
  }
#endif
  return false;
}

arrow::Result<std::string> PathFromUriHelper(const std::string& uri_string,
                                             std::vector<std::string> supported_schemes,
                                             bool accept_local_paths,
                                             AuthorityHandlingBehavior authority_handling) {
  if (internal::DetectAbsolutePath(uri_string)) {
    if (accept_local_paths) {
      // Normalize the path and remove any trailing slash
      return std::string(internal::RemoveTrailingSlash(ToSlashes(uri_string), /*preserve_root=*/true));
    }
    return arrow::Status::Invalid(
        "The filesystem is not capable of loading local paths.  Expected a URI but "
        "received ",
        uri_string);
  }
  Uri uri;
  ARROW_RETURN_NOT_OK(uri.Parse(uri_string));
  const auto scheme = uri.scheme();
  if (std::find(supported_schemes.begin(), supported_schemes.end(), scheme) == supported_schemes.end()) {
    std::string expected_schemes = ::arrow::internal::JoinStrings(supported_schemes, ", ");
    return arrow::Status::Invalid("The filesystem expected a URI with one of the schemes (", expected_schemes,
                                  ") but received ", uri_string);
  }
  std::string host = uri.host();
  std::string path = uri.path();
  if (host.empty()) {
    // Just a path, may be absolute or relative, only allow relative paths if local
    if (path[0] == '/') {
      return std::string(internal::RemoveTrailingSlash(path));
    }
    if (accept_local_paths) {
      return std::string(internal::RemoveTrailingSlash(path));
    }
    return arrow::Status::Invalid("The filesystem does not support relative paths.  Received ", uri_string);
  }
  if (authority_handling == AuthorityHandlingBehavior::kDisallow) {
    return arrow::Status::Invalid(
        "The filesystem does not support the authority (host) component of a URI.  "
        "Received ",
        uri_string);
  }
  if (path[0] != '/') {
    // This should not be possible
    return arrow::Status::Invalid(
        "The provided URI has a host component but a relative path which is not "
        "supported. "
        "Received ",
        uri_string);
  }
  switch (authority_handling) {
    case AuthorityHandlingBehavior::kPrepend:
      return std::string(internal::RemoveTrailingSlash(host + path));
    case AuthorityHandlingBehavior::kWindows:
      return std::string(internal::RemoveTrailingSlash("//" + host + path));
    case AuthorityHandlingBehavior::kIgnore:
      return std::string(internal::RemoveTrailingSlash(path, /*preserve_root=*/true));
    default:
      return arrow::Status::Invalid("Unrecognized authority_handling value");
  }
}

arrow::Result<FileInfoVector> GlobFiles(const std::shared_ptr<FileSystem>& filesystem, const std::string& glob) {
  // TODO: ARROW-17640
  // The candidate entries at the current depth level.
  // We start with the filesystem root.
  FileInfoVector results{FileInfo("", FileType::Directory)};
  // The exact tail that will later require matching with candidate entries
  std::string current_tail;
  auto is_leading_slash = HasLeadingSlash(glob);
  auto split_glob = SplitAbstractPath(glob, '/');

  // Process one depth level at once, from root to leaf
  for (const auto& glob_component : split_glob) {
    if (glob_component.find_first_of("*?") == std::string::npos) {
      // If there are no wildcards at the current level, just append
      // the exact glob path component.
      current_tail = ConcatAbstractPath(current_tail, glob_component);
      continue;
    } else {
      FileInfoVector children;
      for (const auto& res : results) {
        if (res.type() != FileType::Directory) {
          continue;
        }
        FileSelector selector;
        selector.base_dir = current_tail.empty() ? res.path() : ConcatAbstractPath(res.path(), current_tail);
        if (is_leading_slash) {
          selector.base_dir = EnsureLeadingSlash(selector.base_dir);
        }
        ARROW_ASSIGN_OR_RAISE(auto entries, filesystem->GetFileInfo(selector));
        Globber globber(ConcatAbstractPath(selector.base_dir, glob_component));
        for (auto&& entry : entries) {
          if (globber.Matches(entry.path())) {
            children.push_back(std::move(entry));
          }
        }
      }
      results = std::move(children);
      current_tail.clear();
    }
  }

  if (!current_tail.empty()) {
    std::vector<std::string> paths;
    paths.reserve(results.size());
    for (const auto& file : results) {
      paths.push_back(ConcatAbstractPath(file.path(), current_tail));
    }
    ARROW_ASSIGN_OR_RAISE(results, filesystem->GetFileInfo(paths));
  }

  std::vector<FileInfo> out;
  for (auto&& file : results) {
    if (file.type() != FileType::NotFound) {
      out.push_back(std::move(file));
    }
  }

  return out;
}

FileSystemGlobalOptions global_options;

}  // namespace internal
}  // namespace fs
}  // namespace arrow
