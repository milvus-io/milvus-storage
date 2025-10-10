
#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include "arrow/filesystem/filesystem.h"
#include "arrow/io/interfaces.h"
#include "arrow/status.h"
#include "arrow/util/uri.h"
#include "arrow/util/visibility.h"

namespace arrow {
using util::Uri;
namespace fs {
namespace internal {

template <typename OutputType, typename InputType>
inline OutputType checked_cast(InputType&& value) {
  static_assert(
      std::is_class<typename std::remove_pointer<typename std::remove_reference<InputType>::type>::type>::value,
      "checked_cast input type must be a class");
  static_assert(
      std::is_class<typename std::remove_pointer<typename std::remove_reference<OutputType>::type>::type>::value,
      "checked_cast output type must be a class");
#ifdef NDEBUG
  return static_cast<OutputType>(value);
#else
  return dynamic_cast<OutputType>(value);
#endif
}

ARROW_EXPORT
TimePoint CurrentTimePoint();

ARROW_EXPORT
arrow::Status CopyStream(const std::shared_ptr<io::InputStream>& src,
                         const std::shared_ptr<io::OutputStream>& dest,
                         int64_t chunk_size,
                         const io::IOContext& io_context);

ARROW_EXPORT
arrow::Status PathNotFound(std::string_view path);

ARROW_EXPORT
arrow::Status IsADir(std::string_view path);

ARROW_EXPORT
arrow::Status NotADir(std::string_view path);

ARROW_EXPORT
arrow::Status NotEmpty(std::string_view path);

ARROW_EXPORT
arrow::Status NotAFile(std::string_view path);

ARROW_EXPORT
arrow::Status InvalidDeleteDirContents(std::string_view path);

/// \brief Parse the string as a URI
/// \param uri_string the string to parse
///
/// This is the same as Uri::Parse except it tolerates Windows
/// file URIs that contain backslash instead of /
arrow::Result<Uri> ParseFileSystemUri(const std::string& uri_string);

/// \brief check if the string is a local absolute path
ARROW_EXPORT
bool DetectAbsolutePath(const std::string& s);

/// \brief describes how to handle the authority (host) component of the URI
enum class AuthorityHandlingBehavior {
  // Return an invalid status if the authority is non-empty
  kDisallow = 0,
  // Prepend the authority to the path (e.g. authority/some/path)
  kPrepend = 1,
  // Convert to a Windows style network path (e.g. //authority/some/path)
  kWindows = 2,
  // Ignore the authority and just use the path
  kIgnore = 3
};

/// \brief check to see if uri_string matches one of the supported schemes and return the
/// path component
/// \param uri_string a uri or local path to test and convert
/// \param supported_schemes the set of URI schemes that should be accepted
/// \param accept_local_paths if true, allow an absolute path
/// \return the path portion of the URI
arrow::Result<std::string> PathFromUriHelper(const std::string& uri_string,
                                             std::vector<std::string> supported_schemes,
                                             bool accept_local_paths,
                                             AuthorityHandlingBehavior authority_handling);

/// \brief Return files matching the glob pattern on the filesystem
///
/// Globbing starts from the root of the filesystem.
ARROW_EXPORT
arrow::Result<FileInfoVector> GlobFiles(const std::shared_ptr<FileSystem>& filesystem, const std::string& glob);

extern FileSystemGlobalOptions global_options;

ARROW_EXPORT
arrow::Status PathNotFound(std::string_view path);

}  // namespace internal
}  // namespace fs
}  // namespace arrow
