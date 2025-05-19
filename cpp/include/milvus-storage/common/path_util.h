

#pragma once

#include <string>
#include "arrow/status.h"

namespace milvus_storage {

constexpr char kSep = '/';

static inline arrow::Status NotAFile(std::string_view path) {
  return arrow::Status::IOError("Not a regular file: " + std::string(path));
}

static inline bool HasTrailingSlash(std::string_view s) { return !s.empty() && s.back() == kSep; }

static inline std::string EnsureTrailingSlash(std::string_view v) {
  if (!v.empty() && !HasTrailingSlash(v)) {
    // XXX How about "C:" on Windows?  We probably don't want to turn it into "C:/"...
    // Unless the local filesystem always uses absolute paths
    return std::string(v) + kSep;
  } else {
    return std::string(v);
  }
}

static inline std::pair<std::string, std::string> GetAbstractPathParent(const std::string& s) {
  // XXX should strip trailing slash?

  auto pos = s.find_last_of(kSep);
  if (pos == std::string::npos) {
    // Empty parent
    return {{}, s};
  }
  return {s.substr(0, pos), s.substr(pos + 1)};
}

static inline std::string ConcatenateFilePath(const std::string& parent, const std::string& child) {
  return parent + kSep + child;
}

}  // namespace milvus_storage