// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#include "format/deltalake/deltalake_common.h"
#include <filesystem>
#include <arrow/util/uri.h>

#include "milvus-storage/filesystem/local_durability.h"

namespace milvus_storage::deltalake {
namespace {
arrow::Result<std::string> MakeTableUrl(const StorageUri& location, const FilesystemBinding& binding) {
  std::string result;
  if (binding.is_local) {
    const auto path = (std::filesystem::path(binding.local_root) / location.key).lexically_normal();
    ARROW_ASSIGN_OR_RAISE(result, arrow::util::UriFromAbsolutePath(path.generic_string()));
  } else {
    // StorageUri decodes URI paths once; escape the resulting raw key exactly
    // once. In particular a literal "%2F" must remain an object-key component.
    ARROW_ASSIGN_OR_RAISE(auto escaped, arrow::util::UriFromAbsolutePath("/" + location.key));
    result = binding.scheme + "://" + binding.bucket + escaped.substr(7);
  }
  if (result.back() != '/') {
    result += '/';
  }
  return result;
}

}  // namespace

arrow::Result<std::string> ToExternalUri(const std::string& standard_uri, const ArrowFileSystemConfig& config) {
  if (config.storage_type == "local") {
    // Keep a file URL so collection-manifest path normalization cannot turn a
    // source reference into a path relative to the imported collection.
    return standard_uri;
  }
  // Parse the standard URL before constructing the external filesystem URI.
  // The shared StorageUri::Make helper requires a nonempty key, while a table
  // may live at bucket root.
  arrow::util::Uri parsed;
  ARROW_RETURN_NOT_OK(parsed.Parse(standard_uri));
  StorageUri uri;
  uri.scheme = parsed.scheme();
  uri.bucket_name = parsed.host();
  ARROW_ASSIGN_OR_RAISE(auto escaped, arrow::util::UriFromAbsolutePath(parsed.path()));
  uri.key = escaped.substr(8);
  uri.address = config.address;
  // A Delta table can live at bucket root, unlike a normal data-file URI.
  if (uri.key.empty()) {
    arrow::util::Uri address;
    if (address.Parse(uri.address).ok() && (address.scheme() == "http" || address.scheme() == "https")) {
      uri.address = address.host();
      if (address.port() > 0) {
        uri.address += ":" + std::to_string(address.port());
      }
    }
    return uri.scheme + "://" + uri.address + "/" + uri.bucket_name + "/";
  }
  return StorageUri::Make(uri);
}

arrow::Result<ResolvedLocation> ResolveLocation(const std::string& location,
                                                const std::shared_ptr<const api::Properties>& properties) {
  if (location.empty()) {
    return arrow::Status::Invalid("Delta table location must not be empty");
  }
  const bool is_file_url = location.rfind("file:", 0) == 0;
  if (location.find("://") != std::string::npos || is_file_url) {
    arrow::util::Uri input;
    ARROW_RETURN_NOT_OK(input.Parse(location));
    if (location.find_first_of("?#") != std::string::npos || !input.username().empty() || !input.password().empty()) {
      return arrow::Status::Invalid("Delta table URI must not contain credentials, a query, or a fragment");
    }
  }
  // Standard file URLs use the default local filesystem, not an extfs bucket.
  auto filesystem_path = location;
  arrow::util::Uri local_uri;
  if (is_file_url) {
    ARROW_RETURN_NOT_OK(local_uri.Parse(location));
    if (!local_uri.host().empty()) {
      return arrow::Status::Invalid("Delta file URL must not contain a host");
    }
    filesystem_path = local_uri.path();
  }
  ARROW_ASSIGN_OR_RAISE(auto config, FilesystemCache::resolve_config(properties, filesystem_path));
  if (is_file_url && config.storage_type != "local") {
    return arrow::Status::Invalid("Delta file URL requires a local filesystem configuration");
  }
  ARROW_ASSIGN_OR_RAISE(auto parsed, StorageUri::Parse(filesystem_path));
  if (config.storage_type == "local" && parsed.IsAbsoluteUri()) {
    return arrow::Status::NotImplemented("Delta local tables require a path or file URL");
  }
  auto resolved_properties = properties;
  std::string local_root;
  if (config.storage_type == "local") {
    std::error_code error;
    const auto root = std::filesystem::absolute(config.root_path.empty() ? "." : config.root_path, error);
    if (error)
      return arrow::Status::IOError("Cannot resolve Delta filesystem root: ", error.message());
    // Keep the configured namespace in persisted URLs, as other table formats
    // do. Resolve the filesystem separately before collapsing symlink/.., and
    // bind relative roots to this cwd before the shared cache lookup.
    ARROW_ASSIGN_OR_RAISE(config.root_path, PrepareLocalRoot(root.generic_string()));
    local_root = root.lexically_normal().generic_string();
    auto local_properties = std::make_shared<api::Properties>(*properties);
    (*local_properties)[PROPERTY_FS_ROOT_PATH] = config.root_path;
    if (!config.local_durability_root.empty()) {
      ARROW_ASSIGN_OR_RAISE(config.local_durability_root, PrepareLocalRoot(config.local_durability_root));
      (*local_properties)[PROPERTY_FS_LOCAL_DURABILITY_ROOT] = config.local_durability_root;
    }
    resolved_properties = std::move(local_properties);
  }
  const auto identity = config.GetCacheKey();
  ARROW_ASSIGN_OR_RAISE(auto filesystem, FilesystemCache::getInstance().get(resolved_properties, filesystem_path));
  auto scheme = parsed.scheme.empty() ? filesystem->type_name() : parsed.scheme;
  deltalake::FilesystemBinding binding{config.storage_type == "local", scheme, config.bucket_name,
                                       std::move(local_root)};
  ARROW_ASSIGN_OR_RAISE(auto table_url, MakeTableUrl(parsed, binding));
  return ResolvedLocation{std::move(config), std::move(filesystem), std::move(table_url), std::move(binding), identity};
}

arrow::Result<ResolvedDataFile> ResolveDataFile(const std::string& path,
                                                const ResolvedLocation& table,
                                                const std::shared_ptr<const api::Properties>& properties) {
  arrow::util::Uri uri;
  ARROW_RETURN_NOT_OK(uri.Parse(path));
  if (path.find_first_of("?#") != std::string::npos || !uri.username().empty() || !uri.password().empty()) {
    return arrow::Status::Invalid("Invalid Delta data-file URI");
  }
  if (table.binding.is_local) {
    if (uri.scheme() != "file" || !uri.host().empty()) {
      return arrow::Status::Invalid("Delta local data file requires a file URL without a host");
    }
    // Normalize before checking containment: C++ opens the footer before Rust
    // validates snapshot membership. Normalized URLs also share per-file cache
    // entries across equivalent percent spellings and dot segments.
    const auto normalized = std::filesystem::path(uri.path()).lexically_normal();
    auto relative = normalized.lexically_relative(table.binding.local_root);
    if (relative.empty() || *relative.begin() == "..") {
      return arrow::Status::Invalid("Delta data file is outside its filesystem root");
    }
    ARROW_ASSIGN_OR_RAISE(auto url, arrow::util::UriFromAbsolutePath(normalized.generic_string()));
    return ResolvedDataFile{std::move(url), relative.generic_string()};
  }
  ARROW_ASSIGN_OR_RAISE(auto config, FilesystemCache::resolve_config(properties, path));
  if (config.GetCacheKey() != table.filesystem_key) {
    return arrow::Status::Invalid("Delta data file uses a different filesystem from its snapshot");
  }
  ARROW_ASSIGN_OR_RAISE(auto parsed, StorageUri::Parse(path));
  if (parsed.scheme != table.binding.scheme || parsed.bucket_name != table.binding.bucket) {
    return arrow::Status::Invalid("Delta data file uses a different storage identity");
  }
  ARROW_ASSIGN_OR_RAISE(auto escaped, arrow::util::UriFromAbsolutePath("/" + parsed.key));
  return ResolvedDataFile{parsed.scheme + "://" + parsed.bucket_name + escaped.substr(7), parsed.key};
}

}  // namespace milvus_storage::deltalake
