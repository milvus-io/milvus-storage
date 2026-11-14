// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#include "milvus-storage/format/deltalake/deltalake_format.h"

#include <filesystem>

#include <arrow/util/uri.h>
#include <nlohmann/json.hpp>

#include "deltalake/deltalake_bridge.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage {
namespace {

arrow::Result<std::string> MakeTableUrl(const StorageUri& location,
                                        const ArrowFileSystemConfig& config,
                                        const std::string& scheme) {
  std::string result;
  if (config.storage_type == "local") {
    auto path = std::filesystem::path(config.root_path) / location.key;
    ARROW_ASSIGN_OR_RAISE(result, arrow::util::UriFromAbsolutePath(path.lexically_normal().string()));
  } else {
    // StorageUri decodes URI paths once; escape the resulting raw key exactly
    // once. In particular a literal "%2F" must remain an object-key component.
    ARROW_ASSIGN_OR_RAISE(auto escaped, arrow::util::UriFromAbsolutePath("/" + location.key));
    result = scheme + "://" + config.bucket_name + escaped.substr(7);
  }
  if (result.back() != '/') {
    result += '/';
  }
  return result;
}

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

}  // namespace

arrow::Result<std::vector<api::ColumnGroupFile>> DeltaLakeFormat::explore(
    const std::string& explore_dir, const std::shared_ptr<const api::Properties>& properties) {
  if (explore_dir.empty()) {
    return arrow::Status::Invalid("Delta table location must not be empty");
  }
  const bool is_file_url = explore_dir.rfind("file:", 0) == 0;
  if (explore_dir.find("://") != std::string::npos || is_file_url) {
    arrow::util::Uri input;
    ARROW_RETURN_NOT_OK(input.Parse(explore_dir));
    if (explore_dir.find_first_of("?#") != std::string::npos || !input.username().empty() ||
        !input.password().empty()) {
      return arrow::Status::Invalid("Delta table URI must not contain credentials, a query, or a fragment");
    }
  }
  // Standard file URLs use the default local filesystem, not an extfs bucket.
  auto filesystem_path = explore_dir;
  arrow::util::Uri local_uri;
  if (is_file_url) {
    ARROW_RETURN_NOT_OK(local_uri.Parse(explore_dir));
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
  if (config.storage_type == "local") {
    // Match the shared local filesystem: an empty root means cwd. Resolve it
    // once for both the table URL and the Rust filesystem binding.
    std::error_code error;
    auto root = std::filesystem::absolute(config.root_path.empty() ? "." : config.root_path, error);
    if (error) {
      return arrow::Status::IOError("Cannot resolve Delta filesystem root: ", error.message());
    }
    config.root_path = root.lexically_normal().string();
  }
  ARROW_ASSIGN_OR_RAISE(auto filesystem, FilesystemCache::getInstance().get(properties, filesystem_path));
  ARROW_ASSIGN_OR_RAISE(auto version, api::GetValue<int64_t>(properties, PROPERTY_READER_EXTTABLE_SNAPSHOT_ID));
  auto scheme = parsed.scheme.empty() ? filesystem->type_name() : parsed.scheme;
  ARROW_ASSIGN_OR_RAISE(auto table_url, MakeTableUrl(parsed, config, scheme));
  deltalake::FilesystemBinding binding{config.storage_type == "local", scheme, config.bucket_name,
                                       config.storage_type == "local" ? config.root_path : ""};
  ARROW_ASSIGN_OR_RAISE(auto snapshot, deltalake::BlockingDeltaSnapshot::Open(table_url, version, filesystem, binding));
  ARROW_ASSIGN_OR_RAISE(auto table_uri, ToExternalUri(table_url, config));
  const auto descriptor =
      nlohmann::json{{"version", 1}, {"table_uri", table_uri}, {"snapshot_version", snapshot->Version()}}.dump();
  std::vector<api::ColumnGroupFile> files;
  for (const auto& info : snapshot->Files()) {
    ARROW_ASSIGN_OR_RAISE(auto path, ToExternalUri(info.path, config));
    api::ColumnGroupFile file{std::move(path), 0, static_cast<int64_t>(info.row_count), {}};
    file.Set(api::kPropertyFileSize, info.file_size);
    file.Set(api::kPropertyMetadata, descriptor);
    files.push_back(std::move(file));
  }
  return files;
}

arrow::Result<std::shared_ptr<FormatReader>> DeltaLakeFormat::create_reader(
    const std::shared_ptr<arrow::Schema>&,
    const api::ColumnGroupFile&,
    const std::shared_ptr<const api::Properties>&,
    const std::vector<std::string>&,
    const std::function<std::string(const std::string&)>&) {
  return arrow::Status::NotImplemented("Delta Lake file readers are implemented in Stage 2");
}

arrow::Result<std::unique_ptr<FormatWriter>> DeltaLakeFormat::create_writer(
    const std::shared_ptr<arrow::fs::FileSystem>&,
    const std::shared_ptr<arrow::Schema>&,
    const std::string&,
    const std::string&,
    const std::shared_ptr<const api::Properties>&) {
  return arrow::Status::NotImplemented("Delta Lake format does not support writing");
}

}  // namespace milvus_storage
