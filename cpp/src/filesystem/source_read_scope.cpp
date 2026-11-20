// Copyright 2026 Zilliz
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

#include "milvus-storage/filesystem/source_read_scope.h"

#include <algorithm>
#include <map>

namespace milvus_storage {
namespace {

bool IsAsciiAlphaNumeric(unsigned char value) {
  return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') || (value >= '0' && value <= '9');
}

std::string LowerAscii(std::string_view value) {
  std::string result(value);
  for (auto& ch : result) {
    if (ch >= 'A' && ch <= 'Z') {
      ch += 'a' - 'A';
    }
  }
  return result;
}

int HexDigit(unsigned char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  return -1;
}

std::optional<std::string> DecodeComponent(std::string_view input) {
  std::string result;
  result.reserve(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    auto ch = static_cast<unsigned char>(input[i]);
    if (ch == '%') {
      if (i + 2 >= input.size()) {
        return std::nullopt;
      }
      const auto high = HexDigit(input[i + 1]);
      const auto low = HexDigit(input[i + 2]);
      if (high < 0 || low < 0) {
        return std::nullopt;
      }
      ch = static_cast<unsigned char>((high << 4) | low);
      i += 2;
    }
    // Different URI consumers disagree on encoded separators and backslashes.
    // Reject them instead of allowing an authorization check and I/O to resolve
    // the same spelling to different resources.
    if (ch < 0x20 || ch == 0x7f || ch == '/' || ch == '\\') {
      return std::nullopt;
    }
    result.push_back(static_cast<char>(ch));
  }
  if (result.empty() || result == "." || result == "..") {
    return std::nullopt;
  }
  return result;
}

std::optional<std::vector<std::string>> DecodePath(std::string_view path) {
  std::vector<std::string> components;
  while (!path.empty()) {
    const auto slash = path.find('/');
    const auto component = path.substr(0, slash);
    auto decoded = DecodeComponent(component);
    if (!decoded) {
      return std::nullopt;
    }
    components.push_back(std::move(*decoded));
    if (slash == std::string_view::npos) {
      break;
    }
    path.remove_prefix(slash + 1);
  }
  return components;
}

std::string EncodeComponent(std::string_view input) {
  constexpr char kHex[] = "0123456789ABCDEF";
  std::string result;
  for (unsigned char ch : input) {
    if (IsAsciiAlphaNumeric(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~') {
      result.push_back(static_cast<char>(ch));
    } else {
      result.push_back('%');
      result.push_back(kHex[ch >> 4]);
      result.push_back(kHex[ch & 0x0f]);
    }
  }
  return result;
}

bool ValidContainer(std::string_view value) {
  return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '-';
  });
}

bool ValidEndpoint(std::string_view value) {
  if (value.empty() || value.front() == '.' || value.back() == '.') {
    return false;
  }
  bool previous_dot = false;
  for (unsigned char ch : value) {
    if (!IsAsciiAlphaNumeric(ch) && ch != '-' && ch != '.') {
      return false;
    }
    if (ch == '.' && previous_dot) {
      return false;
    }
    previous_dot = ch == '.';
  }
  return true;
}

std::optional<std::string> DecodeQueryValue(std::string_view input) {
  std::string result;
  for (size_t i = 0; i < input.size(); ++i) {
    auto ch = static_cast<unsigned char>(input[i]);
    // Do not choose between literal-plus and form-urlencoded space semantics.
    // Azure SDK escapes literal '+' as %2B.
    if (ch == '+')
      return std::nullopt;
    if (ch == '%') {
      if (i + 2 >= input.size())
        return std::nullopt;
      const auto high = HexDigit(input[i + 1]);
      const auto low = HexDigit(input[i + 2]);
      if (high < 0 || low < 0)
        return std::nullopt;
      ch = static_cast<unsigned char>((high << 4) | low);
      i += 2;
    }
    if (ch < 0x20 || ch == 0x7f || ch == '\\')
      return std::nullopt;
    result.push_back(static_cast<char>(ch));
  }
  return result;
}

std::optional<std::map<std::string, std::string>> ParseQuery(std::string_view query) {
  std::map<std::string, std::string> result;
  if (query.empty() || query.back() == '&')
    return std::nullopt;
  while (!query.empty()) {
    const auto amp = query.find('&');
    const auto item = query.substr(0, amp);
    const auto equal = item.find('=');
    if (equal == std::string_view::npos)
      return std::nullopt;
    const auto name = item.substr(0, equal);
    // Known SDK parameter names are literal lowercase ASCII. Reject aliases
    // such as encoded names instead of leaving their meaning to the service.
    if (name.empty() || !std::all_of(name.begin(), name.end(), [](char ch) { return ch >= 'a' && ch <= 'z'; }))
      return std::nullopt;
    auto value = DecodeQueryValue(item.substr(equal + 1));
    if (!value || !result.emplace(std::string(name), std::move(*value)).second)
      return std::nullopt;
    if (amp == std::string_view::npos)
      break;
    query.remove_prefix(amp + 1);
  }
  return result;
}

}  // namespace

std::optional<AzureSourceResource> AzureSourceResource::Parse(std::string_view uri, std::string_view expected_account) {
  if (expected_account.empty() || uri.find_first_of("?#\\") != std::string_view::npos) {
    return std::nullopt;
  }
  const auto separator = uri.find("://");
  if (separator == std::string_view::npos) {
    return std::nullopt;
  }
  const auto scheme = LowerAscii(uri.substr(0, separator));
  if (scheme != "https" && scheme != "abfss") {
    return std::nullopt;
  }
  uri.remove_prefix(separator + 3);
  const auto slash = uri.find('/');
  if (slash == std::string_view::npos) {
    return std::nullopt;
  }
  auto authority = uri.substr(0, slash);
  auto path = DecodePath(uri.substr(slash + 1));
  if (!path) {
    return std::nullopt;
  }

  AzureSourceResource result;
  if (scheme == "abfss") {
    const auto at = authority.find('@');
    if (at == std::string_view::npos || authority.find('@', at + 1) != std::string_view::npos) {
      return std::nullopt;
    }
    result.container = std::string(authority.substr(0, at));
    authority.remove_prefix(at + 1);
  } else {
    if (path->empty() || authority.find('@') != std::string_view::npos) {
      return std::nullopt;
    }
    result.container = std::move(path->front());
    path->erase(path->begin());
  }
  if (!ValidContainer(result.container)) {
    return std::nullopt;
  }
  const auto host = LowerAscii(authority);
  const auto account_end = host.find('.');
  if (account_end == std::string::npos) {
    return std::nullopt;
  }
  result.account = host.substr(0, account_end);
  if (result.account != expected_account ||
      !std::all_of(result.account.begin(), result.account.end(), IsAsciiAlphaNumeric)) {
    return std::nullopt;
  }
  const auto service = std::string_view(host).substr(account_end + 1);
  if (service.starts_with("blob.") && scheme == "https") {
    result.endpoint_suffix = service.substr(5);
  } else if (service.starts_with("dfs.")) {
    result.endpoint_suffix = service.substr(4);
  } else {
    return std::nullopt;
  }
  if (!ValidEndpoint(result.endpoint_suffix)) {
    return std::nullopt;
  }
  result.path = std::move(*path);
  return result;
}

std::string AzureSourceResource::CanonicalUri() const {
  auto result = "https://" + account + ".blob." + endpoint_suffix + "/" + EncodeComponent(container);
  for (const auto& component : path) {
    result += "/" + EncodeComponent(component);
  }
  return result;
}

bool AzureSourceResource::Contains(const AzureSourceResource& resource) const {
  return account == resource.account && endpoint_suffix == resource.endpoint_suffix &&
         container == resource.container && path.size() <= resource.path.size() &&
         std::equal(path.begin(), path.end(), resource.path.begin());
}

std::optional<AzureSourceHttpRequest> AzureSourceHttpRequest::Parse(std::string_view method,
                                                                    std::string_view unsigned_uri,
                                                                    std::string_view expected_account) {
  if ((method != "GET" && method != "HEAD") || !unsigned_uri.starts_with("https://") ||
      unsigned_uri.find('#') != std::string_view::npos)
    return std::nullopt;
  const auto query_start = unsigned_uri.find('?');
  auto resource = AzureSourceResource::Parse(unsigned_uri.substr(0, query_start), expected_account);
  if (!resource)
    return std::nullopt;
  if (query_start == std::string_view::npos)
    return AzureSourceHttpRequest{std::move(*resource), false};
  // The supported SDK list operations always address the container itself.
  if (method != "GET" || !resource->path.empty())
    return std::nullopt;
  auto query = ParseQuery(unsigned_uri.substr(query_start + 1));
  if (!query)
    return std::nullopt;
  const auto host_end = unsigned_uri.find('/', 8);
  const auto host = LowerAscii(unsigned_uri.substr(8, host_end - 8));
  const bool blob = host == resource->account + ".blob." + resource->endpoint_suffix;
  std::string directory;
  if (blob) {
    if (!query->contains("comp") || query->at("comp") != "list" || !query->contains("restype") ||
        query->at("restype") != "container" || !query->contains("prefix"))
      return std::nullopt;
    directory = query->at("prefix");
    // A non-empty un-delimited Blob prefix includes sibling names.
    if (!directory.empty() && directory.back() != '/')
      return std::nullopt;
    for (const auto& [key, value] : *query) {
      if (key == "comp" || key == "restype" || key == "prefix" || key == "marker" || key == "maxresults")
        continue;
      if ((key == "delimiter" && value == "/") || (key == "include" && value == "metadata"))
        continue;
      return std::nullopt;
    }
  } else {
    if (!query->contains("resource") || query->at("resource") != "filesystem" || !query->contains("directory") ||
        !query->contains("recursive") || (query->at("recursive") != "true" && query->at("recursive") != "false"))
      return std::nullopt;
    directory = query->at("directory");
    for (const auto& [key, value] : *query) {
      if (key != "resource" && key != "directory" && key != "recursive" && key != "continuation" && key != "maxresults")
        return std::nullopt;
    }
  }
  const auto binding =
      AzureSourceStorageBinding::Create(resource->account, resource->endpoint_suffix, resource->container);
  if (!binding)
    return std::nullopt;
  // Query decoding happened once above. ResolvePath consumes decoded keys and
  // does not reinterpret literal percent sequences as another URI encoding.
  auto target = binding->ResolvePath(directory);
  if (!target)
    return std::nullopt;
  return AzureSourceHttpRequest{std::move(*target), true};
}

std::optional<AzureSourceStorageBinding> AzureSourceStorageBinding::Create(std::string_view account,
                                                                           std::string_view endpoint_suffix,
                                                                           std::string_view container) {
  if (account.size() < 3 || account.size() > 24 ||
      !std::all_of(account.begin(), account.end(),
                   [](unsigned char ch) { return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9'); }) ||
      !ValidEndpoint(endpoint_suffix) || !ValidContainer(container)) {
    return std::nullopt;
  }
  auto root = AzureSourceResource::Parse(
      "https://" + std::string(account) + ".blob." + std::string(endpoint_suffix) + "/" + std::string(container),
      account);
  if (!root)
    return std::nullopt;
  return AzureSourceStorageBinding(std::move(*root));
}

std::optional<AzureSourceResource> AzureSourceStorageBinding::ResolveUri(std::string_view uri) const {
  const auto separator = uri.find("://");
  if (separator == std::string_view::npos || uri.find_first_of("?#\\") != std::string_view::npos) {
    return std::nullopt;
  }
  const auto scheme = LowerAscii(uri.substr(0, separator));
  std::optional<AzureSourceResource> resource;
  if (scheme == "https" || scheme == "abfss") {
    resource = AzureSourceResource::Parse(uri, container_.account);
  } else if (scheme == "azure" || scheme == "az") {
    uri.remove_prefix(separator + 3);
    const auto slash = uri.find('/');
    if (slash == std::string_view::npos)
      return std::nullopt;
    const auto authority = uri.substr(0, slash);
    const auto path = uri.substr(slash + 1);
    const auto base = "https://" + container_.account + ".blob." + container_.endpoint_suffix + "/";
    if (LowerAscii(authority) == container_.endpoint_suffix) {
      // Milvus URI: preserve and subsequently check the explicit container.
      resource = AzureSourceResource::Parse(base + std::string(path), container_.account);
    } else if (scheme == "az" && authority == container_.container) {
      // Lance URI: the exact selected binding supplies the absent endpoint.
      resource = AzureSourceResource::Parse(base + container_.container + "/" + std::string(path), container_.account);
    }
  }
  if (!resource || !container_.Contains(*resource))
    return std::nullopt;
  return resource;
}

std::optional<AzureSourceResource> AzureSourceStorageBinding::ResolvePath(std::string_view path) const {
  auto resource = container_;
  while (!path.empty()) {
    const auto slash = path.find('/');
    const auto component = path.substr(0, slash);
    if (component.empty() || component == "." || component == ".." ||
        std::any_of(component.begin(), component.end(),
                    [](unsigned char ch) { return ch < 0x20 || ch == 0x7f || ch == '\\'; })) {
      return std::nullopt;
    }
    resource.path.emplace_back(component);
    if (slash == std::string_view::npos)
      break;
    path.remove_prefix(slash + 1);
  }
  return resource;
}

std::optional<size_t> FindSourceRoot(const std::vector<AzureSourceResource>& roots,
                                     const AzureSourceResource& resource) {
  std::optional<size_t> found;
  for (size_t i = 0; i < roots.size(); ++i) {
    if (roots[i].Contains(resource) && (!found || roots[i].path.size() > roots[*found].path.size())) {
      found = i;
    }
  }
  return found;
}

}  // namespace milvus_storage
