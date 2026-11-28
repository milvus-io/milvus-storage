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

#include "milvus-storage/source_read_context.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <limits>
#include <map>
#include <string_view>

namespace milvus_storage {
namespace {

std::atomic<uint64_t> next_context_identity{1};

// Match the native Azure producer's credential-target restriction. A runtime
// context must not turn a supplied directory root into a SAS exfiltration host.
constexpr std::array<std::string_view, 4> kAzureSourceAuthorities = {"core.windows.net", "core.chinacloudapi.cn",
                                                                     "core.usgovcloudapi.net", "core.cloudapi.de"};

SourceCredentialResponse Failure(LoonSourceCredentialStatus status) {
  SourceCredentialResponse result;
  result.status = status;
  return result;
}

std::optional<int> Decimal(std::string_view input) {
  if (input.empty() ||
      !std::all_of(input.begin(), input.end(), [](unsigned char ch) { return ch >= '0' && ch <= '9'; })) {
    return std::nullopt;
  }
  int value = 0;
  auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
  if (error != std::errc{} || end != input.data() + input.size()) {
    return std::nullopt;
  }
  return value;
}

int Hex(unsigned char ch) {
  if (ch >= '0' && ch <= '9')
    return ch - '0';
  if (ch >= 'a' && ch <= 'f')
    return ch - 'a' + 10;
  if (ch >= 'A' && ch <= 'F')
    return ch - 'A' + 10;
  return -1;
}

std::optional<std::string> DecodeQuery(std::string_view input) {
  std::string result;
  for (size_t i = 0; i < input.size(); ++i) {
    unsigned char ch = input[i];
    if (ch == '%') {
      if (i + 2 >= input.size())
        return std::nullopt;
      const auto high = Hex(input[i + 1]);
      const auto low = Hex(input[i + 2]);
      if (high < 0 || low < 0)
        return std::nullopt;
      ch = static_cast<unsigned char>((high << 4) | low);
      i += 2;
    } else if (ch == '+') {
      ch = ' ';
    }
    if (ch < 0x20 || ch == 0x7f)
      return std::nullopt;
    result.push_back(static_cast<char>(ch));
  }
  return result;
}

std::optional<std::map<std::string, std::string>> SasFields(std::string_view token) {
  if (token.empty() || token.size() > 65536)
    return std::nullopt;
  if (token.front() == '?')
    token.remove_prefix(1);
  std::map<std::string, std::string> fields;
  while (!token.empty()) {
    const auto amp = token.find('&');
    const auto pair = token.substr(0, amp);
    const auto equal = pair.find('=');
    if (equal == std::string_view::npos)
      return std::nullopt;
    auto key = DecodeQuery(pair.substr(0, equal));
    auto value = DecodeQuery(pair.substr(equal + 1));
    if (!key || key->empty() || !value || !fields.emplace(*key, *value).second)
      return std::nullopt;
    if (amp == std::string_view::npos)
      break;
    token.remove_prefix(amp + 1);
  }
  return fields;
}

// Azure's signed expiry is RFC3339. Fractions are truncated to milliseconds so
// local validity never extends beyond the signed instant. Do not include an
// invalid timestamp or the SAS in a diagnostic.
std::optional<int64_t> SignedExpiry(std::string_view value) {
  if (value.size() < 20 || value[4] != '-' || value[7] != '-' || value[10] != 'T' || value[13] != ':' ||
      value[16] != ':')
    return std::nullopt;
  const auto year = Decimal(value.substr(0, 4));
  const auto month = Decimal(value.substr(5, 2));
  const auto day = Decimal(value.substr(8, 2));
  const auto hour = Decimal(value.substr(11, 2));
  const auto minute = Decimal(value.substr(14, 2));
  const auto second = Decimal(value.substr(17, 2));
  if (!year || !month || !day || !hour || !minute || !second || *hour > 23 || *minute > 59 || *second > 59) {
    return std::nullopt;
  }
  const auto date = std::chrono::year{*year} / std::chrono::month{static_cast<unsigned>(*month)} /
                    std::chrono::day{static_cast<unsigned>(*day)};
  if (!date.ok())
    return std::nullopt;
  size_t offset = 19;
  int millis = 0;
  if (value[offset] == '.') {
    const auto start = ++offset;
    while (offset < value.size() && value[offset] >= '0' && value[offset] <= '9') ++offset;
    const auto fraction = value.substr(start, offset - start);
    if (fraction.empty() || fraction.size() > 9)
      return std::nullopt;
    auto padded = std::string(fraction.substr(0, 3));
    padded.append(3 - padded.size(), '0');
    millis = *Decimal(padded);
  }
  int timezone_minutes = 0;
  const auto timezone = value.substr(offset);
  if (timezone != "Z") {
    if (timezone.size() != 6 || (timezone[0] != '+' && timezone[0] != '-') || timezone[3] != ':') {
      return std::nullopt;
    }
    const auto zone_hour = Decimal(timezone.substr(1, 2));
    const auto zone_minute = Decimal(timezone.substr(4, 2));
    if (!zone_hour || !zone_minute || *zone_hour > 23 || *zone_minute > 59)
      return std::nullopt;
    timezone_minutes = (*zone_hour * 60 + *zone_minute) * (timezone[0] == '-' ? -1 : 1);
  }
  const auto instant = std::chrono::sys_days(date) + std::chrono::hours(*hour) +
                       std::chrono::minutes(*minute - timezone_minutes) + std::chrono::seconds(*second) +
                       std::chrono::milliseconds(millis);
  return std::chrono::duration_cast<std::chrono::milliseconds>(instant.time_since_epoch()).count();
}

bool ValidCredential(SourceCredentialResponse& response,
                     const SourceCredentialRequest& request,
                     const AzureSourceResource& root,
                     int64_t now) {
  if (response.kind != LOON_SOURCE_CREDENTIAL_KIND_AZURE_SAS || response.account_name != root.account)
    return false;
  const auto returned_root = AzureSourceResource::Parse(response.scope_root_uri, root.account);
  if (!returned_root || *returned_root != root)
    return false;
  auto fields = SasFields(response.sas_token);
  if (!fields || (*fields)["sig"].empty() || (*fields)["sr"] != "d")
    return false;
  // A credential must not change the already-authorized SDK operation (for
  // example by supplying comp, prefix or directory query parameters).
  constexpr std::string_view sas_keys[] = {"sv",    "spr",   "st",    "se",   "sip",  "sp",   "sr",   "sdd",
                                           "sig",   "skoid", "sktid", "skt",  "ske",  "sks",  "skv",  "saoid",
                                           "suoid", "scid",  "rscc",  "rscd", "rsce", "rscl", "rsct", "ses"};
  for (const auto& [key, value] : *fields) {
    if (std::find(std::begin(sas_keys), std::end(sas_keys), key) == std::end(sas_keys))
      return false;
  }
  const auto& permissions = (*fields)["sp"];
  if (permissions.empty() || permissions.find_first_not_of("rl") != std::string::npos ||
      permissions.find(request.operation == LOON_SOURCE_LIST ? 'l' : 'r') == std::string::npos)
    return false;
  const auto depth = Decimal((*fields)["sdd"]);
  if (!depth || static_cast<size_t>(*depth) != root.path.size())
    return false;
  const auto signed_expiry = SignedExpiry((*fields)["se"]);
  if (!signed_expiry || *signed_expiry <= now || response.expires_at_unix_ms <= now)
    return false;
  if (request.operation_deadline_unix_ms != 0 && *signed_expiry > request.operation_deadline_unix_ms)
    return false;
  if (const auto start = fields->find("st"); start != fields->end()) {
    const auto signed_start = SignedExpiry(start->second);
    if (!signed_start || *signed_start > now)
      return false;
  }
  // The issuer's reported cache lifetime cannot extend the actual signed SAS.
  response.expires_at_unix_ms = std::min(response.expires_at_unix_ms, *signed_expiry);
  response.scope_root_uri = root.CanonicalUri();
  if (response.sas_token.front() == '?')
    response.sas_token.erase(0, 1);
  return true;
}

}  // namespace

arrow::Result<std::shared_ptr<SourceReadContext>> SourceReadContext::Create(
    SourceReadContextOptions options, std::shared_ptr<const SourceCredentialProvider> provider, Clock clock) {
  if (!provider || options.account_name.size() < 3 || options.account_name.size() > 24 ||
      !std::all_of(options.account_name.begin(), options.account_name.end(),
                   [](unsigned char ch) { return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9'); }) ||
      options.allowed_root_uris.empty() || options.request_timeout_ms <= 0 || options.operation_deadline_unix_ms < 0) {
    return arrow::Status::Invalid("Invalid source-read context options");
  }
  std::vector<AzureSourceResource> roots;
  for (const auto& uri : options.allowed_root_uris) {
    auto root = AzureSourceResource::Parse(uri, options.account_name);
    if (!root || root->path.empty()) {
      return arrow::Status::Invalid("Source-read scope must contain explicit Azure directory roots");
    }
    if (std::find(kAzureSourceAuthorities.begin(), kAzureSourceAuthorities.end(), root->endpoint_suffix) ==
        kAzureSourceAuthorities.end()) {
      return arrow::Status::Invalid("Source-read scope must use an Azure Storage authority");
    }
    if (std::find(roots.begin(), roots.end(), *root) == roots.end())
      roots.push_back(std::move(*root));
  }
  if (!clock) {
    clock = [] {
      return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();
    };
  }
  auto identity = next_context_identity.load();
  do {
    if (identity == std::numeric_limits<uint64_t>::max()) {
      return arrow::Status::CapacityError("Source-read context identities exhausted");
    }
  } while (!next_context_identity.compare_exchange_weak(identity, identity + 1));
  return std::shared_ptr<SourceReadContext>(
      new SourceReadContext(identity, std::move(options), std::move(roots), std::move(provider), std::move(clock)));
}

SourceReadContext::SourceReadContext(uint64_t identity,
                                     SourceReadContextOptions options,
                                     std::vector<AzureSourceResource> roots,
                                     std::shared_ptr<const SourceCredentialProvider> provider,
                                     Clock clock)
    : identity_(identity),
      account_name_(std::move(options.account_name)),
      roots_(std::move(roots)),
      operation_deadline_unix_ms_(options.operation_deadline_unix_ms),
      request_timeout_ms_(options.request_timeout_ms),
      provider_(std::move(provider)),
      clock_(std::move(clock)) {}

std::string SourceReadContext::CacheIdentity(size_t root_index) const {
  return "source:" + std::to_string(identity_) + ":" + std::to_string(root_index);
}

void SourceReadContext::Cancel() { terminal_.store(LOON_SOURCE_CREDENTIAL_CANCELLED); }

LoonSourceCredentialStatus SourceReadContext::CheckState() const {
  auto state = terminal_.load();
  if (state == LOON_SOURCE_CREDENTIAL_OK && operation_deadline_unix_ms_ != 0 &&
      clock_() >= operation_deadline_unix_ms_) {
    terminal_.compare_exchange_strong(state, LOON_SOURCE_CREDENTIAL_EXPIRED);
  }
  return terminal_.load();
}

SourceReadTarget SourceReadContext::Resolve(std::string_view resource_uri) const {
  SourceReadTarget target;
  target.status = CheckState();
  if (target.status != LOON_SOURCE_CREDENTIAL_OK)
    return target;
  target.status = LOON_SOURCE_CREDENTIAL_DENIED;
  const auto resource = AzureSourceResource::Parse(resource_uri, account_name_);
  if (!resource)
    return target;
  const auto root = FindSourceRoot(roots_, *resource);
  if (!root)
    return target;
  target.status = LOON_SOURCE_CREDENTIAL_OK;
  target.root_index = *root;
  target.resource_uri = resource->CanonicalUri();
  target.scope_root_uri = roots_[*root].CanonicalUri();
  return target;
}

SourceReadTarget SourceReadContext::ResolveHttpRequest(std::string_view method, std::string_view unsigned_uri) const {
  SourceReadTarget target;
  target.status = CheckState();
  if (target.status != LOON_SOURCE_CREDENTIAL_OK)
    return target;
  target.status = LOON_SOURCE_CREDENTIAL_DENIED;
  const auto request = AzureSourceHttpRequest::Parse(method, unsigned_uri, account_name_);
  if (!request)
    return target;
  target = Resolve(request->resource.CanonicalUri());
  target.operation = request->is_list ? LOON_SOURCE_LIST : LOON_SOURCE_READ;
  return target;
}

SourceCredentialResponse SourceReadContext::FetchHttpRequest(std::string_view method,
                                                             std::string_view unsigned_uri) const {
  const auto target = ResolveHttpRequest(method, unsigned_uri);
  if (target.status != LOON_SOURCE_CREDENTIAL_OK)
    return Failure(target.status);
  // Fetch repeats the state/root check and validates the returned SAS against
  // this actual operation, including the corresponding 'r' or 'l' permission.
  return Fetch(target.resource_uri, target.operation);
}

SourceCredentialResponse SourceReadContext::Fetch(std::string_view resource_uri,
                                                  LoonSourceReadOperation operation) const {
  if (operation != LOON_SOURCE_READ && operation != LOON_SOURCE_LIST)
    return Failure(LOON_SOURCE_CREDENTIAL_DENIED);
  const auto target = Resolve(resource_uri);
  if (target.status != LOON_SOURCE_CREDENTIAL_OK)
    return Failure(target.status);
  const auto started = clock_();
  if (started > std::numeric_limits<int64_t>::max() - request_timeout_ms_)
    return Failure(LOON_SOURCE_CREDENTIAL_EXPIRED);
  auto deadline = started + request_timeout_ms_;
  if (operation_deadline_unix_ms_ != 0)
    deadline = std::min(deadline, operation_deadline_unix_ms_);
  if (started >= deadline)
    return Failure(LOON_SOURCE_CREDENTIAL_EXPIRED);
  SourceCredentialRequest request{target.resource_uri, target.scope_root_uri, operation, deadline,
                                  operation_deadline_unix_ms_};
  SourceCredentialResponse response;
  try {
    response = provider_->Fetch(request);
  } catch (...) {
    // A throwing provider violated its contract. Never expose exception text,
    // which can contain issuer request URLs, response bodies or credentials.
    return Failure(LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE);
  }
  if (response.status == LOON_SOURCE_CREDENTIAL_DENIED) {
    auto expected = LOON_SOURCE_CREDENTIAL_OK;
    terminal_.compare_exchange_strong(expected, LOON_SOURCE_CREDENTIAL_DENIED);
  }
  const auto state = CheckState();
  if (state != LOON_SOURCE_CREDENTIAL_OK)
    return Failure(state);
  const auto finished = clock_();
  if (finished >= deadline)
    return Failure(LOON_SOURCE_CREDENTIAL_TRANSIENT);
  switch (response.status) {
    case LOON_SOURCE_CREDENTIAL_OK:
      if (!ValidCredential(response, request, roots_[target.root_index], finished)) {
        return Failure(LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE);
      }
      return response;
    case LOON_SOURCE_CREDENTIAL_DENIED:
    case LOON_SOURCE_CREDENTIAL_EXPIRED:
    case LOON_SOURCE_CREDENTIAL_TRANSIENT:
    case LOON_SOURCE_CREDENTIAL_CANCELLED:
    case LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE:
      return Failure(response.status);
    default:
      return Failure(LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE);
  }
}

}  // namespace milvus_storage
