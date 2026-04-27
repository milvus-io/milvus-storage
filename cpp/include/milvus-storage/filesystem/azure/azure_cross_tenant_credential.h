// Copyright 2025 Zilliz
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

#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <azure/core/context.hpp>
#include <azure/core/credentials/credentials.hpp>
#include <azure/core/http/transport.hpp>

namespace milvus_storage::fs {

/// \brief Azure TokenCredential that performs the IMDS → AAD two-hop OAuth2
/// federated client_assertion exchange to mint a customer-tenant Bearer.
///
/// The two-hop flow:
///   1. GET 169.254.169.254/metadata/identity/oauth2/token
///        ?api-version=2018-02-01
///        &resource=api://AzureADTokenExchange
///      → JWT signed by *our* tenant, audience=api://AzureADTokenExchange.
///   2. POST https://login.microsoftonline.com/{customer_tenant}/oauth2/v2.0/token
///      Body: client_id={customer_app}, scope=https://storage.azure.com/.default,
///            grant_type=client_credentials, client_assertion={Step-1 JWT},
///            client_assertion_type=urn:ietf:params:oauth:client-assertion-type:jwt-bearer
///      → Customer-tenant Bearer (~3600s lifetime).
///
/// This credential class is the C++ side mirror of `azure_federation.rs` in
/// the Rust bridge: same protocol, same audience constants. We need both
/// because the bridge handles Lance/Iceberg `plan_files` (Rust path) while
/// this class handles Iceberg parquet data reads through the C++ Arrow
/// Azure FS (since `IcebergFormatReader` calls `parquet::ParquetFormatReader`
/// with an `arrow::fs::FileSystem`).
///
/// `ClientAssertionCredential` from Azure Identity 1.10+ would let us do
/// this with just an assertion callback, but the version vendored via Conan
/// (1.7.0-beta.3) doesn't have it yet. Subclassing `TokenCredential`
/// directly is the portable path that works against any 1.x.
///
/// The cached bearer is refreshed when within `kRefreshOffset` seconds of
/// expiry; concurrent `GetToken` calls that miss the cache may both fetch
/// (no global mutex held during HTTP), but the cache stores whichever
/// completes last — credential thrashing is bounded by the small number
/// of in-flight FS operations rather than queue depth.
class AzureCrossTenantCredential final : public Azure::Core::Credentials::TokenCredential {
  public:
  /// \param tenant_id  Customer's Entra ID tenant ID (the AAD authority).
  /// \param client_id  Customer's App Registration client_id with a
  ///                   Federated Identity Credential trusting our MI.
  AzureCrossTenantCredential(std::string tenant_id, std::string client_id);

  ~AzureCrossTenantCredential() override = default;

  Azure::Core::Credentials::AccessToken GetToken(
      Azure::Core::Credentials::TokenRequestContext const& tokenRequestContext,
      Azure::Core::Context const& context) const override;

  private:
  /// Refresh the cached bearer when the remaining lifetime drops below this.
  /// Mirrors `REFRESH_OFFSET_SECS` in `azure_federation.rs`.
  static constexpr std::chrono::seconds kRefreshOffset{300};

  /// Fetch step-1 MI assertion from IMDS. Returns the JWT string.
  /// `transport` is reused across the two hops to share keep-alive.
  std::string FetchManagedIdentityAssertion(Azure::Core::Http::HttpTransport& transport,
                                            Azure::Core::Context const& context) const;

  /// Fetch step-2 customer-tenant bearer. `mi_assertion` is the JWT from
  /// step 1. Returns (bearer, expires_at).
  std::pair<std::string, Azure::DateTime> ExchangeForStorageBearer(Azure::Core::Http::HttpTransport& transport,
                                                                   std::string const& mi_assertion,
                                                                   Azure::Core::Context const& context) const;

  std::string tenant_id_;
  std::string client_id_;

  // Mutable state for cache. Mutex protects only cache entry, NOT the HTTP
  // calls — those run unlocked so concurrent misses don't serialize.
  mutable std::mutex cache_mu_;
  mutable std::optional<Azure::Core::Credentials::AccessToken> cached_;
};

}  // namespace milvus_storage::fs
