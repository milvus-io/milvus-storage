// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Azure cross-tenant access via Managed Identity → customer-tenant bearer.
//!
//! Neither `object_store` nor `opendal`/`reqsign` natively supports this flow:
//!
//! * **`object_store::azure`**: has `WorkloadIdentityOAuthProvider` that takes
//!   a federated token *file*, but on plain Azure VMs we have no such file —
//!   only an IMDS-attached Managed Identity. Its built-in `ImdsManagedIdentity
//!   Provider` requests `https://storage.azure.com/` audience tokens against
//!   *our* tenant, which the customer's storage account rejects.
//! * **`reqsign 0.16`** (under `opendal 0.55`): same set of paths, same gap.
//!   Its `load_via_imds` is single-tenant only.
//!
//! This module hand-rolls the two-hop OAuth2 exchange:
//!
//! 1. `GET 169.254.169.254/metadata/identity/oauth2/token`
//!    `?resource=api://AzureADTokenExchange` — fetch a JWT signed by *our*
//!    tenant's STS that names the MI as subject. This JWT is intended to be
//!    used as a `client_assertion` (audience `api://AzureADTokenExchange`).
//! 2. `POST https://login.microsoftonline.com/{customer_tenant}/oauth2/v2.0/token`
//!    with that JWT as `client_assertion`, `grant_type=client_credentials`,
//!    `scope=https://storage.azure.com/.default`. AAD validates the customer
//!    App Registration's Federated Identity Credential trusts our MI and
//!    issues a customer-tenant bearer.
//!
//! The result is a short-lived bearer (~1h, AAD-decided) that authenticates
//! against the customer's storage account, without our process ever holding
//! the customer's secret/key.
//!
//! # How callers use this
//!
//! [`CrossTenantBearerCache`] is the cached, refresh-on-demand entry point —
//! both bridges share it:
//!
//! * **Iceberg** (`azure_adls_provider.rs`): the cache is wrapped in a custom
//!   `iceberg::io::Storage`. opendal `AzdlsConfig` has no bearer field, so
//!   we build the `Operator` with a placeholder `account_key` (any valid
//!   base64) to satisfy reqsign's "credential present" check, then attach
//!   an `HttpFetch` wrapper that overwrites `Authorization: Bearer ...` with
//!   our cache's current value before each outbound request.
//! * **Lance** (`azure_cross_tenant_provider.rs`): the cache is wrapped in
//!   an `object_store::CredentialProvider` returning
//!   `AzureCredential::BearerToken`, plugged into `MicrosoftAzureBuilder::
//!   with_credentials`. Clean — no placeholder hack needed.
//!
//! # Why the endpoint URLs are inlined
//!
//! Same reasoning as `gcp_impersonation.rs`: no Rust crate in our tree
//! exposes IMDS or AAD as a library call we could reuse. Pulling in
//! `azure_identity` would drag the full Azure SDK transitive tree for two
//! JSON HTTP calls, which isn't worth it. The URLs are stable documented
//! Azure endpoints.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use tokio::sync::RwLock;

/// IMDS endpoint for Azure Managed Identity. Fixed by the platform, not
/// configurable per VM. The hop 1 audience `api://AzureADTokenExchange` is
/// the well-known audience that AAD's `oauth2/v2.0/token` accepts as a
/// `client_assertion` for federated identity credentials.
const IMDS_TOKEN_URL: &str = "http://169.254.169.254/metadata/identity/oauth2/token";
const IMDS_API_VERSION: &str = "2018-02-01";
const MI_AUDIENCE: &str = "api://AzureADTokenExchange";

/// AAD authority host. Fixed for the public Azure cloud; sovereign clouds
/// (Azure China, Azure Government) use different hosts but cross-tenant FIC
/// across sovereign boundaries is not a typical scenario, and adding the
/// configurability would require plumbing through the C++ side. Revisit if
/// needed.
const AAD_AUTHORITY: &str = "https://login.microsoftonline.com";

/// Scope passed to AAD when requesting the storage bearer. The `.default`
/// suffix asks for whatever permissions the customer's App Registration has
/// been granted on `https://storage.azure.com/` — typically Storage Blob
/// Data Reader/Contributor RBAC roles.
const STORAGE_SCOPE: &str = "https://storage.azure.com/.default";

const CLIENT_ASSERTION_TYPE: &str =
    "urn:ietf:params:oauth:client-assertion-type:jwt-bearer";

/// How long before a cached bearer's expiry we trigger a refresh. Mirrors
/// the AWS / GCP paths so callers see consistent refresh behavior across
/// providers.
pub const REFRESH_OFFSET_SECS: u64 = 300;

const HTTP_CONNECT_TIMEOUT_SECS: u64 = 10;
const HTTP_REQUEST_TIMEOUT_SECS: u64 = 30;

/// Neutral store name used in error wrapping; matches the per-provider
/// pattern in `gcp_impersonation.rs`.
const STORE_NAME: &str = "azure_cross_tenant";

fn build_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(HTTP_CONNECT_TIMEOUT_SECS))
        .timeout(Duration::from_secs(HTTP_REQUEST_TIMEOUT_SECS))
        .build()
        .expect("reqwest client builder: valid config")
}

/// IMDS token endpoint response. We only care about the access_token; the
/// other fields are not used (we re-derive expiry from AAD's response, since
/// the second hop's bearer is what actually goes on the wire).
#[derive(Deserialize)]
struct ImdsTokenResponse {
    access_token: String,
}

/// AAD `oauth2/v2.0/token` response. `expires_in` is seconds-from-now for
/// successful client_credentials grants.
#[derive(Deserialize)]
struct AadTokenResponse {
    access_token: String,
    expires_in: u64,
}

/// Hop 1: IMDS → MI assertion JWT (audience=api://AzureADTokenExchange).
async fn fetch_mi_assertion(http: &reqwest::Client) -> Result<String, String> {
    let resp = http
        .get(IMDS_TOKEN_URL)
        .header("Metadata", "true")
        .query(&[
            ("api-version", IMDS_API_VERSION),
            ("resource", MI_AUDIENCE),
        ])
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| {
            format!(
                "IMDS token request failed (this code path requires running on an Azure \
                 VM/host with a system- or user-assigned Managed Identity attached): {e}"
            )
        })?;
    let body: ImdsTokenResponse = resp
        .json()
        .await
        .map_err(|e| format!("IMDS response was not valid JSON: {e}"))?;
    Ok(body.access_token)
}

/// Hop 2: MI assertion → customer-tenant storage bearer.
async fn exchange_for_storage_bearer(
    http: &reqwest::Client,
    tenant_id: &str,
    client_id: &str,
    mi_assertion: &str,
) -> Result<(String, SystemTime), String> {
    let url = format!("{}/{}/oauth2/v2.0/token", AAD_AUTHORITY, tenant_id);
    let form = [
        ("client_id", client_id),
        ("scope", STORAGE_SCOPE),
        ("grant_type", "client_credentials"),
        ("client_assertion_type", CLIENT_ASSERTION_TYPE),
        ("client_assertion", mi_assertion),
    ];
    let resp = http
        .post(&url)
        .form(&form)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| {
            format!(
                "AAD token exchange failed for tenant={tenant_id}, client_id={client_id} \
                 (the customer's App Registration likely has no Federated Identity \
                 Credential trusting our MI, or the audience/issuer/subject in the FIC \
                 don't match): {e}"
            )
        })?;
    let body: AadTokenResponse = resp
        .json()
        .await
        .map_err(|e| format!("AAD token response was not valid JSON: {e}"))?;
    // expires_in is seconds-from-now; convert to absolute SystemTime.
    let expires_at = SystemTime::now() + Duration::from_secs(body.expires_in);
    Ok((body.access_token, expires_at))
}

/// One full IMDS → AAD round-trip, no caching. Useful for tests and one-shot
/// callers; long-lived flows should use [`CrossTenantBearerCache`].
pub async fn fetch_cross_tenant_bearer(
    tenant_id: &str,
    client_id: &str,
) -> Result<(String, SystemTime), String> {
    let http = build_http_client();
    let assertion = fetch_mi_assertion(&http).await?;
    exchange_for_storage_bearer(&http, tenant_id, client_id, &assertion).await
}

#[derive(Clone)]
struct CachedBearer {
    bearer: Arc<String>,
    /// Epoch milliseconds when the cached bearer expires.
    expires_at_ms: u64,
}

/// Cached customer-tenant bearer for a single (tenant_id, client_id) target.
///
/// `current()` returns the cached bearer when fresh, or fetches+caches a
/// new one otherwise. Refresh is triggered when the current bearer is
/// within `refresh_offset` of expiry. A double-checked write-lock pattern
/// keeps concurrent `current()` calls from stampeding the AAD endpoint
/// (mirrors the GCP / AWS providers).
pub struct CrossTenantBearerCache {
    tenant_id: String,
    client_id: String,
    refresh_offset: Duration,
    http: reqwest::Client,
    cache: Arc<RwLock<Option<CachedBearer>>>,
}

impl std::fmt::Debug for CrossTenantBearerCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossTenantBearerCache")
            .field("tenant_id", &self.tenant_id)
            .field("client_id", &self.client_id)
            .field("refresh_offset", &self.refresh_offset)
            .finish_non_exhaustive()
    }
}

impl CrossTenantBearerCache {
    pub fn new(tenant_id: String, client_id: String, refresh_offset: Duration) -> Self {
        Self {
            tenant_id,
            client_id,
            refresh_offset,
            http: build_http_client(),
            cache: Arc::new(RwLock::new(None)),
        }
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_millis() as u64
    }

    fn needs_refresh(&self, cached: &Option<CachedBearer>) -> bool {
        match cached {
            None => true,
            Some(c) => Self::now_ms() + self.refresh_offset.as_millis() as u64 >= c.expires_at_ms,
        }
    }

    /// Fast path with read lock; on miss, escalate to write lock and refresh.
    /// Returns `Ok(None)` when the write lock is contended so the outer
    /// `current` can back off briefly and retry — this matches the pattern
    /// used by GCP / AWS providers.
    async fn try_get(&self) -> Result<Option<Arc<String>>, String> {
        {
            let cached = self.cache.read().await;
            if !self.needs_refresh(&cached) {
                if let Some(c) = &*cached {
                    return Ok(Some(c.bearer.clone()));
                }
            }
        }
        let Ok(mut cache) = self.cache.try_write() else {
            return Ok(None);
        };
        // Double-check after acquiring write lock — another task may have
        // just refreshed.
        if !self.needs_refresh(&cache) {
            if let Some(c) = &*cache {
                return Ok(Some(c.bearer.clone()));
            }
        }
        let assertion = fetch_mi_assertion(&self.http).await?;
        let (bearer, expires_at) =
            exchange_for_storage_bearer(&self.http, &self.tenant_id, &self.client_id, &assertion)
                .await?;
        let expires_at_ms = expires_at
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("expires_at before UNIX epoch: {e}"))?
            .as_millis() as u64;
        let entry = CachedBearer {
            bearer: Arc::new(bearer),
            expires_at_ms,
        };
        *cache = Some(entry.clone());
        Ok(Some(entry.bearer))
    }

    /// Return the current customer-tenant bearer, refreshing on demand.
    pub async fn current(&self) -> Result<Arc<String>, String> {
        loop {
            if let Some(b) = self.try_get().await? {
                return Ok(b);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// Convenience: wrap a `String` error as `object_store::Error::Generic` for
/// callers in the Lance bridge that need an `object_store::Result`.
pub fn into_object_store_err(msg: String) -> object_store::Error {
    object_store::Error::Generic {
        store: STORE_NAME,
        source: msg.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn needs_refresh_when_empty() {
        let cache = CrossTenantBearerCache::new(
            "tenant".into(),
            "client".into(),
            Duration::from_secs(REFRESH_OFFSET_SECS),
        );
        assert!(cache.needs_refresh(&None));
    }

    #[test]
    fn needs_refresh_within_offset() {
        let cache = CrossTenantBearerCache::new(
            "tenant".into(),
            "client".into(),
            Duration::from_secs(REFRESH_OFFSET_SECS),
        );
        // Token expires in 100s, refresh offset is 300s → must refresh.
        let expires_soon = CrossTenantBearerCache::now_ms() + 100_000;
        let cached = Some(CachedBearer {
            bearer: Arc::new("x".into()),
            expires_at_ms: expires_soon,
        });
        assert!(cache.needs_refresh(&cached));
    }

    #[test]
    fn no_refresh_when_fresh() {
        let cache = CrossTenantBearerCache::new(
            "tenant".into(),
            "client".into(),
            Duration::from_secs(REFRESH_OFFSET_SECS),
        );
        // Token expires in 1h, refresh offset is 300s → fresh.
        let expires_far = CrossTenantBearerCache::now_ms() + 3_600_000;
        let cached = Some(CachedBearer {
            bearer: Arc::new("x".into()),
            expires_at_ms: expires_far,
        });
        assert!(!cache.needs_refresh(&cached));
    }
}
