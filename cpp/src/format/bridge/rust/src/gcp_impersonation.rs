// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! GCP Service Account Impersonation for lance-io.
//!
//! Neither `object_store` (lance-io's default GCS backend) nor `opendal`
//! natively supports the "VM default SA → IAM `generateAccessToken` →
//! impersonated target SA" flow. The closest config keys both expect a JSON
//! file path or already-issued credential, not a target-SA email.
//!
//! This module plugs the missing piece in by implementing two traits:
//!
//! * [`ImpersonatingGcsCredentialProvider`] — `object_store::CredentialProvider`
//!   that, on each `get_credential()` call, returns a cached impersonated
//!   token and refreshes it ahead of expiry.
//! * [`ImpersonatingGcsStoreProvider`] — `lance_io::object_store::ObjectStoreProvider`
//!   that builds a `GoogleCloudStorageBuilder` wired to the credential
//!   provider above, and is registered against the `gs` scheme to override
//!   lance-io's default GCS provider for opens that opt in.
//!
//! Wiring lives in `lance_bridgeimpl.rs`, which extracts the bridge-private
//! `gcp_target_service_account` and `gcp_credential_refresh_secs` keys from
//! `storage_options` and installs this provider into a per-call `Session`'s
//! `ObjectStoreRegistry`.
//!
//! # Why the two Google endpoint URLs are inlined here
//!
//! This module hand-rolls two HTTPS calls:
//! 1. `GET http://metadata.google.internal/.../service-accounts/default/token`
//!    — fetch the VM default SA's token.
//! 2. `POST https://iamcredentials.googleapis.com/v1/projects/-/`
//!    `serviceAccounts/{target}:generateAccessToken` — exchange it for
//!    an impersonated bearer.
//!
//! No crate in our tree exposes these as public API:
//!
//! * **`object_store::gcp`** hand-rolls its own GCP auth with `reqwest`
//!   + `ring`. All token-fetching code is crate-private; only
//!   `GcpCredential`, `GoogleConfigKey`, and the builder are public, and
//!   there is no impersonation config key.
//! * **`reqsign::google`** (used by `opendal`) has a `GoogleTokenLoader`
//!   that fetches from the metadata server, but `GoogleToken::access_token`
//!   is `pub(crate)` by design (commented "don't allow get token from
//!   reqsign") — the bearer is sealed for use by `GoogleSigner` only,
//!   so we cannot extract it into a `GcpCredential`. Its
//!   `ImpersonatedServiceAccount` variant is the `authorized_user`
//!   (refresh-token) flow, not VM-SA source.
//!
//! Interface types (`CredentialProvider`, `GcpCredential`,
//! `GoogleCloudStorageBuilder`, `ObjectStoreProvider`) are reused as-is;
//! only the auth business logic — which no upstream exposes — is local.
//! Pulling in `google-cloud-iam-credentials-v1` would remove the URLs
//! but drag in the full gRPC stack (`tonic`/`prost`) for two JSON POSTs,
//! which isn't worth it. The URLs are stable documented GCP endpoints.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use object_store::{
    gcp::{GcpCredential, GoogleCloudStorageBuilder, GoogleConfigKey},
    ClientOptions, CredentialProvider, ObjectStore as OSObjectStore, RetryConfig,
    Result as ObjectStoreResult,
};
use serde::Deserialize;
use snafu::location;
use std::str::FromStr;
use tokio::sync::RwLock;
use url::Url;

// lance-core's error types aren't a direct dep of the bridge; re-use lance's
// re-export (`lance::{Error, Result}` forwards to `lance_core`).
use lance::{Error as LanceError, Result as LanceResult};
use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, DEFAULT_CLOUD_IO_PARALLELISM,
};

/// lance-io's `DEFAULT_CLOUD_BLOCK_SIZE` is crate-private; mirror its 64 KiB
/// value so opens through this provider behave the same as the stock GCS one.
const GCS_DEFAULT_BLOCK_SIZE: usize = 64 * 1024;
/// Mirrors lance-io's hard-coded download retry count (also crate-private).
const GCS_DEFAULT_DOWNLOAD_RETRIES: usize = 3;

/// Default IAM `generateAccessToken` lifetime in seconds (max without an org
/// policy raising the cap).
pub const DEFAULT_TOKEN_LIFETIME_SECS: u64 = 3600;

/// How long before the cached token's expiry we trigger a refresh. Mirrors
/// the AWS path's `REFRESH_OFFSET_SECS = 300` so callers see consistent
/// refresh behavior across providers.
pub const REFRESH_OFFSET_SECS: u64 = 300;

/// Scope passed to `generateAccessToken`. `cloud-platform` is the broadest
/// OAuth scope; the actual GCS permissions come from IAM bindings on the
/// target SA, so a wide scope here doesn't grant anything extra.
const TOKEN_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

const METADATA_TOKEN_URL: &str =
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";

/// Caps on the two outbound HTTPS calls (metadata server, IAM). Without these,
/// `reqwest::Client::default()` has no timeout, so a stalled metadata server
/// or IAM 5xx-with-keepalive leaves the in-flight refresh holding the
/// `RwLock` write guard forever — tokio's writer-preferring policy then
/// blocks every concurrent `get_credential` reader. Typical values observed:
/// metadata ~10ms, IAM ~500ms; 30s total leaves plenty of margin for jitter
/// while bounding the worst-case stall.
const HTTP_CONNECT_TIMEOUT_SECS: u64 = 10;
const HTTP_REQUEST_TIMEOUT_SECS: u64 = 30;

fn build_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(HTTP_CONNECT_TIMEOUT_SECS))
        .timeout(Duration::from_secs(HTTP_REQUEST_TIMEOUT_SECS))
        .build()
        .expect("reqwest client builder: valid config")
}

/// Format a `generateAccessToken` URL for `target_sa`. We use the
/// `projects/-` shortcut so the caller doesn't have to know the target SA's
/// project (Google IAM resolves it from the email).
fn impersonation_url(target_sa: &str) -> String {
    format!(
        "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{}:generateAccessToken",
        target_sa
    )
}

#[derive(Deserialize)]
struct MetadataTokenResponse {
    access_token: String,
}

#[derive(Deserialize)]
struct GenerateAccessTokenResponse {
    #[serde(rename = "accessToken")]
    access_token: String,
    /// RFC3339 timestamp, e.g. `"2026-04-17T12:34:56Z"`.
    #[serde(rename = "expireTime")]
    expire_time: String,
}

/// Neutral store name used in `object_store::Error::Generic` from the shared
/// token-fetch helper; keeps error messages meaningful across both Lance
/// (cached provider) and iceberg (one-shot) callers.
const IMPERSONATION_STORE_NAME: &str = "gcp_impersonation";

/// Run the VM-SA → IAM `generateAccessToken(target_sa)` exchange end-to-end
/// and return the raw `accessToken` + `expireTime`. Shared by the cached
/// Lance `CredentialProvider` and the one-shot iceberg bridge path.
async fn fetch_impersonated_access_token(
    http_client: &reqwest::Client,
    target_sa: &str,
    token_lifetime: Duration,
) -> ObjectStoreResult<GenerateAccessTokenResponse> {
    // 1. Get the VM's default-SA OAuth token from the GCE metadata server.
    //    The `default` alias resolves to whatever SA the VM is configured
    //    with (no need for callers to name it).
    let vm_resp = http_client
        .get(METADATA_TOKEN_URL)
        .header("Metadata-Flavor", "Google")
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| object_store::Error::Generic {
            store: IMPERSONATION_STORE_NAME,
            source: format!(
                "metadata server token request failed (this code path requires running on a \
                 GCE VM with a default service account attached): {e}"
            )
            .into(),
        })?;
    let vm_token: MetadataTokenResponse = vm_resp.json().await.map_err(|e| object_store::Error::Generic {
        store: IMPERSONATION_STORE_NAME,
        source: format!("metadata token response was not valid JSON: {e}").into(),
    })?;

    // 2. Use the VM token as the bearer to call IAM `generateAccessToken`
    //    on the target SA. The VM SA needs `roles/iam.serviceAccountTokenCreator`
    //    on the target SA — failures here usually mean that binding is missing.
    let body = serde_json::json!({
        "scope": [TOKEN_SCOPE],
        "lifetime": format!("{}s", token_lifetime.as_secs()),
    });
    let iam_resp = http_client
        .post(impersonation_url(target_sa))
        .bearer_auth(&vm_token.access_token)
        .json(&body)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| object_store::Error::Generic {
            store: IMPERSONATION_STORE_NAME,
            source: format!(
                "IAM generateAccessToken({target_sa}) failed (the VM SA likely lacks \
                 roles/iam.serviceAccountTokenCreator on the target SA): {e}"
            )
            .into(),
        })?;
    iam_resp.json().await.map_err(|e| object_store::Error::Generic {
        store: IMPERSONATION_STORE_NAME,
        source: format!("generateAccessToken response was not valid JSON: {e}").into(),
    })
}

/// One-shot impersonated bearer fetch (no caching, no refresh).
///
/// Used by the iceberg bridge's `plan_files` path: iceberg-rust's
/// `gcs_config_parse` doesn't recognize `gcs.service-account` as an
/// impersonation target, so the bridge intercepts the key, calls this, and
/// swaps it for `gcs.oauth2.token` (opendal bakes that into `GcsConfig.token`
/// with `usize::MAX` expiry). A 1-hour token is plenty for a transient
/// metadata/manifest read sweep — see
/// `docs/iceberg-gcp-impersonation-analysis.md` for why refresh isn't needed
/// here, in contrast to long-lived Lance scans.
pub async fn fetch_impersonated_bearer(
    target_sa: &str,
    token_lifetime: Duration,
) -> ObjectStoreResult<String> {
    let client = build_http_client();
    let resp = fetch_impersonated_access_token(&client, target_sa, token_lifetime).await?;
    Ok(resp.access_token)
}

#[derive(Clone)]
struct CachedToken {
    credential: Arc<GcpCredential>,
    /// Epoch milliseconds when this token expires.
    expires_at_ms: u64,
}

/// `object_store::CredentialProvider` that mints short-lived impersonated
/// tokens via the GCE metadata server + IAM Credentials API.
///
/// The instance maintains a single cached token guarded by an async `RwLock`.
/// `get_credential` is invoked by `object_store` on every outbound request,
/// so the hot path is a read-lock fast check; refresh only runs when the
/// cached token is within [`REFRESH_OFFSET_SECS`] of expiry. The
/// double-checked write-lock pattern (mirroring AWS's
/// `DynamicStorageOptionsCredentialProvider`) keeps concurrent refreshes
/// from stampeding the IAM endpoint.
pub struct ImpersonatingGcsCredentialProvider {
    target_sa: String,
    /// Lifetime requested from `generateAccessToken`. IAM caps this at 3600s
    /// without an org-policy override.
    token_lifetime: Duration,
    refresh_offset: Duration,
    http_client: reqwest::Client,
    cache: Arc<RwLock<Option<CachedToken>>>,
}

impl fmt::Debug for ImpersonatingGcsCredentialProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImpersonatingGcsCredentialProvider")
            .field("target_sa", &self.target_sa)
            .field("token_lifetime", &self.token_lifetime)
            .field("refresh_offset", &self.refresh_offset)
            .finish_non_exhaustive()
    }
}

impl ImpersonatingGcsCredentialProvider {
    pub fn new(target_sa: String, token_lifetime: Duration, refresh_offset: Duration) -> Self {
        Self {
            target_sa,
            token_lifetime,
            refresh_offset,
            http_client: build_http_client(),
            cache: Arc::new(RwLock::new(None)),
        }
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_millis() as u64
    }

    fn needs_refresh(&self, cached: &Option<CachedToken>) -> bool {
        match cached {
            None => true,
            Some(c) => Self::now_ms() + self.refresh_offset.as_millis() as u64 >= c.expires_at_ms,
        }
    }

    /// Fast path with read lock; on miss, escalate to write lock and refresh.
    /// Returns `Ok(None)` when the write lock is contended so the outer
    /// `get_credential` can back off briefly and retry — this matches the
    /// pattern lance-io uses on the AWS side.
    async fn try_get_credential(&self) -> ObjectStoreResult<Option<Arc<GcpCredential>>> {
        {
            let cached = self.cache.read().await;
            if !self.needs_refresh(&cached) {
                if let Some(c) = &*cached {
                    return Ok(Some(c.credential.clone()));
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
                return Ok(Some(c.credential.clone()));
            }
        }

        let token = self.fetch_impersonated_token().await?;
        *cache = Some(token.clone());
        Ok(Some(token.credential))
    }

    async fn fetch_impersonated_token(&self) -> ObjectStoreResult<CachedToken> {
        let iam_body =
            fetch_impersonated_access_token(&self.http_client, &self.target_sa, self.token_lifetime).await?;

        // Compute the expiry from IAM's RFC3339 `expireTime`. We rely on IAM's
        // clock rather than `now + lifetime` so clock skew between us and
        // Google's auth servers can't push us into stale-but-thinks-fresh.
        let expires_at_ms = parse_rfc3339_to_ms(&iam_body.expire_time).map_err(|e| {
            object_store::Error::Generic {
                store: IMPERSONATION_STORE_NAME,
                source: format!("could not parse expireTime '{}': {e}", iam_body.expire_time)
                    .into(),
            }
        })?;

        Ok(CachedToken {
            credential: Arc::new(GcpCredential {
                bearer: iam_body.access_token,
            }),
            expires_at_ms,
        })
    }
}

#[async_trait]
impl CredentialProvider for ImpersonatingGcsCredentialProvider {
    type Credential = GcpCredential;

    async fn get_credential(&self) -> ObjectStoreResult<Arc<Self::Credential>> {
        // Retry loop — `try_get_credential` returns `None` only when the write
        // lock is held by an in-flight refresh. Yield briefly and retry; the
        // refresher will populate the cache in well under the sleep budget.
        loop {
            if let Some(cred) = self.try_get_credential().await? {
                return Ok(cred);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

fn parse_rfc3339_to_ms(s: &str) -> Result<u64, String> {
    let dt = chrono::DateTime::parse_from_rfc3339(s).map_err(|e| e.to_string())?;
    // Reject pre-epoch timestamps outright instead of letting `i64 as u64`
    // wrap them to ~year 584,554,051. Otherwise `expires_at_ms` would be so
    // far in the future that `needs_refresh` never trips and the cached
    // bearer is silently used past its real expiry.
    u64::try_from(dt.timestamp_millis()).map_err(|_| format!("pre-epoch expireTime: {}", s))
}

/// `lance_io::ObjectStoreProvider` for the `gs` scheme that wires the
/// custom credential provider into a `GoogleCloudStorageBuilder`.
///
/// Registering this against `gs` in an `ObjectStoreRegistry` (see
/// `lance_bridgeimpl::open_dataset`) replaces lance-io's stock GCS provider
/// for that registry only — other schemes and other registries are unaffected.
#[derive(Debug)]
pub struct ImpersonatingGcsStoreProvider {
    target_sa: String,
    token_lifetime: Duration,
    refresh_offset: Duration,
}

impl ImpersonatingGcsStoreProvider {
    pub fn new(target_sa: String, token_lifetime: Duration, refresh_offset: Duration) -> Self {
        Self {
            target_sa,
            token_lifetime,
            refresh_offset,
        }
    }
}

#[async_trait]
impl ObjectStoreProvider for ImpersonatingGcsStoreProvider {
    async fn new_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> LanceResult<ObjectStore> {
        let block_size = params.block_size.unwrap_or(GCS_DEFAULT_BLOCK_SIZE);
        let storage_options: HashMap<String, String> =
            params.storage_options.clone().unwrap_or_default();

        // Forward any GCS-recognized config keys the caller passed (endpoint
        // overrides, retry knobs, etc.) — but never forward credential keys
        // like `google_storage_token`/`google_service_account`, which would
        // race with our impersonated provider. Filter explicitly rather than
        // relying on `as_gcs_options` so we keep the rule visible here.
        let credential_keys = [
            "google_service_account",
            "google_service_account_path",
            "service_account_path",
            "google_service_account_key",
            "service_account_key",
            "google_application_credentials",
            "google_storage_token",
        ];

        let mut builder = GoogleCloudStorageBuilder::new()
            .with_url(base_path.as_ref())
            .with_retry(RetryConfig::default())
            .with_client_options(ClientOptions::default());

        for (key, value) in storage_options.iter() {
            let lower = key.to_ascii_lowercase();
            if credential_keys.contains(&lower.as_str()) {
                continue;
            }
            if let Ok(cfg_key) = GoogleConfigKey::from_str(&lower) {
                builder = builder.with_config(cfg_key, value.clone());
            }
        }

        let credential_provider: Arc<dyn CredentialProvider<Credential = GcpCredential>> =
            Arc::new(ImpersonatingGcsCredentialProvider::new(
                self.target_sa.clone(),
                self.token_lifetime,
                self.refresh_offset,
            ));
        builder = builder.with_credentials(credential_provider);

        let built = builder.build().map_err(|e| LanceError::IO {
            source: Box::new(e),
            location: location!(),
        })?;
        let inner = Arc::new(built) as Arc<dyn OSObjectStore>;

        Ok(ObjectStore::new(
            inner,
            base_path,
            Some(block_size),
            params.object_store_wrapper.clone(),
            params.use_constant_size_upload_parts,
            // GCS list is lexically ordered (matches stock GcsStoreProvider).
            true,
            DEFAULT_CLOUD_IO_PARALLELISM,
            GCS_DEFAULT_DOWNLOAD_RETRIES,
            params.storage_options.as_ref(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rfc3339_basic() {
        // `2026-04-17T03:23:14Z` → known epoch ms
        let ms = parse_rfc3339_to_ms("2026-04-17T03:23:14Z").unwrap();
        // sanity bounds: between 2026-01-01 and 2027-01-01 in ms
        assert!(ms > 1_767_225_600_000);
        assert!(ms < 1_798_761_600_000);
    }

    #[test]
    fn parse_rfc3339_pre_epoch_rejected() {
        // An `expireTime` before 1970 would have timestamp_millis() < 0 and,
        // without explicit handling, wrap to a huge u64 that makes
        // `needs_refresh` never fire. Must surface as an error instead.
        let err = parse_rfc3339_to_ms("1969-12-31T23:59:59Z").unwrap_err();
        assert!(err.contains("pre-epoch"));
    }

    #[test]
    fn parse_rfc3339_malformed_rejected() {
        assert!(parse_rfc3339_to_ms("not-a-timestamp").is_err());
    }

    #[test]
    fn impersonation_url_format() {
        let url = impersonation_url("foo@bar.iam.gserviceaccount.com");
        assert_eq!(
            url,
            "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/\
             foo@bar.iam.gserviceaccount.com:generateAccessToken"
        );
    }

    #[test]
    fn needs_refresh_when_empty() {
        let provider = ImpersonatingGcsCredentialProvider::new(
            "x@y.iam.gserviceaccount.com".to_string(),
            Duration::from_secs(3600),
            Duration::from_secs(300),
        );
        assert!(provider.needs_refresh(&None));
    }

    #[test]
    fn needs_refresh_within_offset() {
        let provider = ImpersonatingGcsCredentialProvider::new(
            "x@y.iam.gserviceaccount.com".to_string(),
            Duration::from_secs(3600),
            Duration::from_secs(300),
        );
        // Token expires in 100s, refresh offset is 300s → must refresh.
        let expires_soon = ImpersonatingGcsCredentialProvider::now_ms() + 100_000;
        let cached = Some(CachedToken {
            credential: Arc::new(GcpCredential {
                bearer: "x".into(),
            }),
            expires_at_ms: expires_soon,
        });
        assert!(provider.needs_refresh(&cached));
    }

    #[test]
    fn no_refresh_when_fresh() {
        let provider = ImpersonatingGcsCredentialProvider::new(
            "x@y.iam.gserviceaccount.com".to_string(),
            Duration::from_secs(3600),
            Duration::from_secs(300),
        );
        let expires_far = ImpersonatingGcsCredentialProvider::now_ms() + 3_600_000;
        let cached = Some(CachedToken {
            credential: Arc::new(GcpCredential {
                bearer: "x".into(),
            }),
            expires_at_ms: expires_far,
        });
        assert!(!provider.needs_refresh(&cached));
    }
}
