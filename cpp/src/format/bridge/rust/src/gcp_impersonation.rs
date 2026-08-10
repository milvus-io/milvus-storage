// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! GCP Service Account Impersonation for Lance and Iceberg.
//!
//! Neither `object_store` (lance-io's default GCS backend) nor `opendal`
//! natively supports the "VM default SA → IAM `generateAccessToken` →
//! impersonated target SA" flow. The closest config keys both expect a JSON
//! file path or already-issued credential, not a target-SA email.
//!
//! This module plugs the missing piece into both storage stacks:
//!
//! * [`ImpersonatingGcsCredentialProvider`] — `object_store::CredentialProvider`
//!   that, on each `get_credential()` call, returns a cached impersonated
//!   token and refreshes it ahead of expiry.
//! * [`ImpersonatingGcsStoreProvider`] — `lance_io::object_store::ObjectStoreProvider`
//!   that builds a `GoogleCloudStorageBuilder` wired to the credential
//!   provider above, and is registered against the `gs` scheme to override
//!   lance-io's default GCS provider for opens that opt in.
//! * [`GcpImpersonationStorageFactory`] / [`GcpImpersonationStorage`] —
//!   thin Iceberg wrappers that inject the current token and delegate storage
//!   behavior to `iceberg-storage-opendal`.
//!
//! Lance wiring lives in `lance_bridgeimpl.rs`, which extracts the bridge-private
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
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use iceberg::io::{
    FileMetadata, FileRead, FileWrite, GCS_DISABLE_CONFIG_LOAD, GCS_DISABLE_VM_METADATA,
    GCS_TOKEN, InputFile, OutputFile, Storage as IcebergStorage, StorageConfig, StorageFactory,
};
use iceberg::{Error as IcebergError, ErrorKind as IcebergErrorKind, Result as IcebergResult};
use iceberg_storage_opendal::OpenDalStorageFactory;
use object_store::{
    ClientOptions, CredentialProvider, ObjectStore as OSObjectStore, Result as ObjectStoreResult,
    RetryConfig,
    gcp::{GcpCredential, GoogleCloudStorageBuilder, GoogleConfigKey},
};
use serde::{Deserialize, Serialize};
use snafu::location;
use std::str::FromStr;
use tokio::sync::RwLock;
use url::Url;

// lance-core's error types aren't a direct dep of the bridge; re-use lance's
// re-export (`lance::{Error, Result}` forwards to `lance_core`).
use lance::session::Session;
use lance::{Error as LanceError, Result as LanceResult};
use lance_io::object_store::{
    DEFAULT_CLOUD_IO_PARALLELISM, ObjectStore, ObjectStoreParams, ObjectStoreProvider,
    ObjectStoreRegistry,
};

/// lance-io's `DEFAULT_CLOUD_BLOCK_SIZE` is crate-private; mirror its 64 KiB
/// value so opens through this provider behave the same as the stock GCS one.
const GCS_DEFAULT_BLOCK_SIZE: usize = 64 * 1024;
/// Mirrors lance-io's hard-coded download retry count (also crate-private).
const GCS_DEFAULT_DOWNLOAD_RETRIES: usize = 3;

/// How long before the cached token's expiry we trigger a refresh. Mirrors
/// the AWS path's `REFRESH_OFFSET_SECS = 300` so callers see consistent
/// refresh behavior across providers.
pub const REFRESH_OFFSET_SECS: u64 = 300;

pub(crate) const LANCE_TARGET_SERVICE_ACCOUNT: &str = "gcp_target_service_account";
pub(crate) const ICEBERG_TARGET_SERVICE_ACCOUNT: &str = "gcs.service-account";
pub(crate) const TOKEN_LIFETIME_SECONDS: &str = "gcp_credential_refresh_secs";

#[derive(Clone, Eq, PartialEq)]
pub(crate) struct GcpImpersonationConfig {
    target_sa: String,
    token_lifetime_secs: u64,
}

impl GcpImpersonationConfig {
    pub(crate) fn extract(
        options: &mut HashMap<String, String>,
        target_key: &str,
    ) -> anyhow::Result<Option<Self>> {
        let target_sa = options.remove(target_key).unwrap_or_default();
        let lifetime = options
            .remove(TOKEN_LIFETIME_SECONDS)
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        if target_sa.is_empty() {
            return Ok(None);
        }
        if !(900..=3600).contains(&lifetime) {
            anyhow::bail!("gcp_credential_refresh_secs must be in [900, 3600], got {lifetime}");
        }
        Ok(Some(Self {
            target_sa,
            token_lifetime_secs: lifetime,
        }))
    }
}

impl fmt::Debug for GcpImpersonationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcpImpersonationConfig")
            .finish_non_exhaustive()
    }
}

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
/// token-fetch helper used by both Lance and Iceberg.
const IMPERSONATION_STORE_NAME: &str = "gcp_impersonation";

/// Run the VM-SA → IAM `generateAccessToken(target_sa)` exchange end-to-end
/// and return the raw `accessToken` + `expireTime` for the shared refreshable
/// provider.
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

    /// Fast path with read lock; on miss, wait for the write lock and refresh.
    async fn get_credential_with<F, Fut>(&self, fetch: F) -> ObjectStoreResult<Arc<GcpCredential>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = ObjectStoreResult<CachedToken>>,
    {
        {
            let cached = self.cache.read().await;
            if !self.needs_refresh(&cached) {
                if let Some(c) = &*cached {
                    return Ok(c.credential.clone());
                }
            }
        }

        let mut cache = self.cache.write().await;

        // Double-check after acquiring write lock — another task may have
        // just refreshed.
        if !self.needs_refresh(&cache) {
            if let Some(c) = &*cache {
                return Ok(c.credential.clone());
            }
        }

        let token = fetch().await?;
        *cache = Some(token.clone());
        Ok(token.credential)
    }

    async fn fetch_impersonated_token(&self) -> ObjectStoreResult<CachedToken> {
        let iam_body =
            fetch_impersonated_access_token(&self.http_client, &self.target_sa, self.token_lifetime)
                .await?;

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
        self.get_credential_with(|| self.fetch_impersonated_token())
            .await
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
    credentials: Arc<ImpersonatingGcsCredentialProvider>,
}

impl ImpersonatingGcsStoreProvider {
    fn new(credentials: Arc<ImpersonatingGcsCredentialProvider>) -> Self {
        Self { credentials }
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
            params.storage_options().cloned().unwrap_or_default();

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

        let credentials: Arc<dyn CredentialProvider<Credential = GcpCredential>> =
            self.credentials.clone();
        builder = builder.with_credentials(credentials);

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
            params.storage_options(),
        ))
    }
}

pub(crate) async fn build_lance_provider(
    config: &GcpImpersonationConfig,
) -> LanceResult<Arc<dyn ObjectStoreProvider>> {
    let credentials = Arc::new(ImpersonatingGcsCredentialProvider::new(
        config.target_sa.clone(),
        Duration::from_secs(config.token_lifetime_secs),
        Duration::from_secs(REFRESH_OFFSET_SECS),
    ));
    credentials
        .get_credential()
        .await
        .map_err(|error| LanceError::io_source(Box::new(error)))?;
    Ok(Arc::new(ImpersonatingGcsStoreProvider::new(credentials)))
}

pub(crate) fn build_lance_session(provider: Arc<dyn ObjectStoreProvider>) -> Arc<Session> {
    let registry = ObjectStoreRegistry::default();
    registry.insert("gs", provider);
    Arc::new(Session::new(0, 0, Arc::new(registry)))
}

fn gcp_iceberg_credential_error(error: object_store::Error) -> IcebergError {
    IcebergError::new(
        IcebergErrorKind::Unexpected,
        format!("GCP impersonation credential resolution failed: {error}"),
    )
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct GcpImpersonationStorageFactory {
    #[serde(skip)]
    provider: Option<Arc<ImpersonatingGcsCredentialProvider>>,
}

impl GcpImpersonationStorageFactory {
    pub(crate) async fn new(config: GcpImpersonationConfig) -> IcebergResult<Self> {
        let provider = Arc::new(ImpersonatingGcsCredentialProvider::new(
            config.target_sa,
            Duration::from_secs(config.token_lifetime_secs),
            Duration::from_secs(REFRESH_OFFSET_SECS),
        ));
        provider
            .get_credential()
            .await
            .map_err(gcp_iceberg_credential_error)?;
        Ok(Self {
            provider: Some(provider),
        })
    }
}

impl fmt::Debug for GcpImpersonationStorageFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcpImpersonationStorageFactory")
            .field("has_runtime_provider", &self.provider.is_some())
            .finish()
    }
}

#[typetag::serde(name = "GcpImpersonationStorageFactory")]
impl StorageFactory for GcpImpersonationStorageFactory {
    fn build(&self, config: &StorageConfig) -> IcebergResult<Arc<dyn IcebergStorage>> {
        let provider = self.provider.clone().ok_or_else(|| {
            IcebergError::new(
                IcebergErrorKind::Unexpected,
                "GCP impersonation runtime provider is unavailable",
            )
        })?;
        Ok(Arc::new(GcpImpersonationStorage {
            props: Arc::new(config.props().clone()),
            provider: Some(provider),
            cached_storage: Arc::new(RwLock::new(None)),
        }))
    }
}

#[derive(Clone)]
struct CachedGcpImpersonationStorage {
    credential: Arc<GcpCredential>,
    storage: Arc<dyn IcebergStorage>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct GcpImpersonationStorage {
    #[serde(skip)]
    props: Arc<HashMap<String, String>>,
    #[serde(skip)]
    provider: Option<Arc<ImpersonatingGcsCredentialProvider>>,
    #[serde(skip)]
    cached_storage: Arc<RwLock<Option<CachedGcpImpersonationStorage>>>,
}

impl fmt::Debug for GcpImpersonationStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcpImpersonationStorage")
            .field("has_runtime_provider", &self.provider.is_some())
            .finish()
    }
}

impl GcpImpersonationStorage {
    async fn current_storage(&self) -> IcebergResult<Arc<dyn IcebergStorage>> {
        let provider = self.provider.as_ref().ok_or_else(|| {
            IcebergError::new(
                IcebergErrorKind::Unexpected,
                "GCP impersonation runtime provider is unavailable",
            )
        })?;
        let credential = provider
            .get_credential()
            .await
            .map_err(gcp_iceberg_credential_error)?;
        {
            let cached = self.cached_storage.read().await;
            if let Some(cached) = cached.as_ref()
                && Arc::ptr_eq(&cached.credential, &credential)
            {
                return Ok(cached.storage.clone());
            }
        }

        let mut cached = self.cached_storage.write().await;
        if let Some(cached) = cached.as_ref()
            && Arc::ptr_eq(&cached.credential, &credential)
        {
            return Ok(cached.storage.clone());
        }

        let mut props = self.props.as_ref().clone();
        props.insert(GCS_TOKEN.to_string(), credential.bearer.clone());
        props.insert(GCS_DISABLE_VM_METADATA.to_string(), "true".to_string());
        props.insert(GCS_DISABLE_CONFIG_LOAD.to_string(), "true".to_string());
        let storage = OpenDalStorageFactory::Gcs.build(&StorageConfig::from_props(props))?;
        *cached = Some(CachedGcpImpersonationStorage {
            credential,
            storage: storage.clone(),
        });
        Ok(storage)
    }
}

#[typetag::serde(name = "GcpImpersonationStorage")]
#[async_trait]
impl IcebergStorage for GcpImpersonationStorage {
    async fn exists(&self, path: &str) -> IcebergResult<bool> {
        self.current_storage().await?.exists(path).await
    }

    async fn metadata(&self, path: &str) -> IcebergResult<FileMetadata> {
        self.current_storage().await?.metadata(path).await
    }

    async fn read(&self, path: &str) -> IcebergResult<bytes::Bytes> {
        self.current_storage().await?.read(path).await
    }

    async fn reader(&self, path: &str) -> IcebergResult<Box<dyn FileRead>> {
        self.current_storage().await?.reader(path).await
    }

    async fn write(&self, path: &str, bytes: bytes::Bytes) -> IcebergResult<()> {
        self.current_storage().await?.write(path, bytes).await
    }

    async fn writer(&self, path: &str) -> IcebergResult<Box<dyn FileWrite>> {
        self.current_storage().await?.writer(path).await
    }

    async fn delete(&self, path: &str) -> IcebergResult<()> {
        self.current_storage().await?.delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> IcebergResult<()> {
        self.current_storage().await?.delete_prefix(path).await
    }

    fn new_input(&self, path: &str) -> IcebergResult<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> IcebergResult<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::future::join_all;
    use iceberg::io::{StorageConfig, StorageFactory};
    use tokio::sync::{Barrier, Notify};

    use super::*;

    fn impersonation_config(target_sa: &str) -> GcpImpersonationConfig {
        GcpImpersonationConfig {
            target_sa: target_sa.to_string(),
            token_lifetime_secs: 3600,
        }
    }

    fn credential_provider(target_sa: &str) -> ImpersonatingGcsCredentialProvider {
        ImpersonatingGcsCredentialProvider::new(
            target_sa.to_string(),
            Duration::from_secs(3600),
            Duration::from_secs(300),
        )
    }

    #[tokio::test]
    async fn get_credential_refresh_is_single_flight() {
        const CALLERS: usize = 100;

        let provider = Arc::new(credential_provider("target@example.com"));
        let fetch_started = Arc::new(Notify::new());
        let release_fetch = Arc::new(Notify::new());
        let fetch_calls = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(CALLERS + 1));
        let tasks = (0..CALLERS)
            .map(|_| {
                let provider = provider.clone();
                let fetch_started = fetch_started.clone();
                let release_fetch = release_fetch.clone();
                let fetch_calls = fetch_calls.clone();
                let start = start.clone();
                tokio::spawn(async move {
                    start.wait().await;
                    provider
                        .get_credential_with(|| async move {
                            fetch_calls.fetch_add(1, Ordering::SeqCst);
                            fetch_started.notify_one();
                            release_fetch.notified().await;
                            Ok(CachedToken {
                                credential: Arc::new(GcpCredential {
                                    bearer: "single-flight".to_string(),
                                }),
                                expires_at_ms: ImpersonatingGcsCredentialProvider::now_ms()
                                    + 3_600_000,
                            })
                        })
                        .await
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();

        start.wait().await;
        fetch_started.notified().await;
        assert_eq!(fetch_calls.load(Ordering::SeqCst), 1);
        release_fetch.notify_one();

        let credentials = join_all(tasks)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(fetch_calls.load(Ordering::SeqCst), 1);
        assert!(
            credentials
                .iter()
                .all(|credential| Arc::ptr_eq(credential, &credentials[0]))
        );
    }

    #[test]
    fn extracts_and_removes_private_options() {
        for target_key in [LANCE_TARGET_SERVICE_ACCOUNT, ICEBERG_TARGET_SERVICE_ACCOUNT] {
            let mut options = HashMap::from([
                (target_key.to_string(), "target@example.com".to_string()),
                (TOKEN_LIFETIME_SECONDS.to_string(), "3600".to_string()),
                ("public_option".to_string(), "value".to_string()),
            ]);

            let config = GcpImpersonationConfig::extract(&mut options, target_key)
                .unwrap()
                .unwrap();

            assert_eq!(config, impersonation_config("target@example.com"));
            assert!(!options.contains_key(target_key));
            assert!(!options.contains_key(TOKEN_LIFETIME_SECONDS));
            assert_eq!(options["public_option"], "value");
        }
    }

    #[test]
    fn rejects_missing_or_out_of_range_token_lifetime() {
        for lifetime in [None, Some("invalid"), Some("899"), Some("3601")] {
            let mut options = HashMap::from([(
                LANCE_TARGET_SERVICE_ACCOUNT.to_string(),
                "target@example.com".to_string(),
            )]);
            if let Some(lifetime) = lifetime {
                options.insert(TOKEN_LIFETIME_SECONDS.to_string(), lifetime.to_string());
            }

            assert!(
                GcpImpersonationConfig::extract(&mut options, LANCE_TARGET_SERVICE_ACCOUNT)
                    .is_err()
            );
        }

        for lifetime in ["900", "3600"] {
            let mut options = HashMap::from([
                (
                    LANCE_TARGET_SERVICE_ACCOUNT.to_string(),
                    "target@example.com".to_string(),
                ),
                (TOKEN_LIFETIME_SECONDS.to_string(), lifetime.to_string()),
            ]);
            assert!(
                GcpImpersonationConfig::extract(&mut options, LANCE_TARGET_SERVICE_ACCOUNT)
                    .unwrap()
                    .is_some()
            );
        }
    }

    #[test]
    fn iceberg_factory_build_does_not_resolve_credentials() {
        let factory = GcpImpersonationStorageFactory {
            provider: Some(Arc::new(credential_provider("target@example.com"))),
        };

        let _storage = factory.build(&StorageConfig::new()).unwrap();
    }

    #[tokio::test]
    async fn iceberg_storage_reuses_delegated_storage_until_token_changes() {
        let provider = Arc::new(credential_provider("target@example.com"));
        let first_credential = Arc::new(GcpCredential {
            bearer: "cached-token".to_string(),
        });
        *provider.cache.write().await = Some(CachedToken {
            credential: first_credential,
            expires_at_ms: ImpersonatingGcsCredentialProvider::now_ms() + 3_600_000,
        });
        let storage = GcpImpersonationStorage {
            props: Arc::new(HashMap::new()),
            provider: Some(provider.clone()),
            cached_storage: Arc::new(RwLock::new(None)),
        };

        let first = storage.current_storage().await.unwrap();
        let second = storage.current_storage().await.unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        *provider.cache.write().await = Some(CachedToken {
            credential: Arc::new(GcpCredential {
                bearer: "refreshed-token".to_string(),
            }),
            expires_at_ms: ImpersonatingGcsCredentialProvider::now_ms() + 3_600_000,
        });
        let refreshed = storage.current_storage().await.unwrap();
        assert!(!Arc::ptr_eq(&first, &refreshed));
    }

    #[test]
    fn iceberg_factory_and_storage_serialization_omit_runtime_credentials() {
        let provider = Arc::new(credential_provider("secret-target@example.com"));
        let factory = GcpImpersonationStorageFactory {
            provider: Some(provider.clone()),
        };
        let storage = GcpImpersonationStorage {
            props: Arc::new(HashMap::from([(
                iceberg::io::GCS_SERVICE_PATH.to_string(),
                "https://secret-endpoint".to_string(),
            )])),
            provider: Some(provider),
            cached_storage: Arc::new(RwLock::new(None)),
        };

        for serialized in [
            serde_json::to_string(&factory).unwrap(),
            serde_json::to_string(&storage).unwrap(),
        ] {
            assert!(!serialized.contains("secret-target"));
            assert!(!serialized.contains("secret-endpoint"));
        }
    }

    #[test]
    fn parse_rfc3339_basic() {
        let ms = parse_rfc3339_to_ms("2026-04-17T03:23:14Z").unwrap();
        assert!(ms > 1_767_225_600_000);
        assert!(ms < 1_798_761_600_000);
    }

    #[test]
    fn parse_rfc3339_pre_epoch_rejected() {
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
        let provider = credential_provider("x@y.iam.gserviceaccount.com");
        assert!(provider.needs_refresh(&None));
    }

    #[test]
    fn needs_refresh_within_offset() {
        let provider = credential_provider("x@y.iam.gserviceaccount.com");
        let cached = Some(CachedToken {
            credential: Arc::new(GcpCredential { bearer: "x".into() }),
            expires_at_ms: ImpersonatingGcsCredentialProvider::now_ms() + 100_000,
        });
        assert!(provider.needs_refresh(&cached));
    }

    #[test]
    fn no_refresh_when_fresh() {
        let provider = credential_provider("x@y.iam.gserviceaccount.com");
        let cached = Some(CachedToken {
            credential: Arc::new(GcpCredential { bearer: "x".into() }),
            expires_at_ms: ImpersonatingGcsCredentialProvider::now_ms() + 3_600_000,
        });
        assert!(!provider.needs_refresh(&cached));
    }
}
