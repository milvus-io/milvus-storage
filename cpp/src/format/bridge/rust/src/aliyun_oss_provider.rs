// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Aliyun OSS with per-tenant `role_arn` support for the Lance path.
//!
//! This file provides:
//!   * [`AliyunOssStoreProvider`] — lance-io's [`ObjectStoreProvider`], used
//!     by the Lance reader/writer path.
//!   * A common env-sweep + bucket/root helper plus Lance-shaped
//!     `storage_options` parsing (`oss_endpoint`, `oss_role_arn`, ...).
//!   * [`apply_ram_mode_if_requested`] — resolves `role_arn` to concrete STS
//!     creds when `ALIYUN_ROLE_ARN_AUTH_MODE=ram` is set (ECS IMDS →
//!     `sts:AssumeRole` flow). No-op in the default OIDC mode.
//!   * The inner [`ram`] module — IMDSv2 for RAM mode plus the POP v1
//!     `sts:AssumeRole` helper reused by both RAM and OIDC step 2.
//!
//! # Two AssumeRole flows
//!
//! In the default OIDC mode we resolve `role_arn` with a two-step chain in
//! this module: `AssumeRoleWithOIDC` mints machine-identity STS credentials,
//! then `sts:AssumeRole` mints customer-side STS credentials. The final
//! credentials are injected into opendal's static-credential slots and
//! `role_arn` is stripped so reqsign's single-step AssumeRoleWithOIDC path
//! cannot fire.
//!
//! In RAM mode (`ALIYUN_ROLE_ARN_AUTH_MODE=ram`) we resolve `role_arn` to
//! short-lived STS credentials *in this module* (IMDS then
//! `sts:AssumeRole`) and perform the same static-credential hand-off. There
//! is no auto-fallback — an unset or missing-OIDC-token environment would
//! otherwise silently route through the wrong path, which is exactly the kind
//! of thing an explicit env var is for.
//!
//! # Static AK/SK must not be forwarded on the OIDC role_arn path
//!
//! reqsign loads credentials in this order
//! (`reqsign::aliyun::credential.rs` — `load_via_static` before
//! `load_via_assume_role_with_oidc`): if static creds are set alongside
//! `role_arn`, the OIDC path is silently skipped. `lance_common.cpp` must
//! not emit AK/SK when `role_arn` is set. In RAM mode this is turned on its
//! head — we *do* want reqsign to use the static creds we just derived, and
//! we strip `role_arn` ourselves to keep that route unambiguous.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::{stream, StreamExt, TryStreamExt};
use object_store::{
    path::Path, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    Result as ObjectStoreResult,
};
use object_store_opendal::OpendalStore;
use opendal::{Operator, services::Oss};
use snafu::location;
use tokio::sync::RwLock;
use url::Url;

use lance::session::Session;
use lance::{Error as LanceError, Result as LanceResult};
use lance_io::object_store::{
    DEFAULT_CLOUD_IO_PARALLELISM, ObjectStore, ObjectStoreParams, ObjectStoreProvider,
    ObjectStoreRegistry, StorageOptions,
};
/// lance-io's `DEFAULT_CLOUD_BLOCK_SIZE` is crate-private; mirror the 64 KiB
/// value used by stock `OssStoreProvider`.
const OSS_DEFAULT_BLOCK_SIZE: usize = 64 * 1024;

/// Mirrors stock `OssStoreProvider::download_retry_count()` fallback.
const OSS_DEFAULT_DOWNLOAD_RETRIES: usize = 3;

const ALIYUN_OSS_STORE_NAME: &str = "aliyun_oss";
const OSS_CREDENTIAL_REFRESH_SECS_KEY: &str = "oss_credential_refresh_secs";
const DEFAULT_OSS_CREDENTIAL_REFRESH_SECS: u64 = 50 * 60;
const REFRESH_LOCK_RETRY_MS: u64 = 10;

// ============================================================================
// Shared: opendal OSS config assembly + RAM-mode credential swap
// ============================================================================

/// Build the env-and-URL part of the opendal OSS config map.
fn init_oss_env_config(bucket: String, base_path: &Url) -> HashMap<String, String> {
    // Env sweep: env vars starting with `OSS_`, `AWS_`, or `ALIBABA_CLOUD_`
    // are lowercased, and those three substrings are removed from the key
    // via `.replace()` (all occurrences, not just the prefix — for any real
    // env var this is equivalent to stripping the prefix). The stripped,
    // lowercased key is what opendal's `Oss` service expects. Machine
    // identity env vars `ALIBABA_CLOUD_OIDC_TOKEN_FILE` and
    // `ALIBABA_CLOUD_OIDC_PROVIDER_ARN` flow into opendal as
    // `oidc_token_file` / `oidc_provider_arn` via this path. Shape mirrors
    // stock lance-io `OssStoreProvider`.
    let mut config_map: HashMap<String, String> = std::env::vars()
        .filter(|(k, _)| {
            k.starts_with("OSS_") || k.starts_with("AWS_") || k.starts_with("ALIBABA_CLOUD_")
        })
        .map(|(k, v)| {
            let key = k
                .to_lowercase()
                .replace("oss_", "")
                .replace("aws_", "")
                .replace("alibaba_cloud_", "");
            (key, v)
        })
        .collect();

    config_map.insert("bucket".to_string(), bucket);

    // Mirrors stock lance-io `OssStoreProvider`: when the URL has a
    // non-empty path, write `root="/"`. This is a no-op at the opendal
    // layer (opendal's default `root` is already `/`) — the actual URL
    // prefix is applied by lance at the `ObjectStore` level, not through
    // the operator. The `prefix` local is used only as the empty-path
    // guard; it is deliberately not forwarded. Kept in this shape so the
    // block stays recognisable against stock for future diffs.
    let prefix = base_path.path().trim_start_matches('/').to_string();
    if !prefix.is_empty() {
        config_map.insert("root".to_string(), "/".to_string());
    }

    config_map
}

/// Returns `Err` when no endpoint ended up in the config map.
fn require_endpoint(config_map: &HashMap<String, String>) -> Result<(), String> {
    if !config_map.contains_key("endpoint") {
        return Err(
            "OSS endpoint is required. Provide an OSS endpoint in storage options or set OSS_ENDPOINT environment variable".to_string(),
        );
    }
    Ok(())
}

/// Lance-style (underscored) storage_options → opendal OSS config. Keys
/// consumed: `oss_endpoint`, `oss_access_key_id`, `oss_secret_access_key`,
/// `oss_region`, `oss_role_arn`, `oss_role_session_name`, `oss_external_id`.
pub(crate) fn build_oss_config_from_lance_opts(
    bucket: String,
    base_path: &Url,
    storage_options: &HashMap<String, String>,
) -> Result<HashMap<String, String>, String> {
    let mut config_map = init_oss_env_config(bucket, base_path);

    // storage_options overrides (later wins over env). Stock four keys:
    if let Some(endpoint) = storage_options.get("oss_endpoint") {
        config_map.insert("endpoint".to_string(), endpoint.clone());
    }
    if let Some(access_key_id) = storage_options.get("oss_access_key_id") {
        config_map.insert("access_key_id".to_string(), access_key_id.clone());
    }
    if let Some(secret_access_key) = storage_options.get("oss_secret_access_key") {
        config_map.insert("access_key_secret".to_string(), secret_access_key.clone());
    }
    if let Some(region) = storage_options.get("oss_region") {
        config_map.insert("region".to_string(), region.clone());
    }

    // The three keys stock lance-io `OssStoreProvider` does NOT forward.
    // This is the reason this provider exists — per-tenant role_arn /
    // session_name / external_id can only reach opendal via `storage_options`.
    // `external_id` is consumed by the OIDC chain / RAM mode helpers below
    // (step-2 sts:AssumeRole), not by opendal/reqsign directly.
    if let Some(role_arn) = storage_options.get("oss_role_arn") {
        config_map.insert("role_arn".to_string(), role_arn.clone());
    }
    if let Some(role_session_name) = storage_options.get("oss_role_session_name") {
        config_map.insert("role_session_name".to_string(), role_session_name.clone());
    }
    if let Some(external_id) = storage_options.get("oss_external_id") {
        config_map.insert("external_id".to_string(), external_id.clone());
    }

    require_endpoint(&config_map)?;
    Ok(config_map)
}
/// OIDC-mode credential chain: when no `ALIYUN_ROLE_ARN_AUTH_MODE=ram` is
/// set and `role_arn` is present in `config_map`, do the two-step chain
///
///   1. `AssumeRoleWithOIDC` for the *machine identity* role from env
///      (`ALIBABA_CLOUD_ROLE_ARN` + `ALIBABA_CLOUD_OIDC_PROVIDER_ARN` +
///      `ALIBABA_CLOUD_OIDC_TOKEN_FILE`) — same account, the only shape
///      Aliyun STS accepts for AssumeRoleWithOIDC.
///   2. `sts:AssumeRole` into the caller-supplied target role using the
///      step-1 STS creds. This is the only step that crosses accounts;
///      the customer's role trust policy must list the step-1 role as
///      Principal.
///
/// Single-step (handing the customer role straight to AssumeRoleWithOIDC,
/// which is what stock reqsign would do if we left `role_arn` in the
/// config map) fails whenever `RoleArn` and `OIDCProviderArn` live in
/// different accounts — Aliyun rejects with `AssumeRolePolicy ImplicitDeny`.
/// This function exists exclusively to fix that cross-tenant case; the
/// shape mirrors the C++ `AliyunOIDCAssumeRoleChainProvider`.
///
/// On success the chain hand-off matches `ram::apply_credentials`: static
/// creds + `security_token` injected into `config_map`, `role_arn` stripped
/// so reqsign's own AssumeRoleWithOIDC path cannot re-fire.
pub(crate) async fn apply_oidc_chain_if_requested(
    config_map: &mut HashMap<String, String>,
) -> Result<(), String> {
    apply_oidc_chain_if_requested_with_expiration(config_map)
        .await
        .map(|_| ())
}

async fn apply_oidc_chain_if_requested_with_expiration(
    config_map: &mut HashMap<String, String>,
) -> Result<Option<u64>, String> {
    // RAM mode owns this config_map; do not double-resolve.
    if std::env::var(ram::AUTH_MODE_ENV).as_deref() == Ok(ram::AUTH_MODE_RAM) {
        return Ok(None);
    }
    let Some(target_role_arn) = config_map.get("role_arn").cloned() else {
        return Ok(None);
    };

    // Three machine-identity env vars are required for the inner step.
    // Failing here keeps the misconfig at the operator-construction site
    // rather than letting it surface as a silent anonymous OSS request
    // a few stack frames down.
    let inner_role_arn = std::env::var("ALIBABA_CLOUD_ROLE_ARN").map_err(|_| {
        "Aliyun role_arn requires ALIBABA_CLOUD_ROLE_ARN, ALIBABA_CLOUD_OIDC_TOKEN_FILE \
         and ALIBABA_CLOUD_OIDC_PROVIDER_ARN in process environment \
         (or set ALIYUN_ROLE_ARN_AUTH_MODE=ram for ECS IMDS-based AssumeRole)"
            .to_string()
    })?;
    let token_file = std::env::var("ALIBABA_CLOUD_OIDC_TOKEN_FILE").map_err(|_| {
        "Aliyun role_arn requires ALIBABA_CLOUD_OIDC_TOKEN_FILE in process environment".to_string()
    })?;
    let provider_arn = std::env::var("ALIBABA_CLOUD_OIDC_PROVIDER_ARN").map_err(|_| {
        "Aliyun role_arn requires ALIBABA_CLOUD_OIDC_PROVIDER_ARN in process environment"
            .to_string()
    })?;
    let token = std::fs::read_to_string(&token_file)
        .map_err(|e| format!("read OIDC token file {token_file}: {e}"))?;

    let session_name = config_map
        .get("role_session_name")
        .cloned()
        .unwrap_or_else(ram::default_session_name);
    // ExternalId belongs to step 2 only — Aliyun's AssumeRoleWithOIDC API
    // has no ExternalId concept. Empty == not sent.
    let external_id = config_map.get("external_id").cloned().unwrap_or_default();

    let client = ram::build_http_client()?;
    let inner =
        call_assume_role_with_oidc(&client, &inner_role_arn, &provider_arn, token.trim()).await?;
    let outer =
        ram::call_assume_role(&client, &inner, &target_role_arn, &session_name, &external_id)
            .await?;
    let expires_at_ms = outer.expires_at_ms;

    ram::apply_credentials(config_map, outer);
    Ok(expires_at_ms)
}

/// Step 1 of the OIDC chain: `AssumeRoleWithOIDC` against same-account
/// `role_arn` + `provider_arn` using a freshly-read OIDC `token`. Parses
/// the JSON response (Format=JSON) into the same `AssumeRoleCreds` shape
/// the RAM-mode helpers use, so step 2 can reuse `ram::call_assume_role`
/// unchanged.
async fn call_assume_role_with_oidc(
    client: &reqwest::Client,
    role_arn: &str,
    provider_arn: &str,
    token: &str,
) -> Result<ram::AssumeRoleCreds, String> {
    use chrono::Utc;
    let timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let nonce = uuid::Uuid::now_v7().to_string();
    // AssumeRoleWithOIDC is unsigned: the OIDC token in the form body is
    // the auth material. POP v1 signing (which the RAM-mode step 2 needs)
    // would be wrong here.
    let form = [
        ("Action", "AssumeRoleWithOIDC"),
        ("Version", "2015-04-01"),
        ("Format", "JSON"),
        ("Timestamp", &timestamp),
        ("SignatureNonce", &nonce),
        ("RoleArn", role_arn),
        ("OIDCProviderArn", provider_arn),
        ("OIDCToken", token),
        ("RoleSessionName", "milvus-storage-oidc-chain"),
    ];
    let resp = client
        .post("https://sts.aliyuncs.com/")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&form)
        .send()
        .await
        .map_err(|e| format!("AssumeRoleWithOIDC POST: {e}"))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| format!("AssumeRoleWithOIDC body read: {e}"))?;
    if !status.is_success() {
        return Err(format!("AssumeRoleWithOIDC HTTP {status}: {text}"));
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "PascalCase")]
    struct CredsJson {
        access_key_id: String,
        access_key_secret: String,
        security_token: String,
    }
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "PascalCase")]
    struct OidcResponseJson {
        credentials: CredsJson,
    }
    let parsed: OidcResponseJson = serde_json::from_str(&text)
        .map_err(|e| format!("AssumeRoleWithOIDC JSON parse failed ({e}); body was: {text}"))?;
    Ok(ram::AssumeRoleCreds {
        access_key_id: parsed.credentials.access_key_id,
        access_key_secret: parsed.credentials.access_key_secret,
        security_token: parsed.credentials.security_token,
        expires_at_ms: None,
    })
}

/// RAM-mode credential swap: when `ALIYUN_ROLE_ARN_AUTH_MODE=ram` is set in
/// env and `role_arn` is present in `config_map`, resolve it to concrete STS
/// creds via the ECS IMDS → `sts:AssumeRole` flow and mutate `config_map`
/// so opendal sees static creds instead. No-op in every other case — OIDC
/// callers, plain AK/SK callers, and any caller without the env var flipped
/// fall straight through. Lance `role_arn` stores wrap this one-shot swap in
/// [`RefreshableAliyunOssStore`] so long-lived scans rebuild the opendal store
/// on the configured refresh interval. Iceberg intentionally calls this from
/// `create_operator()` for each I/O operation instead of keeping a cache.
pub(crate) async fn apply_ram_mode_if_requested(
    config_map: &mut HashMap<String, String>,
) -> Result<(), String> {
    apply_ram_mode_if_requested_with_expiration(config_map)
        .await
        .map(|_| ())
}

async fn apply_ram_mode_if_requested_with_expiration(
    config_map: &mut HashMap<String, String>,
) -> Result<Option<u64>, String> {
    if std::env::var(ram::AUTH_MODE_ENV).as_deref() != Ok(ram::AUTH_MODE_RAM) {
        return Ok(None);
    }
    let Some(role_arn) = config_map.get("role_arn").cloned() else {
        return Ok(None);
    };
    let session_name = config_map
        .get("role_session_name")
        .cloned()
        .unwrap_or_else(ram::default_session_name);
    let external_id = config_map.get("external_id").cloned().unwrap_or_default();
    let creds = ram::fetch_assume_role_creds(&role_arn, &session_name, &external_id).await?;
    let expires_at_ms = creds.expires_at_ms;
    ram::apply_credentials(config_map, creds);
    Ok(expires_at_ms)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis() as u64
}

fn parse_aliyun_expiration_to_ms(s: &str) -> Result<u64, String> {
    let dt = chrono::DateTime::parse_from_rfc3339(s).map_err(|e| e.to_string())?;
    u64::try_from(dt.timestamp_millis()).map_err(|_| format!("pre-epoch Expiration: {s}"))
}

fn parse_oss_credential_refresh_interval(
    storage_options: &HashMap<String, String>,
) -> Result<Duration, String> {
    let Some(raw) = storage_options.get(OSS_CREDENTIAL_REFRESH_SECS_KEY) else {
        return Ok(Duration::from_secs(DEFAULT_OSS_CREDENTIAL_REFRESH_SECS));
    };
    let secs = raw.parse::<u64>().map_err(|e| {
        format!("{OSS_CREDENTIAL_REFRESH_SECS_KEY} must be a positive integer seconds value: {e}")
    })?;
    if secs == 0 {
        return Err(format!(
            "{OSS_CREDENTIAL_REFRESH_SECS_KEY} must be greater than 0"
        ));
    }
    Ok(Duration::from_secs(secs))
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn should_refresh(last_refresh_at_ms: u64, now_ms: u64, refresh_interval: Duration) -> bool {
    now_ms >= last_refresh_at_ms.saturating_add(duration_millis(refresh_interval))
}

fn validate_refresh_interval_before_expiration(
    refresh_interval: Duration,
    issued_at_ms: u64,
    expires_at_ms: u64,
) -> Result<(), String> {
    if expires_at_ms <= issued_at_ms {
        return Err("Aliyun STS credentials are already expired".to_string());
    }
    let lifetime_ms = expires_at_ms - issued_at_ms;
    let refresh_ms = duration_millis(refresh_interval);
    if refresh_ms >= lifetime_ms {
        return Err(format!(
            "Aliyun OSS credential refresh interval {}s must be less than server credential lifetime {}s",
            refresh_interval.as_secs(),
            lifetime_ms / 1000
        ));
    }
    Ok(())
}

fn aliyun_object_store_error(message: impl Into<String>) -> object_store::Error {
    let message: String = message.into();
    object_store::Error::Generic {
        store: ALIYUN_OSS_STORE_NAME,
        source: message.into(),
    }
}

#[derive(Clone)]
struct CachedAliyunOssStore {
    store: Arc<dyn OSObjectStore>,
    refreshed_at_ms: u64,
}

#[derive(Clone)]
struct RefreshableAliyunOssStore {
    base_config: HashMap<String, String>,
    refresh_interval: Duration,
    cache: Arc<RwLock<Option<CachedAliyunOssStore>>>,
}

impl RefreshableAliyunOssStore {
    fn new(base_config: HashMap<String, String>, refresh_interval: Duration) -> Self {
        Self {
            base_config,
            refresh_interval,
            cache: Arc::new(RwLock::new(None)),
        }
    }

    fn cached_store_is_fresh(&self, cached: &Option<CachedAliyunOssStore>) -> bool {
        match cached {
            Some(c) => !should_refresh(c.refreshed_at_ms, now_ms(), self.refresh_interval),
            None => false,
        }
    }

    async fn current_store(&self) -> ObjectStoreResult<Arc<dyn OSObjectStore>> {
        loop {
            if let Some(store) = self.try_current_store().await? {
                return Ok(store);
            }
            tokio::time::sleep(Duration::from_millis(REFRESH_LOCK_RETRY_MS)).await;
        }
    }

    async fn try_current_store(&self) -> ObjectStoreResult<Option<Arc<dyn OSObjectStore>>> {
        {
            let cached = self.cache.read().await;
            if self.cached_store_is_fresh(&cached) {
                if let Some(c) = &*cached {
                    return Ok(Some(c.store.clone()));
                }
            }
        }

        let Ok(mut cached) = self.cache.try_write() else {
            return Ok(None);
        };

        if self.cached_store_is_fresh(&cached) {
            if let Some(c) = &*cached {
                return Ok(Some(c.store.clone()));
            }
        }

        let refreshed = self.refresh_store().await?;
        let store = refreshed.store.clone();
        *cached = Some(refreshed);
        Ok(Some(store))
    }

    async fn refresh_store(&self) -> ObjectStoreResult<CachedAliyunOssStore> {
        let mut config_map = self.base_config.clone();

        let ram_expires_at_ms = apply_ram_mode_if_requested_with_expiration(&mut config_map)
            .await
            .map_err(|e| {
                aliyun_object_store_error(format!(
                    "Aliyun RAM-mode credential refresh failed: {e}"
                ))
            })?;
        let oidc_expires_at_ms = apply_oidc_chain_if_requested_with_expiration(&mut config_map)
            .await
            .map_err(|e| {
                aliyun_object_store_error(format!(
                    "Aliyun OIDC chain credential refresh failed: {e}"
                ))
            })?;
        let expires_at_ms = ram_expires_at_ms.or(oidc_expires_at_ms).ok_or_else(|| {
            aliyun_object_store_error("Aliyun role_arn credential refresh did not return an Expiration")
        })?;
        let refreshed_at_ms = now_ms();

        validate_refresh_interval_before_expiration(
            self.refresh_interval,
            refreshed_at_ms,
            expires_at_ms,
        )
        .map_err(aliyun_object_store_error)?;

        let operator = Operator::from_iter::<Oss>(config_map)
            .map_err(|e| {
                aliyun_object_store_error(format!("Failed to create OSS operator: {e:?}"))
            })?
            .finish();
        Ok(CachedAliyunOssStore {
            store: Arc::new(OpendalStore::new(operator)) as Arc<dyn OSObjectStore>,
            refreshed_at_ms,
        })
    }
}

impl fmt::Debug for RefreshableAliyunOssStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefreshableAliyunOssStore")
            .field("refresh_interval", &self.refresh_interval)
            .field("has_role_arn", &self.base_config.contains_key("role_arn"))
            .finish_non_exhaustive()
    }
}

impl fmt::Display for RefreshableAliyunOssStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RefreshableAliyunOssStore(refresh_interval={}s)",
            self.refresh_interval.as_secs()
        )
    }
}

#[async_trait::async_trait]
impl OSObjectStore for RefreshableAliyunOssStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> ObjectStoreResult<PutResult> {
        self.current_store()
            .await?
            .put_opts(location, payload, opts)
            .await
    }

    async fn put_multipart(
        &self,
        location: &Path,
    ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
        self.current_store().await?.put_multipart(location).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
        self.current_store()
            .await?
            .put_multipart_opts(location, opts)
            .await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> ObjectStoreResult<GetResult> {
        self.current_store().await?.get_opts(location, options).await
    }

    async fn delete(&self, location: &Path) -> ObjectStoreResult<()> {
        self.current_store().await?.delete(location).await
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> futures::stream::BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
        let this = self.clone();
        let prefix = prefix.cloned();
        stream::once(async move {
            let store = this.current_store().await?;
            Ok::<_, object_store::Error>(store.list(prefix.as_ref()))
        })
        .try_flatten()
        .boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> futures::stream::BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
        let this = self.clone();
        let prefix = prefix.cloned();
        let offset = offset.clone();
        stream::once(async move {
            let store = this.current_store().await?;
            Ok::<_, object_store::Error>(store.list_with_offset(prefix.as_ref(), &offset))
        })
        .try_flatten()
        .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> ObjectStoreResult<ListResult> {
        self.current_store()
            .await?
            .list_with_delimiter(prefix)
            .await
    }

    async fn copy(&self, from: &Path, to: &Path) -> ObjectStoreResult<()> {
        self.current_store().await?.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> ObjectStoreResult<()> {
        self.current_store()
            .await?
            .copy_if_not_exists(from, to)
            .await
    }
}

// ============================================================================
// Lance: ObjectStoreProvider
// ============================================================================

#[derive(Default, Debug)]
pub struct AliyunOssStoreProvider;

#[async_trait::async_trait]
impl ObjectStoreProvider for AliyunOssStoreProvider {
    async fn new_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> LanceResult<ObjectStore> {
        let block_size = params.block_size.unwrap_or(OSS_DEFAULT_BLOCK_SIZE);
        let storage_options =
            StorageOptions(params.storage_options.clone().unwrap_or_default());

        let bucket = base_path
            .host_str()
            .ok_or_else(|| {
                LanceError::invalid_input("OSS URL must contain bucket name", location!())
            })?
            .to_string();

        let config_map = build_oss_config_from_lance_opts(bucket, &base_path, &storage_options.0)
            .map_err(|e| LanceError::invalid_input(e, location!()))?;

        let inner = if config_map.contains_key("role_arn") {
            let refresh_interval = parse_oss_credential_refresh_interval(&storage_options.0)
                .map_err(|e| LanceError::invalid_input(e, location!()))?;
            let refreshable = Arc::new(RefreshableAliyunOssStore::new(
                config_map,
                refresh_interval,
            ));
            refreshable.current_store().await.map_err(|e| LanceError::IO {
                source: Box::new(e),
                location: location!(),
            })?;
            refreshable as Arc<dyn OSObjectStore>
        } else {
            let operator = Operator::from_iter::<Oss>(config_map)
                .map_err(|e| {
                    LanceError::invalid_input(
                        format!("Failed to create OSS operator: {:?}", e),
                        location!(),
                    )
                })?
                .finish();
            Arc::new(OpendalStore::new(operator)) as Arc<dyn OSObjectStore>
        };

        let mut url = base_path;
        if !url.path().ends_with('/') {
            url.set_path(&format!("{}/", url.path()));
        }

        Ok(ObjectStore::new(
            inner,
            url,
            Some(block_size),
            params.object_store_wrapper.clone(),
            params.use_constant_size_upload_parts,
            // OSS object listings are lexically ordered (matches stock provider).
            params.list_is_lexically_ordered.unwrap_or(true),
            DEFAULT_CLOUD_IO_PARALLELISM,
            OSS_DEFAULT_DOWNLOAD_RETRIES,
            params.storage_options.as_ref(),
        ))
    }
}

/// Build a `Session` whose `ObjectStoreRegistry` overrides the `oss` scheme
/// with [`AliyunOssStoreProvider`]. Used when `storage_options` carries a
/// per-tenant `oss_role_arn`: stock lance-io's `OssStoreProvider` silently
/// drops that key, so a per-tenant role can only reach opendal through this
/// provider.
///
/// A fresh `Session` per call matches the GCP impersonation pattern and
/// prevents concurrent opens with different roles from colliding on a
/// shared registry. Cache sizes of zero match what the FFI entry points
/// already pass to `BlockingDataset::open`.
pub fn build_aliyun_oss_session() -> Arc<Session> {
    let registry = ObjectStoreRegistry::default();
    registry.insert("oss", Arc::new(AliyunOssStoreProvider::default()));
    Arc::new(Session::new(0, 0, Arc::new(registry)))
}

// ============================================================================
// RAM-mode helpers (ECS IMDS → sts:AssumeRole + POP v1 signing)
// ============================================================================

/// ECS IMDS helpers plus the POP v1 `sts:AssumeRole` call shared by RAM mode
/// and OIDC step 2.
pub(crate) mod ram {
    use std::collections::HashMap;
    use std::time::Duration;

    use base64::Engine;
    use chrono::Utc;
    use hmac::{Hmac, Mac};
    use serde::Deserialize;
    use sha1::Sha1;

    /// Explicit opt-in env var for the ECS-IMDS → AssumeRole flow. `"ram"` selects
    /// it; anything else (including unset) keeps the default OIDC behaviour.
    pub(crate) const AUTH_MODE_ENV: &str = "ALIYUN_ROLE_ARN_AUTH_MODE";
    pub(crate) const AUTH_MODE_RAM: &str = "ram";

    /// ECS metadata service — Aliyun's fixed link-local HTTP endpoint.
    const IMDS_BASE: &str = "http://100.100.100.200";
    const IMDS_ROLE_LIST_PATH: &str = "/latest/meta-data/ram/security-credentials/";
    const IMDS_V2_TOKEN_PATH: &str = "/latest/api/token";
    /// Max TTL IMDS accepts for a V2 session token.
    const IMDS_V2_TTL_SECS: u64 = 21600;

    const STS_ENDPOINT: &str = "https://sts.aliyuncs.com/";

    /// Short-lived credentials returned from STS.
    #[derive(Debug, Clone)]
    pub(crate) struct AssumeRoleCreds {
        pub(crate) access_key_id: String,
        pub(crate) access_key_secret: String,
        pub(crate) security_token: String,
        /// Epoch milliseconds when STS says these credentials expire. IMDS
        /// source creds do not need this; final `sts:AssumeRole` creds do.
        pub(crate) expires_at_ms: Option<u64>,
    }

    /// Mutate `config_map` for the RAM-mode hand-off: insert the concrete STS
    /// creds that opendal's static-credential path expects, and remove the
    /// `role_arn` / `role_session_name` / `external_id` keys so reqsign cannot
    /// re-enter its own AssumeRoleWithOIDC flow (see module-level comment:
    /// static creds alongside `role_arn` make reqsign silently pick the static
    /// path — here that's what we want, *and* keeping `role_arn` around would
    /// be a correctness landmine if reqsign's preference ever inverted).
    /// `external_id` is stripped for hygiene: opendal does not consume it,
    /// but leaving it in the config map would let stale state leak across
    /// operator constructions if a future caller reused the same map.
    pub(crate) fn apply_credentials(
        config_map: &mut HashMap<String, String>,
        creds: AssumeRoleCreds,
    ) {
        config_map.remove("role_arn");
        config_map.remove("role_session_name");
        config_map.remove("external_id");
        config_map.insert("access_key_id".to_string(), creds.access_key_id);
        config_map.insert("access_key_secret".to_string(), creds.access_key_secret);
        config_map.insert("security_token".to_string(), creds.security_token);
    }

    pub(crate) fn default_session_name() -> String {
        format!("milvus-storage-ram-{}", uuid::Uuid::now_v7())
    }

    pub(crate) async fn fetch_assume_role_creds(
        role_arn: &str,
        role_session_name: &str,
        external_id: &str,
    ) -> Result<AssumeRoleCreds, String> {
        let client = build_http_client()?;
        let caller = fetch_imds_credentials(&client).await?;
        call_assume_role(&client, &caller, role_arn, role_session_name, external_id).await
    }

    /// Build a reqwest client with short timeouts tuned for IMDS + STS. IMDS
    /// answers in milliseconds on a healthy ECS; STS is a single round-trip on
    /// the public internet. A stalled request here would block the caller's
    /// whole `new_store`, so the tight caps are intentional.
    pub(super) fn build_http_client() -> Result<reqwest::Client, String> {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(15))
            .build()
            .map_err(|e| format!("build reqwest client: {e}"))
    }

    /// IMDSv2: PUT returns a session token in the response body. Anything other
    /// than 200 (404 on V1-only instances, 403 on some hardened variants) falls
    /// back to V1 with an empty token. Callers treat that as "no V2, use plain
    /// GETs".
    async fn fetch_imds_v2_token(client: &reqwest::Client) -> String {
        let resp = client
            .put(format!("{IMDS_BASE}{IMDS_V2_TOKEN_PATH}"))
            .header(
                "X-aliyun-ecs-metadata-token-ttl-seconds",
                IMDS_V2_TTL_SECS.to_string(),
            )
            .body("")
            .send()
            .await;
        match resp {
            Ok(r) if r.status().is_success() => r.text().await.unwrap_or_default().trim().to_string(),
            _ => String::new(),
        }
    }

    async fn fetch_imds_credentials(client: &reqwest::Client) -> Result<AssumeRoleCreds, String> {
        let v2 = fetch_imds_v2_token(client).await;

        // Step 1: role name listing. Aliyun returns a single line ("ecs-xxx");
        // multiple attached roles aren't a supported ECS concept.
        let list_url = format!("{IMDS_BASE}{IMDS_ROLE_LIST_PATH}");
        let mut req = client.get(&list_url);
        if !v2.is_empty() {
            req = req.header("X-aliyun-ecs-metadata-token", &v2);
        }
        let role_name = req
            .send()
            .await
            .map_err(|e| format!("IMDS role-list request: {e}"))?
            .error_for_status()
            .map_err(|e| {
                format!(
                    "IMDS role-list HTTP error (no RAM role attached to this ECS?): {e}"
                )
            })?
            .text()
            .await
            .map_err(|e| format!("IMDS role-list body: {e}"))?
            .trim()
            .to_string();
        if role_name.is_empty() {
            return Err("IMDS returned empty role name".into());
        }

        // Step 2: STS creds for that role.
        let creds_url = format!("{IMDS_BASE}{IMDS_ROLE_LIST_PATH}{role_name}");
        let mut req = client.get(&creds_url);
        if !v2.is_empty() {
            req = req.header("X-aliyun-ecs-metadata-token", &v2);
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "PascalCase")]
        struct ImdsCredsJson {
            access_key_id: String,
            access_key_secret: String,
            security_token: String,
        }
        let parsed: ImdsCredsJson = req
            .send()
            .await
            .map_err(|e| format!("IMDS creds request: {e}"))?
            .error_for_status()
            .map_err(|e| format!("IMDS creds HTTP error: {e}"))?
            .json()
            .await
            .map_err(|e| format!("IMDS creds JSON parse: {e}"))?;
        Ok(AssumeRoleCreds {
            access_key_id: parsed.access_key_id,
            access_key_secret: parsed.access_key_secret,
            security_token: parsed.security_token,
            expires_at_ms: None,
        })
    }

    /// POP v1 percent-encoding: RFC 3986 unreserved set, space as "%20".
    fn pop_encode(value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        for b in value.bytes() {
            if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
                out.push(b as char);
            } else {
                out.push_str(&format!("%{:02X}", b));
            }
        }
        out
    }

    /// POP v1 signature over an already-built canonical query. The signing
    /// algorithm (percent-encoding, StringToSign layout, HMAC-SHA1 + base64)
    /// mirrors `AliyunRAMSTSClient.cpp` and must stay in sync with it. The
    /// canonical query content itself is independent between the two sides
    /// — e.g. this Rust path uses `Format=JSON`, the C++ mirror uses XML —
    /// each side signs only its own request.
    fn pop_sign(access_key_secret: &str, canonical_query: &str) -> String {
        let string_to_sign = format!("POST&{}&{}", pop_encode("/"), pop_encode(canonical_query));
        let signing_key = format!("{access_key_secret}&");
        let mut mac = Hmac::<Sha1>::new_from_slice(signing_key.as_bytes())
            .expect("HMAC accepts any key size");
        mac.update(string_to_sign.as_bytes());
        base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
    }

    pub(super) async fn call_assume_role(
        client: &reqwest::Client,
        caller: &AssumeRoleCreds,
        role_arn: &str,
        role_session_name: &str,
        external_id: &str,
    ) -> Result<AssumeRoleCreds, String> {
        // BTreeMap for deterministic ASCII ordering of keys (POP v1 requirement).
        let mut params = std::collections::BTreeMap::new();
        params.insert("AccessKeyId", caller.access_key_id.as_str());
        params.insert("Action", "AssumeRole");
        // ExternalId is optional. We only insert it when non-empty so the
        // canonical query (and therefore the signature) stays identical to
        // the no-ExternalId case when the caller hasn't asked for one —
        // sending an empty `ExternalId=` would be a different request and
        // would fail the target role's trust-policy check if that policy
        // requires the parameter to be absent.
        if !external_id.is_empty() {
            params.insert("ExternalId", external_id);
        }
        params.insert("Format", "JSON");
        params.insert("RoleArn", role_arn);
        params.insert("RoleSessionName", role_session_name);
        params.insert("SecurityToken", caller.security_token.as_str());
        params.insert("SignatureMethod", "HMAC-SHA1");
        let nonce = uuid::Uuid::now_v7().to_string();
        params.insert("SignatureNonce", &nonce);
        params.insert("SignatureVersion", "1.0");
        let timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        params.insert("Timestamp", &timestamp);
        params.insert("Version", "2015-04-01");

        let canonical_query = params
            .iter()
            .map(|(k, v)| format!("{}={}", pop_encode(k), pop_encode(v)))
            .collect::<Vec<_>>()
            .join("&");
        let signature = pop_sign(&caller.access_key_secret, &canonical_query);
        let body = format!(
            "{canonical_query}&Signature={}",
            pop_encode(&signature)
        );

        // HTTP 4xx responses from STS carry the error message as raw body
        // text; we surface it verbatim in the `Err` rather than parsing into
        // a dedicated error struct, since nothing downstream branches on the
        // STS error code. Success responses get parsed into typed structs
        // below.
        let resp = client
            .post(STS_ENDPOINT)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .map_err(|e| format!("sts:AssumeRole POST: {e}"))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| format!("sts:AssumeRole body read: {e}"))?;
        if !status.is_success() {
            return Err(format!("sts:AssumeRole HTTP {status}: {text}"));
        }

        #[derive(Deserialize)]
        #[serde(rename_all = "PascalCase")]
        struct CredentialsJson {
            access_key_id: String,
            access_key_secret: String,
            security_token: String,
            expiration: String,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "PascalCase")]
        struct AssumeRoleResponseJson {
            credentials: CredentialsJson,
        }
        let parsed: AssumeRoleResponseJson = serde_json::from_str(&text).map_err(|e| {
            format!("sts:AssumeRole JSON parse failed ({e}); body was: {text}")
        })?;
        let expires_at_ms = super::parse_aliyun_expiration_to_ms(&parsed.credentials.expiration)
            .map_err(|e| format!("sts:AssumeRole Expiration parse failed: {e}"))?;
        Ok(AssumeRoleCreds {
            access_key_id: parsed.credentials.access_key_id,
            access_key_secret: parsed.credentials.access_key_secret,
            security_token: parsed.credentials.security_token,
            expires_at_ms: Some(expires_at_ms),
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn apply_credentials_strips_role_arn_and_injects_static_creds() {
            // Simulates the state right before the opendal operator is built in
            // RAM mode: `role_arn` + `role_session_name` + `external_id` must
            // disappear, concrete AKID / SK / token must be present. If any of
            // these assertions break we risk reqsign re-entering
            // AssumeRoleWithOIDC with a role_arn it can't authenticate against,
            // or a stale external_id leaking into a future operator.
            let mut cfg: HashMap<String, String> = HashMap::new();
            cfg.insert("bucket".to_string(), "b".to_string());
            cfg.insert("endpoint".to_string(), "e".to_string());
            cfg.insert("role_arn".to_string(), "acs:ram::1:role/x".to_string());
            cfg.insert("role_session_name".to_string(), "sess".to_string());
            cfg.insert("external_id".to_string(), "tenant-A-ext".to_string());
            apply_credentials(
                &mut cfg,
                AssumeRoleCreds {
                    access_key_id: "AKID".to_string(),
                    access_key_secret: "AKSK".to_string(),
                    security_token: "TOKEN".to_string(),
                    expires_at_ms: Some(1),
                },
            );
            assert!(!cfg.contains_key("role_arn"));
            assert!(!cfg.contains_key("role_session_name"));
            assert!(!cfg.contains_key("external_id"));
            assert_eq!(cfg.get("access_key_id").map(String::as_str), Some("AKID"));
            assert_eq!(cfg.get("access_key_secret").map(String::as_str), Some("AKSK"));
            assert_eq!(cfg.get("security_token").map(String::as_str), Some("TOKEN"));
            // Non-credential keys are preserved.
            assert_eq!(cfg.get("bucket").map(String::as_str), Some("b"));
            assert_eq!(cfg.get("endpoint").map(String::as_str), Some("e"));
        }

        #[test]
        fn pop_encode_unreserved_set() {
            // RFC 3986 unreserved chars stay as-is; everything else gets %XX upper-hex.
            assert_eq!(pop_encode("ABCabc012-_.~"), "ABCabc012-_.~");
            assert_eq!(pop_encode("a b"), "a%20b");
            assert_eq!(pop_encode("/"), "%2F");
            assert_eq!(pop_encode(":"), "%3A");
            assert_eq!(pop_encode("acs:ram::1:role/x"), "acs%3Aram%3A%3A1%3Arole%2Fx");
        }

        #[test]
        fn pop_sign_known_vector() {
            // Aliyun POP v1 DescribeRegions example, signed with POST (our
            // sts:AssumeRole is POSTed). Aliyun's doc publishes the GET variant
            // `OLeaidS1JvxuMvnyHOwuJ+uX5qY=`; the POST expected below was
            // independently recomputed from the POP v1 spec.
            let canonical =
                "AccessKeyId=testid&Action=DescribeRegions&Format=XML&SignatureMethod=HMAC-SHA1\
                 &SignatureNonce=3ee8c1b8-83d3-44af-a94f-4e0ad82fd6cf&SignatureVersion=1.0\
                 &Timestamp=2016-02-23T12%3A46%3A24Z&Version=2014-05-26";
            let sig = pop_sign("testsecret", canonical);
            assert_eq!(sig, "MxbnVAM4w6sft9xjVpe/GCKueuk=");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(kvs: &[(&str, &str)]) -> HashMap<String, String> {
        kvs.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn lance_opts_forward_role_arn_session_name_and_external_id() {
        let url = Url::parse("oss://my-bucket/prefix").unwrap();
        let storage_options = opts(&[
            ("oss_endpoint", "oss-cn-hangzhou.aliyuncs.com"),
            ("oss_role_arn", "acs:ram::111:role/tenant-A"),
            ("oss_role_session_name", "tenant-A-session"),
            ("oss_external_id", "tenant-A-ext"),
        ]);
        let cfg = build_oss_config_from_lance_opts("my-bucket".to_string(), &url, &storage_options)
            .unwrap();
        assert_eq!(cfg.get("role_arn").map(String::as_str), Some("acs:ram::111:role/tenant-A"));
        assert_eq!(cfg.get("role_session_name").map(String::as_str), Some("tenant-A-session"));
        assert_eq!(cfg.get("external_id").map(String::as_str), Some("tenant-A-ext"));
        assert_eq!(cfg.get("endpoint").map(String::as_str), Some("oss-cn-hangzhou.aliyuncs.com"));
        assert_eq!(cfg.get("bucket").map(String::as_str), Some("my-bucket"));
        // AK/SK not set by caller — must not appear in config (would trigger
        // reqsign's static-creds path ahead of AssumeRoleWithOIDC).
        assert!(!cfg.contains_key("access_key_id"));
        assert!(!cfg.contains_key("access_key_secret"));
    }

    #[test]
    fn missing_endpoint_is_rejected() {
        // No endpoint in storage_options and no OSS_ENDPOINT in env — expect
        // the Lance-path error string.
        let url = Url::parse("oss://my-bucket/").unwrap();
        // If the test host happens to have OSS_ENDPOINT set, skip — the
        // assertions below would be meaningless.
        if std::env::var("OSS_ENDPOINT").is_ok() {
            return;
        }

        let lance_err = build_oss_config_from_lance_opts(
            "my-bucket".to_string(),
            &url,
            &opts(&[("oss_role_arn", "acs:ram::111:role/tenant-A")]),
        )
        .unwrap_err();
        assert!(lance_err.contains("OSS endpoint is required"));
    }
}
