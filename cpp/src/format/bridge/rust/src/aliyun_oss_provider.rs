// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Aliyun OSS store provider with per-tenant `role_arn` support.
//!
//! In the default OIDC mode this is a thin shim over opendal's `Oss` service:
//! opendal + reqsign handle `AssumeRoleWithOIDC` natively. The shim mirrors
//! lance-io's stock `OssStoreProvider` but also forwards `oss_role_arn` and
//! `oss_role_session_name` from `storage_options` into opendal's config,
//! which stock does not.
//!
//! In RAM mode (opt-in, see below) the module additionally resolves
//! `role_arn` to concrete short-lived credentials itself — IMDS +
//! `sts:AssumeRole` + POP v1 signing live in this file rather than in
//! opendal.
//!
//! Machine identity (`ALIBABA_CLOUD_OIDC_TOKEN_FILE` /
//! `ALIBABA_CLOUD_OIDC_PROVIDER_ARN`) continues to live in process env — the
//! env sweep preserved here routes them into opendal config.
//!
//! # Why a custom provider and not the stock one
//!
//! Stock `OssStoreProvider` only reads four `storage_options` keys
//! (`oss_endpoint`, `oss_access_key_id`, `oss_secret_access_key`, `oss_region`).
//! It cannot forward per-tenant `role_arn` / `role_session_name` — those can
//! only reach opendal via process env. Process env is per-tenant-hostile
//! (single-tenant only), so we add the missing two keys here.
//!
//! # RAM-mode alternative (ECS IMDS → sts:AssumeRole)
//!
//! On an Aliyun ECS instance with an attached RAM role (and no OIDC token),
//! set `ALIYUN_ROLE_ARN_AUTH_MODE=ram`. This path resolves
//! `role_arn` to short-lived STS credentials *in this module* (via IMDS then
//! `sts:AssumeRole`), then injects them into opendal's static-credential
//! slots (`access_key_id` / `access_key_secret` / `security_token`) and
//! strips `role_arn` so reqsign's AssumeRoleWithOIDC path never fires.
//! There is no auto-fallback — an unset or missing-OIDC-token environment
//! would otherwise silently route through the wrong path, which is exactly
//! the kind of thing an explicit env var is for.
//!
//! All RAM-mode helpers live in the inner [`ram`] module. The OIDC path
//! does not touch anything in `ram::*` — it flows through `build_oss_config`
//! and straight into opendal/reqsign.
//!
//! # Static AK/SK must not be forwarded on the OIDC role_arn path
//!
//! reqsign loads credentials in this order
//! (`reqsign::aliyun::credential.rs` — `load_via_static` before
//! `load_via_assume_role_with_oidc`): if static creds are set alongside
//! `role_arn`, the OIDC path is silently skipped. `lance_common.cpp` must
//! not emit AK/SK when `role_arn` is set. In RAM mode this is turned on its
//! head — we *do* want reqsign to use the static creds we just derived,
//! and we strip `role_arn` ourselves to keep that route unambiguous.

use std::collections::HashMap;
use std::sync::Arc;

use object_store::ObjectStore as OSObjectStore;
use object_store_opendal::OpendalStore;
use opendal::{Operator, services::Oss};
use snafu::location;
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

        let mut config_map = build_oss_config(bucket, &base_path, &storage_options)?;

        // RAM-mode swap: when `ALIYUN_ROLE_ARN_AUTH_MODE=ram` is set in env
        // and `role_arn` is present in the config, turn the indirection into
        // concrete creds here. No-op in every other case — OIDC callers,
        // plain AK/SK callers, and any caller without the env var flipped
        // fall straight through to `Operator::from_iter` below.
        //
        // FIXME(aliyun-ram-refresh): the STS creds resolved here are injected
        // as static `access_key_id` / `access_key_secret` / `security_token`
        // into opendal's config and never refreshed. Aliyun `sts:AssumeRole`
        // tokens expire after ~1h, so any operator kept alive past that
        // window (long scans, compactions) will start getting 403s until the
        // next `new_store` call. Not fixed here because production deploys
        // use the OIDC path (`ALIBABA_CLOUD_OIDC_TOKEN_FILE` +
        // `AssumeRoleWithOIDC`), where reqsign handles refresh internally —
        // RAM mode exists for ECS-with-RAM-role dev/bench environments only.
        // If RAM mode ever becomes a supported production path, wrap the
        // operator so it re-fetches via `fetch_assume_role_creds` before
        // expiration.
        if std::env::var(ram::AUTH_MODE_ENV).as_deref() == Ok(ram::AUTH_MODE_RAM) {
            if let Some(role_arn) = config_map.get("role_arn").cloned() {
                let session_name = config_map
                    .get("role_session_name")
                    .cloned()
                    .unwrap_or_else(ram::default_session_name);
                let creds = ram::fetch_assume_role_creds(&role_arn, &session_name)
                    .await
                    .map_err(|e| {
                        LanceError::invalid_input(
                            format!("Aliyun RAM-mode credential resolution failed: {e}"),
                            location!(),
                        )
                    })?;
                ram::apply_credentials(&mut config_map, creds);
            }
        }

        let operator = Operator::from_iter::<Oss>(config_map)
            .map_err(|e| {
                LanceError::invalid_input(
                    format!("Failed to create OSS operator: {:?}", e),
                    location!(),
                )
            })?
            .finish();

        let inner = Arc::new(OpendalStore::new(operator)) as Arc<dyn OSObjectStore>;

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

/// Build the opendal OSS config map from env + `storage_options`.
///
/// Extracted as a pure function so tests can assert the forwarding logic
/// without building a live `Operator`.
fn build_oss_config(
    bucket: String,
    base_path: &Url,
    storage_options: &StorageOptions,
) -> LanceResult<HashMap<String, String>> {
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

    // storage_options overrides (later wins over env). Stock four keys:
    if let Some(endpoint) = storage_options.0.get("oss_endpoint") {
        config_map.insert("endpoint".to_string(), endpoint.clone());
    }
    if let Some(access_key_id) = storage_options.0.get("oss_access_key_id") {
        config_map.insert("access_key_id".to_string(), access_key_id.clone());
    }
    if let Some(secret_access_key) = storage_options.0.get("oss_secret_access_key") {
        config_map.insert("access_key_secret".to_string(), secret_access_key.clone());
    }
    if let Some(region) = storage_options.0.get("oss_region") {
        config_map.insert("region".to_string(), region.clone());
    }

    // The two keys stock does NOT forward. This is the reason this provider
    // exists — per-tenant role_arn / session_name can only reach opendal via
    // `storage_options`.
    if let Some(role_arn) = storage_options.0.get("oss_role_arn") {
        config_map.insert("role_arn".to_string(), role_arn.clone());
    }
    if let Some(role_session_name) = storage_options.0.get("oss_role_session_name") {
        config_map.insert("role_session_name".to_string(), role_session_name.clone());
    }

    if !config_map.contains_key("endpoint") {
        return Err(LanceError::invalid_input(
            "OSS endpoint is required. Provide 'oss_endpoint' in storage options or set OSS_ENDPOINT environment variable",
            location!(),
        ));
    }

    Ok(config_map)
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

/// ECS IMDS → `sts:AssumeRole` path. Nothing outside this module should call
/// into `ram::*`; the OIDC path bypasses it entirely.
mod ram {
    use std::collections::HashMap;
    use std::time::Duration;

    use base64::Engine;
    use chrono::Utc;
    use hmac::{Hmac, Mac};
    use serde::Deserialize;
    use sha1::Sha1;

    /// Explicit opt-in env var for the ECS-IMDS → AssumeRole flow. `"ram"` selects
    /// it; anything else (including unset) keeps the default OIDC behaviour.
    pub(super) const AUTH_MODE_ENV: &str = "ALIYUN_ROLE_ARN_AUTH_MODE";
    pub(super) const AUTH_MODE_RAM: &str = "ram";

    /// ECS metadata service — Aliyun's fixed link-local HTTP endpoint.
    const IMDS_BASE: &str = "http://100.100.100.200";
    const IMDS_ROLE_LIST_PATH: &str = "/latest/meta-data/ram/security-credentials/";
    const IMDS_V2_TOKEN_PATH: &str = "/latest/api/token";
    /// Max TTL IMDS accepts for a V2 session token.
    const IMDS_V2_TTL_SECS: u64 = 21600;

    const STS_ENDPOINT: &str = "https://sts.aliyuncs.com/";

    /// Short-lived credentials returned from `sts:AssumeRole`.
    #[derive(Debug, Clone)]
    pub(super) struct AssumeRoleCreds {
        pub(super) access_key_id: String,
        pub(super) access_key_secret: String,
        pub(super) security_token: String,
    }

    /// Mutate `config_map` for the RAM-mode hand-off: insert the concrete STS
    /// creds that opendal's static-credential path expects, and remove the
    /// `role_arn` / `role_session_name` keys so reqsign cannot re-enter its own
    /// AssumeRoleWithOIDC flow (see module-level comment: static creds alongside
    /// `role_arn` make reqsign silently pick the static path — here that's what
    /// we want, *and* keeping `role_arn` around would be a correctness landmine
    /// if reqsign's preference ever inverted).
    pub(super) fn apply_credentials(
        config_map: &mut HashMap<String, String>,
        creds: AssumeRoleCreds,
    ) {
        config_map.remove("role_arn");
        config_map.remove("role_session_name");
        config_map.insert("access_key_id".to_string(), creds.access_key_id);
        config_map.insert("access_key_secret".to_string(), creds.access_key_secret);
        config_map.insert("security_token".to_string(), creds.security_token);
    }

    pub(super) fn default_session_name() -> String {
        format!("milvus-storage-ram-{}", uuid::Uuid::now_v7())
    }

    pub(super) async fn fetch_assume_role_creds(
        role_arn: &str,
        role_session_name: &str,
    ) -> Result<AssumeRoleCreds, String> {
        let client = build_http_client()?;
        let caller = fetch_imds_credentials(&client).await?;
        call_assume_role(&client, &caller, role_arn, role_session_name).await
    }

    /// Build a reqwest client with short timeouts tuned for IMDS + STS. IMDS
    /// answers in milliseconds on a healthy ECS; STS is a single round-trip on
    /// the public internet. A stalled request here would block the caller's
    /// whole `new_store`, so the tight caps are intentional.
    fn build_http_client() -> Result<reqwest::Client, String> {
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

    async fn call_assume_role(
        client: &reqwest::Client,
        caller: &AssumeRoleCreds,
        role_arn: &str,
        role_session_name: &str,
    ) -> Result<AssumeRoleCreds, String> {
        // BTreeMap for deterministic ASCII ordering of keys (POP v1 requirement).
        let mut params = std::collections::BTreeMap::new();
        params.insert("AccessKeyId", caller.access_key_id.as_str());
        params.insert("Action", "AssumeRole");
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
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "PascalCase")]
        struct AssumeRoleResponseJson {
            credentials: CredentialsJson,
        }
        let parsed: AssumeRoleResponseJson = serde_json::from_str(&text).map_err(|e| {
            format!("sts:AssumeRole JSON parse failed ({e}); body was: {text}")
        })?;
        Ok(AssumeRoleCreds {
            access_key_id: parsed.credentials.access_key_id,
            access_key_secret: parsed.credentials.access_key_secret,
            security_token: parsed.credentials.security_token,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn apply_credentials_strips_role_arn_and_injects_static_creds() {
            // Simulates the state right before the opendal operator is built in
            // RAM mode: `role_arn` + `role_session_name` must disappear, concrete
            // AKID / SK / token must be present. If any of these assertions break
            // we risk reqsign re-entering AssumeRoleWithOIDC with a role_arn it
            // can't authenticate against.
            let mut cfg: HashMap<String, String> = HashMap::new();
            cfg.insert("bucket".to_string(), "b".to_string());
            cfg.insert("endpoint".to_string(), "e".to_string());
            cfg.insert("role_arn".to_string(), "acs:ram::1:role/x".to_string());
            cfg.insert("role_session_name".to_string(), "sess".to_string());
            apply_credentials(
                &mut cfg,
                AssumeRoleCreds {
                    access_key_id: "AKID".to_string(),
                    access_key_secret: "AKSK".to_string(),
                    security_token: "TOKEN".to_string(),
                },
            );
            assert!(!cfg.contains_key("role_arn"));
            assert!(!cfg.contains_key("role_session_name"));
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
            // Aliyun's canonical POP v1 example from their SDK reference doc.
            // Given SecretAccessKey="testsecret" + "&" as the signing key, and
            // the canonical query below, the signature must reproduce exactly.
            // If this fails we're miscomputing either StringToSign or the HMAC.
            let canonical =
                "AccessKeyId=testid&Action=DescribeRegions&Format=XML&SignatureMethod=HMAC-SHA1\
                 &SignatureNonce=3ee8c1b8-83d3-44af-a94f-4e0ad82fd6cf&SignatureVersion=1.0\
                 &Timestamp=2016-02-23T12%3A46%3A24Z&Version=2014-05-26";
            let sig = pop_sign("testsecret", canonical);
            assert_eq!(sig, "OLeaidS1JvxuMvnyHOwuJ+uX5qY=");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(kvs: &[(&str, &str)]) -> StorageOptions {
        StorageOptions(kvs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
    }

    #[test]
    fn forwards_role_arn_and_session_name() {
        let url = Url::parse("oss://my-bucket/prefix").unwrap();
        let storage_options = opts(&[
            ("oss_endpoint", "oss-cn-hangzhou.aliyuncs.com"),
            ("oss_role_arn", "acs:ram::111:role/tenant-A"),
            ("oss_role_session_name", "tenant-A-session"),
        ]);
        let cfg = build_oss_config("my-bucket".to_string(), &url, &storage_options).unwrap();
        assert_eq!(cfg.get("role_arn").map(String::as_str), Some("acs:ram::111:role/tenant-A"));
        assert_eq!(cfg.get("role_session_name").map(String::as_str), Some("tenant-A-session"));
        assert_eq!(cfg.get("endpoint").map(String::as_str), Some("oss-cn-hangzhou.aliyuncs.com"));
        assert_eq!(cfg.get("bucket").map(String::as_str), Some("my-bucket"));
        // AK/SK not set by caller — must not appear in config (would trigger
        // reqsign's static-creds path ahead of AssumeRoleWithOIDC).
        assert!(!cfg.contains_key("access_key_id"));
        assert!(!cfg.contains_key("access_key_secret"));
    }

    #[test]
    fn storage_options_override_env() {
        // Verify that values from storage_options land in the opendal config
        // map. We intentionally do not call `set_var` to set a conflicting
        // env var here — Rust tests run in parallel by default and `set_var`
        // is process-global, which would race with any other test that reads
        // env. A stricter "storage_options beats env" check would need to
        // serialise env access.
        let url = Url::parse("oss://my-bucket/").unwrap();
        let storage_options = opts(&[
            ("oss_endpoint", "from-storage-options.aliyuncs.com"),
            ("oss_role_arn", "acs:ram::222:role/from-storage-options"),
        ]);
        let cfg = build_oss_config("my-bucket".to_string(), &url, &storage_options).unwrap();
        assert_eq!(cfg.get("endpoint").map(String::as_str), Some("from-storage-options.aliyuncs.com"));
        assert_eq!(
            cfg.get("role_arn").map(String::as_str),
            Some("acs:ram::222:role/from-storage-options")
        );
    }

    #[test]
    fn missing_endpoint_is_rejected() {
        // No endpoint in storage_options and no OSS_ENDPOINT in env — expect an
        // invalid_input error, matching stock provider behavior.
        let url = Url::parse("oss://my-bucket/").unwrap();
        let storage_options = opts(&[("oss_role_arn", "acs:ram::111:role/tenant-A")]);
        // If the test host happens to have OSS_ENDPOINT set, skip — the assertion
        // below would be meaningless.
        if std::env::var("OSS_ENDPOINT").is_ok() {
            return;
        }
        let err = build_oss_config("my-bucket".to_string(), &url, &storage_options).unwrap_err();
        assert!(format!("{}", err).contains("OSS endpoint is required"));
    }
}
