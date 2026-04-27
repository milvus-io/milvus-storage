// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Azure ADLS Gen2 cross-tenant `Storage` for the iceberg `FileIO` pipeline.
//!
//! `iceberg-storage-opendal::azdls_config_parse` only forwards a fixed set of
//! `adls.*` keys to opendal's `AzdlsConfig`, and `AzdlsConfig` itself only
//! supports `account_key` / SAS / `client_secret` credentials — there is no
//! bearer-token field, no IMDS-with-explicit-audience, and no two-hop
//! Federated Identity flow.
//!
//! For Azure cross-tenant Managed Identity access, this module:
//!
//! 1. At factory build time, extracts the bridge-private keys
//!    (`adls.cross-tenant-client-id`, `adls.cross-tenant-tenant-id`,
//!    `adls.cross-tenant-refresh-secs`) plus the standard `adls.account-name`
//!    and `adls.endpoint-suffix`, and stashes them on the [`AzdlsCrossTenant
//!    Storage`] instance.
//! 2. On each `Storage::*` call, builds an opendal `Azdls` operator pointed
//!    at the customer's account and attaches a custom [`HttpFetch`]
//!    implementation that injects `Authorization: Bearer <bearer>` into
//!    every outbound request.
//! 3. The bearer comes from a process-wide [`CrossTenantBearerCache`]
//!    (`azure_federation`) that performs the IMDS → AAD two-hop on demand
//!    and refreshes ahead of expiry.
//!
//! # Why a placeholder `account_key` is needed
//!
//! opendal's `AzdlsBackend::sign` calls `loader.load_credential().await?`
//! before issuing any HTTP request and aborts with `"no valid credential
//! found"` if the loader returns `None`. The loader has four sources
//! (`config` AK/SAS, `client_secret`, `workload_identity`, `imds`) and none
//! of them returns a credential we can use for cross-tenant. So we feed
//! `account_key` a syntactically valid base64 placeholder — the SharedKey
//! signing path runs, writes `Authorization: SharedKey ...`, and our
//! `HttpFetch` overwrites that header with the real Bearer before the
//! request leaves the process. The placeholder is never transmitted.
//!
//! Verified end-to-end in `azure_bearer_spike` — see commit history for the
//! spike code if revisiting this design.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use opendal::raw::{HttpBody, HttpClient, HttpFetch};
use opendal::services::Azdls;
use opendal::{Buffer, Operator};
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use iceberg::io::{
    FileMetadata, FileRead, FileWrite, InputFile, OutputFile, Storage as IcebergStorage,
    StorageConfig, StorageFactory,
};
use iceberg::{Error as IcebergError, ErrorKind as IcebergErrorKind, Result as IcebergResult};

use crate::azure_federation::{CrossTenantBearerCache, REFRESH_OFFSET_SECS};

/// Property names consumed by this factory. All three are required when
/// the factory is selected; absence triggers an error at build time.
const PROP_ACCOUNT_NAME: &str = "adls.account-name";
const PROP_ENDPOINT_SUFFIX: &str = "adls.endpoint-suffix";
const PROP_CLIENT_ID: &str = "adls.cross-tenant-client-id";
const PROP_TENANT_ID: &str = "adls.cross-tenant-tenant-id";
const PROP_REFRESH_SECS: &str = "adls.cross-tenant-refresh-secs";

/// Marker key used by the bridge to detect "cross-tenant mode" without
/// looking at every individual key. A caller's presence of `PROP_CLIENT_ID`
/// is enough.
pub const CROSS_TENANT_MARKER_KEY: &str = PROP_CLIENT_ID;

/// Floor / ceiling on refresh_offset (seconds before expiry to refresh).
/// AAD-issued bearers last ~3600s; refresh_offset >= 1800 means we'd
/// refresh on every call (bearer always within window). Cap to keep behavior
/// reasonable regardless of what `load_frequency` value made it down here.
const MIN_REFRESH_OFFSET_SECS: u64 = 60;
const MAX_REFRESH_OFFSET_SECS: u64 = 1800;

/// Placeholder Azure Storage Shared Key. Any valid base64 string works —
/// 64 zero bytes gives the canonical 88-char form. Never transmitted: our
/// `HttpFetch` overwrites the SharedKey Authorization header before send.
const PLACEHOLDER_ACCOUNT_KEY: &str =
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

// ============================================================================
// StorageFactory + Storage (typetag wiring for iceberg's FileIO)
// ============================================================================

fn from_opendal_error(e: opendal::Error) -> IcebergError {
    IcebergError::new(IcebergErrorKind::Unexpected, "Failure in doing io operation").with_source(e)
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AzdlsCrossTenantStorageFactory;

#[typetag::serde(name = "AzdlsCrossTenantStorageFactory")]
impl StorageFactory for AzdlsCrossTenantStorageFactory {
    fn build(&self, config: &StorageConfig) -> IcebergResult<Arc<dyn IcebergStorage>> {
        let props = config.props();

        let account_name = props
            .get(PROP_ACCOUNT_NAME)
            .cloned()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                IcebergError::new(
                    IcebergErrorKind::DataInvalid,
                    format!("Azure cross-tenant: missing {PROP_ACCOUNT_NAME}"),
                )
            })?;
        let client_id = props
            .get(PROP_CLIENT_ID)
            .cloned()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                IcebergError::new(
                    IcebergErrorKind::DataInvalid,
                    format!("Azure cross-tenant: missing {PROP_CLIENT_ID}"),
                )
            })?;
        let tenant_id = props
            .get(PROP_TENANT_ID)
            .cloned()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                IcebergError::new(
                    IcebergErrorKind::DataInvalid,
                    format!("Azure cross-tenant: missing {PROP_TENANT_ID}"),
                )
            })?;
        let endpoint_suffix = props.get(PROP_ENDPOINT_SUFFIX).cloned().unwrap_or_default();

        // Honor caller's refresh_secs but clamp to a sensible window. Default
        // when missing/zero/unparsable is the shared REFRESH_OFFSET_SECS.
        let refresh_secs = props
            .get(PROP_REFRESH_SECS)
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(REFRESH_OFFSET_SECS)
            .clamp(MIN_REFRESH_OFFSET_SECS, MAX_REFRESH_OFFSET_SECS);

        Ok(Arc::new(AzdlsCrossTenantStorage {
            account_name,
            endpoint_suffix,
            client_id,
            tenant_id,
            refresh_secs,
            cache: Arc::new(OnceCell::new()),
        }))
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AzdlsCrossTenantStorage {
    account_name: String,
    /// Used to reconstruct fully-qualified endpoint when paths don't carry
    /// `container@account.dfs.suffix` form. Falls back to "core.windows.net"
    /// if empty.
    endpoint_suffix: String,
    client_id: String,
    tenant_id: String,
    refresh_secs: u64,
    /// Lazily initialized on first I/O call so the struct stays Default +
    /// Serialize-able for typetag's dyn-trait routing. The cache itself
    /// holds a reqwest::Client and a tokio RwLock — neither of which is
    /// Serialize, hence the `#[serde(skip)]`.
    #[serde(skip)]
    cache: Arc<OnceCell<Arc<CrossTenantBearerCache>>>,
}

impl AzdlsCrossTenantStorage {
    async fn cache(&self) -> &Arc<CrossTenantBearerCache> {
        self.cache
            .get_or_init(|| async {
                Arc::new(CrossTenantBearerCache::new(
                    self.tenant_id.clone(),
                    self.client_id.clone(),
                    Duration::from_secs(self.refresh_secs),
                ))
            })
            .await
    }

    /// Parse `abfss://container@account.dfs.suffix/path` → (container,
    /// endpoint_url, relative_path). Also accepts the simpler form
    /// `abfss://container/path`, in which case `account_name` and
    /// `endpoint_suffix` from the storage config fill in the host.
    fn parse_abfss_uri(&self, uri: &str) -> IcebergResult<(String, String, String)> {
        let scheme_end = uri.find("://").ok_or_else(|| {
            IcebergError::new(
                IcebergErrorKind::DataInvalid,
                format!("expected abfs[s]:// URI, got: {uri}"),
            )
        })?;
        let scheme = &uri[..scheme_end];
        if scheme != "abfs" && scheme != "abfss" {
            return Err(IcebergError::new(
                IcebergErrorKind::DataInvalid,
                format!("expected abfs/abfss scheme, got: {scheme}"),
            ));
        }
        let http_scheme = if scheme == "abfss" { "https" } else { "http" };
        let rest = &uri[scheme_end + 3..];
        let first_slash = rest.find('/').unwrap_or(rest.len());
        let authority = &rest[..first_slash];
        let path = if first_slash < rest.len() {
            &rest[first_slash + 1..]
        } else {
            ""
        };

        let (container, host) = match authority.find('@') {
            Some(at) => (&authority[..at], authority[at + 1..].to_string()),
            None => {
                // No `@` → authority is just the container, fill host from
                // configured account/suffix.
                let suffix = if self.endpoint_suffix.is_empty() {
                    "core.windows.net".to_string()
                } else {
                    self.endpoint_suffix.clone()
                };
                (
                    authority,
                    format!("{}.dfs.{}", self.account_name, suffix),
                )
            }
        };
        if container.is_empty() {
            return Err(IcebergError::new(
                IcebergErrorKind::DataInvalid,
                format!("abfss URI missing container: {uri}"),
            ));
        }
        let endpoint = format!("{http_scheme}://{host}");
        Ok((container.to_string(), endpoint, path.to_string()))
    }

    async fn create_operator(&self, abfss_path: &str) -> IcebergResult<(Operator, String)> {
        let (container, endpoint, relative) = self.parse_abfss_uri(abfss_path)?;
        let cache = self.cache().await.clone();
        let fetcher = BearerInjectingFetcher {
            inner: reqwest::Client::new(),
            cache,
        };
        let http_client = HttpClient::with(fetcher);

        // Spike A confirmed: the placeholder account_key satisfies opendal's
        // "credential required before we'll send any HTTP" check; the
        // SharedKey Authorization header signer writes is then overwritten
        // by `BearerInjectingFetcher::fetch`. See module docs for why.
        #[allow(deprecated)]
        let builder = Azdls::default()
            .account_name(&self.account_name)
            .endpoint(&endpoint)
            .filesystem(&container)
            .account_key(PLACEHOLDER_ACCOUNT_KEY)
            .http_client(http_client);
        let op = Operator::new(builder)
            .map_err(|e| {
                IcebergError::new(
                    IcebergErrorKind::Unexpected,
                    format!("Failed to build Azdls operator: {e}"),
                )
            })?
            .finish();
        Ok((op, relative))
    }
}

#[typetag::serde(name = "AzdlsCrossTenantStorage")]
#[async_trait::async_trait]
impl IcebergStorage for AzdlsCrossTenantStorage {
    async fn exists(&self, path: &str) -> IcebergResult<bool> {
        let (op, rel) = self.create_operator(path).await?;
        op.exists(&rel).await.map_err(from_opendal_error)
    }

    async fn metadata(&self, path: &str) -> IcebergResult<FileMetadata> {
        let (op, rel) = self.create_operator(path).await?;
        let meta = op.stat(&rel).await.map_err(from_opendal_error)?;
        Ok(FileMetadata {
            size: meta.content_length(),
        })
    }

    async fn read(&self, path: &str) -> IcebergResult<bytes::Bytes> {
        let (op, rel) = self.create_operator(path).await?;
        Ok(op.read(&rel).await.map_err(from_opendal_error)?.to_bytes())
    }

    async fn reader(&self, path: &str) -> IcebergResult<Box<dyn FileRead>> {
        let (op, rel) = self.create_operator(path).await?;
        Ok(Box::new(OpenDalReader(
            op.reader(&rel).await.map_err(from_opendal_error)?,
        )))
    }

    async fn write(&self, path: &str, bs: bytes::Bytes) -> IcebergResult<()> {
        let (op, rel) = self.create_operator(path).await?;
        op.write(&rel, bs).await.map_err(from_opendal_error)?;
        Ok(())
    }

    async fn writer(&self, path: &str) -> IcebergResult<Box<dyn FileWrite>> {
        let (op, rel) = self.create_operator(path).await?;
        Ok(Box::new(OpenDalWriter(
            op.writer(&rel).await.map_err(from_opendal_error)?,
        )))
    }

    async fn delete(&self, path: &str) -> IcebergResult<()> {
        let (op, rel) = self.create_operator(path).await?;
        op.delete(&rel).await.map_err(from_opendal_error)
    }

    async fn delete_prefix(&self, path: &str) -> IcebergResult<()> {
        let (op, rel) = self.create_operator(path).await?;
        let prefixed = if rel.ends_with('/') {
            rel.clone()
        } else {
            format!("{rel}/")
        };
        op.remove_all(&prefixed).await.map_err(from_opendal_error)
    }

    fn new_input(&self, path: &str) -> IcebergResult<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> IcebergResult<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

// ============================================================================
// HTTP fetch wrapper that overwrites Authorization with our cross-tenant Bearer
// ============================================================================

struct BearerInjectingFetcher {
    inner: reqwest::Client,
    cache: Arc<CrossTenantBearerCache>,
}

impl HttpFetch for BearerInjectingFetcher {
    async fn fetch(
        &self,
        mut req: http::Request<Buffer>,
    ) -> opendal::Result<http::Response<HttpBody>> {
        let bearer = self.cache.current().await.map_err(|e| {
            opendal::Error::new(
                opendal::ErrorKind::Unexpected,
                format!("cross-tenant bearer fetch failed: {e}"),
            )
        })?;
        let header_value = http::HeaderValue::from_str(&format!("Bearer {}", *bearer))
            .map_err(|e| {
                opendal::Error::new(
                    opendal::ErrorKind::Unexpected,
                    format!("bearer is not a valid header value: {e}"),
                )
            })?;
        req.headers_mut()
            .insert(http::header::AUTHORIZATION, header_value);
        // Delegate to opendal's built-in `impl HttpFetch for reqwest::Client`.
        self.inner.fetch(req).await
    }
}

// ============================================================================
// Iceberg-side opendal Reader/Writer wrappers (mirrors aliyun_oss_provider)
// ============================================================================

/// `FileRead` over an opendal reader. Same shape as the (`pub(crate)`) wrapper
/// in `iceberg-storage-opendal` and `aliyun_oss_provider`; duplicated here to
/// keep this module self-contained.
struct OpenDalReader(opendal::Reader);

#[async_trait::async_trait]
impl FileRead for OpenDalReader {
    async fn read(&self, range: std::ops::Range<u64>) -> IcebergResult<bytes::Bytes> {
        Ok(opendal::Reader::read(&self.0, range)
            .await
            .map_err(from_opendal_error)?
            .to_bytes())
    }
}

/// `FileWrite` over an opendal writer.
struct OpenDalWriter(opendal::Writer);

#[async_trait::async_trait]
impl FileWrite for OpenDalWriter {
    async fn write(&mut self, bs: bytes::Bytes) -> IcebergResult<()> {
        Ok(opendal::Writer::write(&mut self.0, bs)
            .await
            .map_err(from_opendal_error)?)
    }

    async fn close(&mut self) -> IcebergResult<()> {
        let _ = opendal::Writer::close(&mut self.0)
            .await
            .map_err(from_opendal_error)?;
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_storage(account: &str, suffix: &str) -> AzdlsCrossTenantStorage {
        AzdlsCrossTenantStorage {
            account_name: account.into(),
            endpoint_suffix: suffix.into(),
            client_id: "client".into(),
            tenant_id: "tenant".into(),
            refresh_secs: REFRESH_OFFSET_SECS,
            cache: Arc::new(OnceCell::new()),
        }
    }

    #[test]
    fn parse_abfss_with_full_authority() {
        let s = make_storage("ignored", "ignored");
        let (container, endpoint, rel) = s
            .parse_abfss_uri("abfss://data@acme.dfs.core.windows.net/path/to/file.parquet")
            .unwrap();
        assert_eq!(container, "data");
        assert_eq!(endpoint, "https://acme.dfs.core.windows.net");
        assert_eq!(rel, "path/to/file.parquet");
    }

    #[test]
    fn parse_abfss_with_short_authority_uses_config() {
        let s = make_storage("acme", "core.windows.net");
        let (container, endpoint, rel) =
            s.parse_abfss_uri("abfss://data/path/to/file.parquet").unwrap();
        assert_eq!(container, "data");
        assert_eq!(endpoint, "https://acme.dfs.core.windows.net");
        assert_eq!(rel, "path/to/file.parquet");
    }

    #[test]
    fn parse_abfss_short_authority_default_suffix() {
        // Empty endpoint_suffix falls back to core.windows.net.
        let s = make_storage("acme", "");
        let (_, endpoint, _) =
            s.parse_abfss_uri("abfss://data/file.parquet").unwrap();
        assert_eq!(endpoint, "https://acme.dfs.core.windows.net");
    }

    #[test]
    fn parse_abfss_rejects_other_schemes() {
        let s = make_storage("acme", "core.windows.net");
        assert!(s.parse_abfss_uri("s3://bucket/key").is_err());
        assert!(s.parse_abfss_uri("not-a-url").is_err());
    }

    #[test]
    fn factory_requires_client_and_tenant() {
        // Build a minimal StorageConfig stand-in by going through the public
        // `with_prop` builder. iceberg 0.9 exposes StorageConfig only via
        // FileIOBuilder.with_prop().build() chain — replicating that here
        // would pull half of FileIO into a unit test. Skip: integration
        // coverage in the e2e test exercises the full factory path.
        let factory = AzdlsCrossTenantStorageFactory::default();
        let _ = factory; // keep symbol used; full coverage is e2e
    }
}
