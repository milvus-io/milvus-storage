// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result as AnyResult, anyhow, bail};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use iceberg::io::{
    ADLS_SAS_TOKEN, FileMetadata, FileRead, FileWrite, InputFile, OutputFile,
    Storage as IcebergStorage, StorageConfig, StorageFactory,
};
use iceberg::{Error as IcebergError, ErrorKind as IcebergErrorKind, Result as IcebergResult};
use iceberg_storage_opendal::OpenDalStorageFactory;
use lance_core::error::{Error as LanceError, Result as LanceResult};
use lance_io::object_store::StorageOptionsProvider;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

pub(crate) const AZURE_BROKER_ENDPOINT: &str = "azure_broker_endpoint";
pub(crate) const AZURE_BROKER_CLIENT_ID: &str = "azure_broker_client_id";
pub(crate) const AZURE_BROKER_TENANT_ID: &str = "azure_broker_tenant_id";
pub(crate) const AZURE_BROKER_ACCOUNT_NAME: &str = "azure_broker_account_name";
pub(crate) const AZURE_BROKER_REGION: &str = "azure_broker_region";
pub(crate) const AZURE_BROKER_BUCKET: &str = "azure_broker_bucket";
pub(crate) const AZURE_BROKER_DURATION_SECONDS: &str = "azure_broker_duration_seconds";
pub(crate) const AZURE_BROKER_REQUEST_TIMEOUT_MS: &str = "azure_broker_request_timeout_ms";

const REFRESH_OFFSET_SECONDS: i64 = 60;

// These options are private to the C++/Rust bridge. They must be consumed
// before the remaining storage options are passed to Lance or OpenDAL.
const BROKER_KEYS: [&str; 8] = [
    AZURE_BROKER_ENDPOINT,
    AZURE_BROKER_CLIENT_ID,
    AZURE_BROKER_TENANT_ID,
    AZURE_BROKER_ACCOUNT_NAME,
    AZURE_BROKER_REGION,
    AZURE_BROKER_BUCKET,
    AZURE_BROKER_DURATION_SECONDS,
    AZURE_BROKER_REQUEST_TIMEOUT_MS,
];

/// Typed Azure credential broker configuration produced from bridge-private
/// storage options populated by the C++ Lance and Iceberg adapters.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct AzureBrokerConfig {
    endpoint: String,
    client_id: String,
    tenant_id: String,
    account_name: String,
    region: String,
    bucket: String,
    duration_seconds: u64,
    request_timeout_ms: u64,
}

impl AzureBrokerConfig {
    /// Removes and parses the broker options from `options`.
    ///
    /// No broker key means that broker authentication is disabled. Once any
    /// broker key is present, the configuration is treated as all-or-nothing so
    /// a partial setup cannot silently fall back to another Azure auth mode.
    pub(crate) fn extract(options: &mut HashMap<String, String>) -> AnyResult<Option<Self>> {
        let enabled = BROKER_KEYS.iter().any(|key| options.contains_key(*key));
        if !enabled {
            return Ok(None);
        }

        let mut take = |key: &str| options.remove(key).unwrap_or_default();
        let config = Self {
            endpoint: take(AZURE_BROKER_ENDPOINT),
            client_id: take(AZURE_BROKER_CLIENT_ID),
            tenant_id: take(AZURE_BROKER_TENANT_ID),
            account_name: take(AZURE_BROKER_ACCOUNT_NAME),
            region: take(AZURE_BROKER_REGION),
            bucket: take(AZURE_BROKER_BUCKET),
            duration_seconds: take(AZURE_BROKER_DURATION_SECONDS).parse().unwrap_or(0),
            request_timeout_ms: take(AZURE_BROKER_REQUEST_TIMEOUT_MS).parse().unwrap_or(0),
        };

        if config.endpoint.is_empty()
            || config.client_id.is_empty()
            || config.tenant_id.is_empty()
            || config.account_name.is_empty()
            || config.region.is_empty()
            || config.bucket.is_empty()
            || config.duration_seconds == 0
            || config.request_timeout_ms == 0
        {
            bail!("incomplete Azure credential broker configuration");
        }

        let endpoint = url::Url::parse(&config.endpoint)
            .map_err(|_| anyhow!("Azure credential broker endpoint is not a valid URL"))?;
        if (endpoint.scheme() != "http" && endpoint.scheme() != "https") || !endpoint.has_host() {
            bail!("Azure credential broker endpoint must use HTTP or HTTPS");
        }

        Ok(Some(config))
    }

    fn provider_id(&self) -> String {
        // Lance uses this opaque ID to distinguish providers without embedding
        // the raw broker configuration in the identifier.
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        format!("azure_sas_broker_{:016x}", hasher.finish())
    }
}

#[derive(Clone)]
pub(crate) struct AzureSasCredential {
    pub(crate) token: String,
    pub(crate) expires_at: DateTime<Utc>,
}

#[derive(Clone)]
pub(crate) struct AzureBrokerClient {
    config: AzureBrokerConfig,
    client: reqwest::Client,
}

impl fmt::Debug for AzureBrokerClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureBrokerClient")
            .field("provider_id", &self.config.provider_id())
            .finish()
    }
}

#[derive(Serialize)]
struct BrokerRequest<'a> {
    csp: &'static str,
    region: &'a str,
    bucket: &'a str,
    #[serde(rename = "durationSeconds")]
    duration_seconds: u64,
    #[serde(rename = "azureClientId")]
    azure_client_id: &'a str,
    #[serde(rename = "azureTenantId")]
    azure_tenant_id: &'a str,
    #[serde(rename = "azureAccountName")]
    azure_account_name: &'a str,
}

#[derive(Deserialize)]
struct BrokerResponse {
    success: bool,
    credentials: Option<BrokerCredentials>,
}

#[derive(Deserialize)]
struct BrokerCredentials {
    #[serde(rename = "tempAk")]
    temp_ak: String,
    #[serde(rename = "sessionToken")]
    session_token: String,
    #[serde(rename = "expiredAt")]
    expired_at: String,
}

impl AzureBrokerClient {
    pub(crate) fn new(config: AzureBrokerConfig) -> AnyResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.request_timeout_ms))
            .build()
            .map_err(|_| anyhow!("failed to construct Azure credential broker HTTP client"))?;
        Ok(Self { config, client })
    }

    pub(crate) async fn fetch(&self, now: DateTime<Utc>) -> AnyResult<AzureSasCredential> {
        let request = BrokerRequest {
            csp: "azure",
            region: &self.config.region,
            bucket: &self.config.bucket,
            duration_seconds: self.config.duration_seconds,
            azure_client_id: &self.config.client_id,
            azure_tenant_id: &self.config.tenant_id,
            azure_account_name: &self.config.account_name,
        };
        let response = self
            .client
            .post(&self.config.endpoint)
            .json(&request)
            .send()
            .await
            .map_err(|_| anyhow!("transport_error"))?;
        let status = response.status();
        if !status.is_success() {
            bail!("http_status={}", status.as_u16());
        }
        let response: BrokerResponse =
            response.json().await.map_err(|_| anyhow!("invalid_json"))?;
        if !response.success {
            bail!("business_failure");
        }
        let credentials = response
            .credentials
            .ok_or_else(|| anyhow!("missing_credentials"))?;
        if credentials.temp_ak != self.config.account_name {
            bail!("account_mismatch");
        }
        let token = credentials
            .session_token
            .trim_start_matches('?')
            .to_string();
        if token.is_empty() {
            bail!("empty_sas");
        }
        if !url::form_urlencoded::parse(token.as_bytes())
            .any(|(key, value)| key == "sig" && !value.is_empty())
        {
            bail!("missing_sas_signature");
        }
        let expires_at = DateTime::parse_from_rfc3339(&credentials.expired_at)
            .map_err(|_| anyhow!("invalid_expiration"))?
            .with_timezone(&Utc);
        if expires_at <= now {
            bail!("expired_credential");
        }
        Ok(AzureSasCredential { token, expires_at })
    }
}

#[async_trait]
trait AzureSasFetcher: Send + Sync {
    async fn fetch(&self, now: DateTime<Utc>) -> AnyResult<AzureSasCredential>;
}

#[async_trait]
impl AzureSasFetcher for AzureBrokerClient {
    async fn fetch(&self, now: DateTime<Utc>) -> AnyResult<AzureSasCredential> {
        AzureBrokerClient::fetch(self, now).await
    }
}

type Clock = Arc<dyn Fn() -> DateTime<Utc> + Send + Sync>;

pub(crate) struct AzureSasStorageOptionsProvider {
    provider_id: String,
    fetcher: Arc<dyn AzureSasFetcher>,
    clock: Clock,
    cached: RwLock<Option<AzureSasCredential>>,
}

impl fmt::Debug for AzureSasStorageOptionsProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureSasStorageOptionsProvider")
            .field("provider_id", &self.provider_id)
            .field(
                "has_cached_sas",
                &self.cached.try_read().map(|v| v.is_some()).unwrap_or(false),
            )
            .finish()
    }
}

impl AzureSasStorageOptionsProvider {
    pub(crate) fn new(config: AzureBrokerConfig) -> AnyResult<Self> {
        let provider_id = config.provider_id();
        let fetcher = Arc::new(AzureBrokerClient::new(config)?);
        Ok(Self {
            provider_id,
            fetcher,
            clock: Arc::new(Utc::now),
            cached: RwLock::new(None),
        })
    }

    #[cfg(test)]
    fn with_fetcher(
        config: AzureBrokerConfig,
        fetcher: Arc<dyn AzureSasFetcher>,
        clock: Clock,
    ) -> Self {
        Self {
            provider_id: config.provider_id(),
            fetcher,
            clock,
            cached: RwLock::new(None),
        }
    }

    fn is_fresh(credential: &AzureSasCredential, now: DateTime<Utc>) -> bool {
        credential.expires_at - now > chrono::Duration::seconds(REFRESH_OFFSET_SECONDS)
    }

    fn to_options(credential: &AzureSasCredential) -> HashMap<String, String> {
        HashMap::from([
            (
                "azure_storage_sas_token".to_string(),
                credential.token.clone(),
            ),
            (
                "expires_at_millis".to_string(),
                credential.expires_at.timestamp_millis().to_string(),
            ),
        ])
    }

    fn lance_error(error: &anyhow::Error) -> LanceError {
        LanceError::io_source(Box::new(std::io::Error::other(format!(
            "Azure SAS credential broker failure: {error}"
        ))))
    }

    pub(crate) async fn current_credential(&self) -> AnyResult<AzureSasCredential> {
        let mut now = (self.clock)();
        {
            let cached = self.cached.read().await;
            if let Some(credential) = cached.as_ref()
                && Self::is_fresh(credential, now)
            {
                return Ok(credential.clone());
            }
        }

        let mut cached = self.cached.write().await;
        now = (self.clock)();
        if let Some(credential) = cached.as_ref()
            && Self::is_fresh(credential, now)
        {
            return Ok(credential.clone());
        }

        match self.fetcher.fetch(now).await {
            Ok(credential) => {
                *cached = Some(credential.clone());
                Ok(credential)
            }
            Err(error) => {
                let has_cached_sas = cached.is_some();
                let cached_expired = cached
                    .as_ref()
                    .map(|credential| credential.expires_at <= now)
                    .unwrap_or(false);
                eprintln!(
                    "Warning: Azure SAS credential broker refresh failed: {}, has_cached_sas={}, cached_expired={}",
                    error, has_cached_sas, cached_expired
                );
                Err(error)
            }
        }
    }
}

#[async_trait]
impl StorageOptionsProvider for AzureSasStorageOptionsProvider {
    async fn fetch_storage_options(&self) -> LanceResult<Option<HashMap<String, String>>> {
        self.current_credential()
            .await
            .map(|credential| Some(Self::to_options(&credential)))
            .map_err(|error| Self::lance_error(&error))
    }

    fn provider_id(&self) -> String {
        self.provider_id.clone()
    }
}

fn azure_iceberg_credential_error(error: anyhow::Error) -> IcebergError {
    IcebergError::new(
        IcebergErrorKind::Unexpected,
        format!("Azure SAS credential resolution failed: {error}"),
    )
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct AzureSasStorageFactory {
    inner: OpenDalStorageFactory,
    #[serde(skip)]
    provider: Option<Arc<AzureSasStorageOptionsProvider>>,
}

impl AzureSasStorageFactory {
    pub(crate) async fn new(
        inner: OpenDalStorageFactory,
        config: AzureBrokerConfig,
    ) -> IcebergResult<Self> {
        let provider = Arc::new(
            AzureSasStorageOptionsProvider::new(config).map_err(azure_iceberg_credential_error)?,
        );
        provider
            .current_credential()
            .await
            .map_err(azure_iceberg_credential_error)?;
        Ok(Self {
            inner,
            provider: Some(provider),
        })
    }
}

impl fmt::Debug for AzureSasStorageFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureSasStorageFactory")
            .field("has_runtime_provider", &self.provider.is_some())
            .finish()
    }
}

#[typetag::serde(name = "AzureSasStorageFactory")]
impl StorageFactory for AzureSasStorageFactory {
    fn build(&self, config: &StorageConfig) -> IcebergResult<Arc<dyn IcebergStorage>> {
        let provider = self.provider.clone().ok_or_else(|| {
            IcebergError::new(
                IcebergErrorKind::Unexpected,
                "Azure SAS runtime provider is unavailable",
            )
        })?;
        Ok(Arc::new(AzureSasStorage {
            inner: self.inner.clone(),
            props: Arc::new(config.props().clone()),
            provider: Some(provider),
            cached_storage: Arc::new(RwLock::new(None)),
        }))
    }
}

#[derive(Clone)]
struct CachedAzureSasStorage {
    token: String,
    storage: Arc<dyn IcebergStorage>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct AzureSasStorage {
    inner: OpenDalStorageFactory,
    #[serde(skip)]
    props: Arc<HashMap<String, String>>,
    #[serde(skip)]
    provider: Option<Arc<AzureSasStorageOptionsProvider>>,
    #[serde(skip)]
    cached_storage: Arc<RwLock<Option<CachedAzureSasStorage>>>,
}

impl fmt::Debug for AzureSasStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureSasStorage")
            .field("has_runtime_provider", &self.provider.is_some())
            .finish()
    }
}

impl AzureSasStorage {
    async fn current_storage(&self) -> IcebergResult<Arc<dyn IcebergStorage>> {
        let provider = self.provider.as_ref().ok_or_else(|| {
            IcebergError::new(
                IcebergErrorKind::Unexpected,
                "Azure SAS runtime provider is unavailable",
            )
        })?;
        let credential = provider
            .current_credential()
            .await
            .map_err(azure_iceberg_credential_error)?;
        let token = credential.token;
        {
            let cached = self.cached_storage.read().await;
            if let Some(cached) = cached.as_ref()
                && cached.token == token
            {
                return Ok(cached.storage.clone());
            }
        }

        let mut cached = self.cached_storage.write().await;
        if let Some(cached) = cached.as_ref()
            && cached.token == token
        {
            return Ok(cached.storage.clone());
        }

        let mut props = self.props.as_ref().clone();
        props.insert(ADLS_SAS_TOKEN.to_string(), token.clone());
        let storage = self.inner.build(&StorageConfig::from_props(props))?;
        *cached = Some(CachedAzureSasStorage {
            token,
            storage: storage.clone(),
        });
        Ok(storage)
    }
}

#[typetag::serde(name = "AzureSasStorage")]
#[async_trait]
impl IcebergStorage for AzureSasStorage {
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

pub(crate) async fn build_lance_provider(
    config: AzureBrokerConfig,
) -> LanceResult<Arc<AzureSasStorageOptionsProvider>> {
    let provider = Arc::new(
        AzureSasStorageOptionsProvider::new(config)
            .map_err(|error| LanceError::invalid_input(error.to_string()))?,
    );
    provider
        .current_credential()
        .await
        .map_err(|error| AzureSasStorageOptionsProvider::lance_error(&error))?;
    Ok(provider)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::future::join_all;
    use iceberg::io::{StorageConfig, StorageFactory};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::{Barrier, Notify};

    use super::*;

    fn config() -> AzureBrokerConfig {
        AzureBrokerConfig {
            endpoint: "http://credential-broker/v1/credentials/assume-role".to_string(),
            client_id: "client".to_string(),
            tenant_id: "tenant".to_string(),
            account_name: "account".to_string(),
            region: "westus3".to_string(),
            bucket: "container".to_string(),
            duration_seconds: 3600,
            request_timeout_ms: 1000,
        }
    }

    fn inner_factory() -> OpenDalStorageFactory {
        serde_json::from_str(r#"{"Azdls":{"configured_scheme":"Abfss"}}"#).unwrap()
    }

    struct MockFetcher {
        responses: Mutex<VecDeque<Result<AzureSasCredential, &'static str>>>,
        calls: AtomicUsize,
        started: Option<Arc<Notify>>,
        release: Option<Arc<Notify>>,
    }

    impl MockFetcher {
        fn new(responses: Vec<Result<AzureSasCredential, &'static str>>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                calls: AtomicUsize::new(0),
                started: None,
                release: None,
            }
        }

        fn with_gate(
            responses: Vec<Result<AzureSasCredential, &'static str>>,
            started: Arc<Notify>,
            release: Arc<Notify>,
        ) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                calls: AtomicUsize::new(0),
                started: Some(started),
                release: Some(release),
            }
        }
    }

    #[async_trait]
    impl AzureSasFetcher for MockFetcher {
        async fn fetch(&self, _now: DateTime<Utc>) -> AnyResult<AzureSasCredential> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(started) = &self.started {
                started.notify_one();
            }
            if let Some(release) = &self.release {
                release.notified().await;
            }
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(Err("no_response"))
                .map_err(|error| anyhow!(error))
        }
    }

    fn credential(now: DateTime<Utc>, signature: &str) -> AzureSasCredential {
        AzureSasCredential {
            token: format!("sv=1&sig={signature}"),
            expires_at: now + chrono::Duration::hours(1),
        }
    }

    fn test_provider(
        fetcher: Arc<dyn AzureSasFetcher>,
        now: DateTime<Utc>,
    ) -> Arc<AzureSasStorageOptionsProvider> {
        Arc::new(AzureSasStorageOptionsProvider::with_fetcher(
            config(),
            fetcher,
            Arc::new(move || now),
        ))
    }

    #[tokio::test]
    async fn iceberg_factory_build_does_not_fetch_credentials() {
        let now = Utc::now();
        let fetcher = Arc::new(MockFetcher::new(vec![Ok(credential(now, "warm"))]));
        let provider = test_provider(fetcher.clone(), now);
        provider.current_credential().await.unwrap();
        let factory = AzureSasStorageFactory {
            inner: inner_factory(),
            provider: Some(provider),
        };

        let _storage = factory.build(&StorageConfig::new()).unwrap();
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn iceberg_storage_reuses_delegated_storage_until_sas_changes() {
        let now = Utc::now();
        let provider = test_provider(Arc::new(MockFetcher::new(Vec::new())), now);
        *provider.cached.write().await = Some(credential(now, "first"));
        let storage = AzureSasStorage {
            inner: inner_factory(),
            props: Arc::new(HashMap::new()),
            provider: Some(provider.clone()),
            cached_storage: Arc::new(RwLock::new(None)),
        };

        let first = storage.current_storage().await.unwrap();
        let second = storage.current_storage().await.unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        *provider.cached.write().await = Some(credential(now, "second"));
        let refreshed = storage.current_storage().await.unwrap();
        assert!(!Arc::ptr_eq(&first, &refreshed));
    }

    #[test]
    fn iceberg_storage_serialization_does_not_retain_unrelated_props() {
        let now = Utc::now();
        let factory = AzureSasStorageFactory {
            inner: inner_factory(),
            provider: Some(test_provider(Arc::new(MockFetcher::new(Vec::new())), now)),
        };
        let storage = factory
            .build(
                &StorageConfig::new()
                    .with_prop(iceberg::io::ADLS_AUTHORITY_HOST, "https://login.example")
                    .with_prop("unrelated.secret", "props-secret-sentinel"),
            )
            .unwrap();

        let serialized = serde_json::to_string(&storage).unwrap();
        assert!(!serialized.contains("props-secret-sentinel"));
        assert!(!serialized.contains("unrelated.secret"));
    }

    #[test]
    fn extracts_and_removes_private_options() {
        let mut options = HashMap::from([
            (AZURE_BROKER_ENDPOINT.to_string(), config().endpoint),
            (AZURE_BROKER_CLIENT_ID.to_string(), "client".to_string()),
            (AZURE_BROKER_TENANT_ID.to_string(), "tenant".to_string()),
            (AZURE_BROKER_ACCOUNT_NAME.to_string(), "account".to_string()),
            (AZURE_BROKER_REGION.to_string(), "westus3".to_string()),
            (AZURE_BROKER_BUCKET.to_string(), "container".to_string()),
            (
                AZURE_BROKER_DURATION_SECONDS.to_string(),
                "3600".to_string(),
            ),
            (
                AZURE_BROKER_REQUEST_TIMEOUT_MS.to_string(),
                "1000".to_string(),
            ),
            (
                "azure_storage_account_name".to_string(),
                "account".to_string(),
            ),
            (
                "adls.endpoint-suffix".to_string(),
                "core.windows.net".to_string(),
            ),
        ]);
        let extracted = AzureBrokerConfig::extract(&mut options).unwrap().unwrap();
        assert_eq!(extracted, config());
        assert!(BROKER_KEYS.iter().all(|key| !options.contains_key(*key)));
        assert_eq!(options["azure_storage_account_name"], "account");
        assert_eq!(options["adls.endpoint-suffix"], "core.windows.net");
    }

    #[test]
    fn rejects_partial_or_non_http_configuration() {
        let mut partial = HashMap::from([(
            AZURE_BROKER_ENDPOINT.to_string(),
            "http://credential-broker".to_string(),
        )]);
        assert!(AzureBrokerConfig::extract(&mut partial).is_err());

        let mut invalid = HashMap::from([
            (
                AZURE_BROKER_ENDPOINT.to_string(),
                "file:///tmp/token".to_string(),
            ),
            (AZURE_BROKER_CLIENT_ID.to_string(), "client".to_string()),
            (AZURE_BROKER_TENANT_ID.to_string(), "tenant".to_string()),
            (AZURE_BROKER_ACCOUNT_NAME.to_string(), "account".to_string()),
            (AZURE_BROKER_REGION.to_string(), "westus3".to_string()),
            (AZURE_BROKER_BUCKET.to_string(), "container".to_string()),
            (
                AZURE_BROKER_DURATION_SECONDS.to_string(),
                "3600".to_string(),
            ),
            (
                AZURE_BROKER_REQUEST_TIMEOUT_MS.to_string(),
                "1000".to_string(),
            ),
        ]);
        assert!(AzureBrokerConfig::extract(&mut invalid).is_err());
    }

    #[tokio::test]
    async fn rejects_sas_without_non_empty_signature() {
        for token in ["sv=1", "sv=1&sig="] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let now = Utc::now();
            let response_body = serde_json::json!({
                "success": true,
                "credentials": {
                    "tempAk": "account",
                    "sessionToken": token,
                    "expiredAt": (now + chrono::Duration::hours(1)).to_rfc3339(),
                }
            })
            .to_string();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = [0_u8; 4096];
                socket.read(&mut request).await.unwrap();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            });

            let mut broker_config = config();
            broker_config.endpoint = format!("http://{address}");
            let client = AzureBrokerClient::new(broker_config).unwrap();
            let error = match client.fetch(now).await {
                Ok(_) => panic!("expected SAS signature validation failure"),
                Err(error) => error,
            };
            assert_eq!(error.to_string(), "missing_sas_signature");
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn returns_refresh_error_with_cached_token() {
        let now = Arc::new(Mutex::new(Utc::now()));
        let initial_now = *now.lock().unwrap();
        let fetcher = Arc::new(MockFetcher::new(vec![
            Ok(AzureSasCredential {
                token: "sv=1&sig=old".to_string(),
                expires_at: initial_now + chrono::Duration::seconds(120),
            }),
            Err("http_status=500"),
            Err("http_status=500"),
            Ok(AzureSasCredential {
                token: "sv=2&sig=new".to_string(),
                expires_at: initial_now + chrono::Duration::hours(2),
            }),
        ]));
        let clock_now = now.clone();
        let provider = AzureSasStorageOptionsProvider::with_fetcher(
            config(),
            fetcher.clone(),
            Arc::new(move || *clock_now.lock().unwrap()),
        );

        let first = provider.current_credential().await.unwrap();
        assert_eq!(first.token, "sv=1&sig=old");

        *now.lock().unwrap() += chrono::Duration::seconds(61);
        let error = provider.current_credential().await.err().unwrap();
        assert_eq!(error.to_string(), "http_status=500");
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 2);

        *now.lock().unwrap() += chrono::Duration::seconds(60);
        assert!(*now.lock().unwrap() > initial_now + chrono::Duration::seconds(120));
        let error = provider.current_credential().await.err().unwrap();
        assert_eq!(error.to_string(), "http_status=500");
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 3);

        let refreshed = provider.current_credential().await.unwrap();
        assert_eq!(refreshed.token, "sv=2&sig=new");
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn fails_closed_without_cached_token() {
        let now = Utc::now();
        let fetcher = Arc::new(MockFetcher::new(vec![Err("transport_error")]));
        let provider =
            AzureSasStorageOptionsProvider::with_fetcher(config(), fetcher, Arc::new(move || now));
        assert!(provider.current_credential().await.is_err());
    }

    #[tokio::test]
    async fn current_credential_refresh_is_single_flight() {
        const CALLERS: usize = 100;

        let now = Utc::now();
        let fetch_started = Arc::new(Notify::new());
        let release_fetch = Arc::new(Notify::new());
        let fetcher = Arc::new(MockFetcher::with_gate(
            vec![Ok(AzureSasCredential {
                token: "sv=1&sig=single-flight".to_string(),
                expires_at: now + chrono::Duration::hours(1),
            })],
            fetch_started.clone(),
            release_fetch.clone(),
        ));
        let provider = Arc::new(AzureSasStorageOptionsProvider::with_fetcher(
            config(),
            fetcher.clone(),
            Arc::new(move || now),
        ));
        *provider.cached.write().await = Some(AzureSasCredential {
            token: "sv=1&sig=stale".to_string(),
            expires_at: now + chrono::Duration::seconds(1),
        });

        let start = Arc::new(Barrier::new(CALLERS + 1));
        let attempted = Arc::new(AtomicUsize::new(0));
        let all_callers_attempted = Arc::new(Notify::new());
        let tasks = (0..CALLERS)
            .map(|_| {
                let provider = provider.clone();
                let start = start.clone();
                let attempted = attempted.clone();
                let all_callers_attempted = all_callers_attempted.clone();
                tokio::spawn(async move {
                    start.wait().await;
                    if attempted.fetch_add(1, Ordering::SeqCst) + 1 == CALLERS {
                        all_callers_attempted.notify_one();
                    }
                    provider.current_credential().await.unwrap()
                })
            })
            .collect::<Vec<_>>();

        start.wait().await;
        all_callers_attempted.notified().await;
        fetch_started.notified().await;
        assert_eq!(attempted.load(Ordering::SeqCst), CALLERS);
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 1);
        release_fetch.notify_one();

        let credentials = join_all(tasks)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();

        assert!(
            credentials
                .iter()
                .all(|credential| credential.token == "sv=1&sig=single-flight")
        );
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn lance_provider_warms_once() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let server_requests = requests.clone();
        let response_body = serde_json::json!({
            "success": true,
            "credentials": {
                "tempAk": "account",
                "sessionToken": "sv=1&sig=warmed",
                "expiredAt": (Utc::now() + chrono::Duration::hours(1)).to_rfc3339(),
            }
        })
        .to_string();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            server_requests.fetch_add(1, Ordering::SeqCst);
            let mut request = [0_u8; 4096];
            socket.read(&mut request).await.unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let mut broker_config = config();
        broker_config.endpoint = format!("http://{address}");
        let provider = build_lance_provider(broker_config).await.unwrap();
        server.await.unwrap();
        assert_eq!(requests.load(Ordering::SeqCst), 1);

        let options = provider.fetch_storage_options().await.unwrap().unwrap();
        assert_eq!(options["azure_storage_sas_token"], "sv=1&sig=warmed");
        assert_eq!(requests.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn provider_id_is_hashed_and_contains_no_credential_material() {
        let id = config().provider_id();
        assert!(id.starts_with("azure_sas_broker_"));
        assert!(!id.contains("client"));
        assert!(!id.contains("tenant"));
    }
}
