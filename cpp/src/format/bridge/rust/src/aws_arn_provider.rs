use std::sync::Arc;
use std::time::{Duration, SystemTime};

use aws_config::sts::AssumeRoleProvider;
use aws_credential_types::provider::ProvideCredentials;
use iceberg::io::StorageFactory;
use iceberg_storage_opendal::{
    AwsCredential, AwsCredentialLoad, CustomAwsCredentialLoader, OpenDalStorageFactory,
};
use lance::session::Session;
use lance::{Error as LanceError, Result as LanceResult};
use lance_io::object_store::providers::aws::AwsStoreProvider;
use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, ObjectStoreRegistry,
};
use object_store::CredentialProvider;
use tokio::sync::{Mutex, RwLock};
use url::Url;

#[derive(Debug)]
struct SingleFlightAwsCredentialProvider {
    inner: Arc<dyn ProvideCredentials>,
    cached: RwLock<Option<aws_credential_types::Credentials>>,
    refresh_lock: Mutex<()>,
    refresh_offset: Duration,
}

impl SingleFlightAwsCredentialProvider {
    fn new(inner: Arc<dyn ProvideCredentials>, refresh_offset: Duration) -> Self {
        Self {
            inner,
            cached: RwLock::new(None),
            refresh_lock: Mutex::new(()),
            refresh_offset,
        }
    }

    fn current_credential(
        &self,
        credentials: Option<&aws_credential_types::Credentials>,
    ) -> Option<Arc<object_store::aws::AwsCredential>> {
        let credentials = credentials?;
        let fresh = credentials
            .expiry()
            .map(|expires_at| {
                expires_at
                    .duration_since(SystemTime::now())
                    .is_ok_and(|remaining| remaining > self.refresh_offset)
            })
            .unwrap_or(true);
        fresh.then(|| Self::to_object_store_credential(credentials))
    }

    fn to_object_store_credential(
        credentials: &aws_credential_types::Credentials,
    ) -> Arc<object_store::aws::AwsCredential> {
        Arc::new(object_store::aws::AwsCredential {
            key_id: credentials.access_key_id().to_string(),
            secret_key: credentials.secret_access_key().to_string(),
            token: credentials.session_token().map(ToString::to_string),
        })
    }
}

#[async_trait::async_trait]
impl CredentialProvider for SingleFlightAwsCredentialProvider {
    type Credential = object_store::aws::AwsCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
        {
            let cached = self.cached.read().await;
            if let Some(credential) = self.current_credential(cached.as_ref()) {
                return Ok(credential);
            }
        }

        let _refresh = self.refresh_lock.lock().await;
        {
            let cached = self.cached.read().await;
            if let Some(credential) = self.current_credential(cached.as_ref()) {
                return Ok(credential);
            }
        }

        let refreshed = self
            .inner
            .provide_credentials()
            .await
            .map_err(|source| object_store::Error::Generic {
                store: "AWS",
                source: Box::new(source),
            })?;
        let credential = Self::to_object_store_credential(&refreshed);
        *self.cached.write().await = Some(refreshed);
        Ok(credential)
    }
}

pub(crate) struct AssumeRoleConfig {
    role_arn: String,
    session_name: String,
    external_id: String,
    region: String,
    credential_refresh_secs: u64,
}

impl AssumeRoleConfig {
    pub(crate) fn parse(
        role_arn: &str,
        session_name: &str,
        external_id: &str,
        region: &str,
        credential_refresh_secs: u64,
    ) -> LanceResult<Option<Self>> {
        if role_arn.is_empty() {
            return Ok(None);
        }
        if credential_refresh_secs < 900 || credential_refresh_secs > 43200 {
            return Err(LanceError::invalid_input(
                format!(
                    "credential_refresh_secs must be in [900, 43200], got {}",
                    credential_refresh_secs
                ),
            ));
        }
        Ok(Some(Self {
            role_arn: role_arn.to_string(),
            session_name: session_name.to_string(),
            external_id: external_id.to_string(),
            region: region.to_string(),
            credential_refresh_secs,
        }))
    }

    async fn build_credentials(&self) -> LanceResult<object_store::aws::AwsCredentialProvider> {
        const REFRESH_OFFSET_SECS: u64 = 300;

        let mut builder = AssumeRoleProvider::builder(&self.role_arn)
            .session_length(Duration::from_secs(self.credential_refresh_secs));

        if !self.session_name.is_empty() {
            builder = builder.session_name(&self.session_name);
        }
        if !self.external_id.is_empty() {
            builder = builder.external_id(&self.external_id);
        }
        if !self.region.is_empty() {
            builder = builder.region(aws_config::Region::new(self.region.clone()));
        }

        let assume_role_provider = builder.build().await;
        let provider: object_store::aws::AwsCredentialProvider =
            Arc::new(SingleFlightAwsCredentialProvider::new(
                Arc::new(assume_role_provider),
                Duration::from_secs(REFRESH_OFFSET_SECS),
            ));
        provider
            .get_credential()
            .await
            .map_err(|error| LanceError::IO {
                source: Box::new(error),
                location: snafu::location!(),
            })?;
        Ok(provider)
    }
}

#[derive(Debug)]
struct AwsArnStoreProvider {
    credentials: object_store::aws::AwsCredentialProvider,
}

impl AwsArnStoreProvider {
    fn new(credentials: object_store::aws::AwsCredentialProvider) -> Self {
        Self { credentials }
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for AwsArnStoreProvider {
    async fn new_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> LanceResult<ObjectStore> {
        let mut params = params.clone();
        params.aws_credentials = Some(self.credentials.clone());
        AwsStoreProvider::default()
            .new_store(base_path, &params)
            .await
    }
}

#[derive(Clone)]
struct ObjectStoreAwsCredentialLoader {
    provider: object_store::aws::AwsCredentialProvider,
}

#[async_trait::async_trait]
impl AwsCredentialLoad for ObjectStoreAwsCredentialLoader {
    async fn load_credential(
        &self,
        _client: reqwest::Client,
    ) -> anyhow::Result<Option<AwsCredential>> {
        let credential = self
            .provider
            .get_credential()
            .await
            .map_err(|error| anyhow::anyhow!("AWS credential provider failed: {error}"))?;
        Ok(Some(AwsCredential {
            access_key_id: credential.key_id.clone(),
            secret_access_key: credential.secret_key.clone(),
            session_token: credential.token.clone(),
            expires_in: None,
        }))
    }
}

pub(crate) async fn build_lance_provider(
    config: &AssumeRoleConfig,
) -> LanceResult<Arc<dyn ObjectStoreProvider>> {
    let credentials = config.build_credentials().await?;
    Ok(Arc::new(AwsArnStoreProvider::new(credentials)))
}

pub(crate) fn build_lance_session(provider: Arc<dyn ObjectStoreProvider>) -> Arc<Session> {
    let registry = ObjectStoreRegistry::default();
    registry.insert("s3", provider.clone());
    registry.insert("s3+ddb", provider);
    Arc::new(Session::new(0, 0, Arc::new(registry)))
}

fn iceberg_factory(
    scheme: &str,
    provider: object_store::aws::AwsCredentialProvider,
) -> Arc<dyn StorageFactory> {
    Arc::new(OpenDalStorageFactory::S3 {
        configured_scheme: scheme.to_string(),
        customized_credential_load: Some(CustomAwsCredentialLoader::new(Arc::new(
            ObjectStoreAwsCredentialLoader { provider },
        ))),
    })
}

pub(crate) async fn build_iceberg_factory(
    scheme: &str,
    config: &AssumeRoleConfig,
) -> LanceResult<Arc<dyn StorageFactory>> {
    Ok(iceberg_factory(scheme, config.build_credentials().await?))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::future::join_all;
    use object_store::StaticCredentialProvider;

    use crate::cloud_provider_cache::GlobalLruCache;

    use super::*;

    fn static_credentials(key_id: &str) -> object_store::aws::AwsCredentialProvider {
        Arc::new(StaticCredentialProvider::new(
            object_store::aws::AwsCredential {
                key_id: key_id.to_string(),
                secret_key: "secret".to_string(),
                token: None,
            },
        ))
    }

    #[derive(Debug)]
    struct CountingCredentialsProvider {
        calls: Arc<AtomicUsize>,
    }

    impl ProvideCredentials for CountingCredentialsProvider {
        fn provide_credentials<'a>(
            &'a self,
        ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            let calls = self.calls.clone();
            aws_credential_types::provider::future::ProvideCredentials::new(async move {
                let call = calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                Ok(aws_credential_types::Credentials::new(
                    format!("key-{call}"),
                    "secret",
                    None,
                    Some(SystemTime::now() + Duration::from_secs(3600)),
                    "test",
                ))
            })
        }
    }

    #[test]
    fn assume_role_config_preserves_region() {
        let config = AssumeRoleConfig::parse(
            "arn:aws:iam::123456789012:role/test",
            "session",
            "external-id",
            "cn-north-1",
            900,
        )
        .unwrap()
        .unwrap();

        assert_eq!(config.region, "cn-north-1");
    }

    #[tokio::test]
    async fn credential_refresh_is_single_flight() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(SingleFlightAwsCredentialProvider::new(
            Arc::new(CountingCredentialsProvider {
                calls: calls.clone(),
            }),
            Duration::from_secs(300),
        ));
        *provider.cached.write().await = Some(aws_credential_types::Credentials::new(
            "stale",
            "secret",
            None,
            Some(SystemTime::now() + Duration::from_secs(1)),
            "test",
        ));

        let credentials = join_all((0..100).map(|_| {
            let provider = provider.clone();
            async move { provider.get_credential().await.unwrap() }
        }))
        .await;

        assert!(credentials.iter().all(|credential| credential.key_id == "key-0"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn lance_provider_is_reused() {
        let cache: GlobalLruCache<Arc<dyn ObjectStoreProvider>> = GlobalLruCache::new(2);
        let first = cache
            .get("aws", || async {
                Ok::<_, ()>(Arc::new(AwsArnStoreProvider::new(static_credentials(
                    "first",
                ))) as Arc<dyn ObjectStoreProvider>)
            })
            .await
            .unwrap();
        let second = cache
            .get("aws", || async {
                Ok::<_, ()>(Arc::new(AwsArnStoreProvider::new(static_credentials(
                    "second",
                ))) as Arc<dyn ObjectStoreProvider>)
            })
            .await
            .unwrap();

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[tokio::test]
    async fn iceberg_factory_is_reused() {
        let cache: GlobalLruCache<Arc<dyn StorageFactory>> = GlobalLruCache::new(2);
        let first = cache
            .get("aws", || async {
                Ok::<_, ()>(iceberg_factory("s3", static_credentials("first")))
            })
            .await
            .unwrap();
        let second = cache
            .get("aws", || async {
                Ok::<_, ()>(iceberg_factory("s3", static_credentials("second")))
            })
            .await
            .unwrap();

        assert!(Arc::ptr_eq(&first, &second));
    }
}
