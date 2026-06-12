// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Azure cross-tenant Managed Identity for `lance-io`'s `az` scheme.
//!
//! `object_store::azure::MicrosoftAzureBuilder` has a clean
//! `with_credentials(Arc<dyn CredentialProvider<Credential = AzureCredential>>)`
//! hook. We wire up an [`AzureCredential::BearerToken`] sourced from the
//! shared [`CrossTenantBearerCache`] (`azure_federation`), which performs
//! the IMDS → AAD two-hop exchange and refreshes ahead of expiry.
//!
//! Why a custom provider rather than relying on object_store's built-in
//! Azure auth paths:
//!
//! * **`WorkloadIdentityOAuthProvider`** wants a federated token *file*,
//!   which AKS workload-identity sets up via `AZURE_FEDERATED_TOKEN_FILE`.
//!   Our deployment is a plain Azure VM with a system-assigned Managed
//!   Identity — no token file exists.
//! * **`ImdsManagedIdentityProvider`** asks IMDS for a
//!   `https://storage.azure.com/` audience token in *our* tenant. The
//!   customer's storage account lives in a *different* tenant and rejects
//!   that token's issuer.
//! * **`ClientSecretOAuthProvider`** would need the customer to share a
//!   long-lived secret, which defeats the cross-tenant-without-secret point.
//!
//! Lance registers `az` to its built-in `AzureBlobStoreProvider` which uses
//! `with_url + with_config(k, v)` and never plumbs custom credentials. We
//! override `az` in a per-call `Session`'s `ObjectStoreRegistry` (see
//! `lance_bridgeimpl::pick_custom_session`) so only opens that opt in via
//! cross-tenant storage_options pick this provider up; everything else
//! continues to go through stock.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use object_store::azure::{AzureCredential, MicrosoftAzureBuilder};
use object_store::{
    CredentialProvider, ObjectStore as OSObjectStore, RetryConfig, Result as ObjectStoreResult,
};
use snafu::location;
use url::Url;

use lance::{Error as LanceError, Result as LanceResult};
use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, StorageOptions,
    DEFAULT_CLOUD_IO_PARALLELISM,
};

use crate::azure_federation::{into_object_store_err, CrossTenantBearerCache};

/// lance-io's `DEFAULT_CLOUD_BLOCK_SIZE` is crate-private; mirror its 64 KiB
/// value so opens through this provider behave the same as the stock Azure one.
const AZURE_DEFAULT_BLOCK_SIZE: usize = 64 * 1024;

/// `object_store::CredentialProvider` returning a customer-tenant Bearer
/// minted via the shared cache. `get_credential` is a hot path
/// (object_store calls it on every outbound request); cache hits return in a
/// single read-lock acquisition, so the steady-state cost is negligible.
#[derive(Debug)]
pub struct CrossTenantAzureCredentialProvider {
    cache: Arc<CrossTenantBearerCache>,
}

impl CrossTenantAzureCredentialProvider {
    pub fn new(cache: Arc<CrossTenantBearerCache>) -> Self {
        Self { cache }
    }
}

#[async_trait]
impl CredentialProvider for CrossTenantAzureCredentialProvider {
    type Credential = AzureCredential;

    async fn get_credential(&self) -> ObjectStoreResult<Arc<Self::Credential>> {
        let bearer = self
            .cache
            .current()
            .await
            .map_err(into_object_store_err)?;
        // `(*bearer).clone()` materializes the inner `String` for the
        // BearerToken variant — `Arc<String>` itself can't be moved into
        // `BearerToken(String)` without dereferencing.
        Ok(Arc::new(AzureCredential::BearerToken((*bearer).clone())))
    }
}

/// Lance `ObjectStoreProvider` for `az://` opens that should use cross-tenant
/// MI. Wires a [`CrossTenantAzureCredentialProvider`] into the standard
/// `MicrosoftAzureBuilder`, and forwards every non-credential storage option
/// (account name, retry knobs, etc.) so end users can still tune behaviour.
#[derive(Debug)]
pub struct CrossTenantAzureStoreProvider {
    cache: Arc<CrossTenantBearerCache>,
}

impl CrossTenantAzureStoreProvider {
    pub fn new(cache: Arc<CrossTenantBearerCache>) -> Self {
        Self { cache }
    }
}

/// Storage-option keys that would compete with our credential provider if
/// passed through to `MicrosoftAzureBuilder` (and silently win, since the
/// builder's auth selection runs independently of `with_credentials`).
/// We strip them defensively even though the C++ side
/// (`lance_common.cpp` Azure cross-tenant branch) already declines to emit
/// them — a property file or env var sweep could still leak one in.
const CONFLICTING_AZURE_KEYS: &[&str] = &[
    "azure_storage_account_key",
    "azure_account_key",
    "account_key",
    "azure_storage_sas_token",
    "azure_storage_sas_key",
    "sas_token",
    "sas_key",
    "azure_client_secret",
    "client_secret",
    "azure_storage_token",
    "azure_bearer_token",
    "bearer_token",
];

#[async_trait]
impl ObjectStoreProvider for CrossTenantAzureStoreProvider {
    async fn new_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> LanceResult<ObjectStore> {
        let block_size = params.block_size.unwrap_or(AZURE_DEFAULT_BLOCK_SIZE);

        // Pre-filter storage_options before lance-io's `StorageOptions` env
        // sweep runs, so a stray account_key in the process env can't shadow
        // our credentials.
        let raw_options = params.storage_options.clone().unwrap_or_default();
        let filtered: HashMap<String, String> = raw_options
            .into_iter()
            .filter(|(k, _)| {
                let lower = k.to_ascii_lowercase();
                !CONFLICTING_AZURE_KEYS.contains(&lower.as_str())
            })
            .collect();
        let mut storage_options = StorageOptions(filtered);
        storage_options.with_env_azure();
        let download_retry_count = storage_options.download_retry_count();
        let max_retries = storage_options.client_max_retries();
        let retry_timeout = storage_options.client_retry_timeout();

        let retry_config = RetryConfig {
            backoff: Default::default(),
            max_retries,
            retry_timeout: Duration::from_secs(retry_timeout),
        };

        let mut builder = MicrosoftAzureBuilder::new()
            .with_url(base_path.as_ref())
            .with_retry(retry_config);
        // Forward Azure-recognized config keys (account name, endpoint, etc.).
        // `as_azure_options()` filters via `AzureConfigKey::from_str`, so it
        // naturally drops our bridge-private `azure_cross_tenant_*` keys
        // (those don't parse as known Azure config keys).
        for (key, value) in storage_options.as_azure_options() {
            // Defense-in-depth: even though we filtered raw keys above,
            // AzureConfigKey enumerates more than CONFLICTING_AZURE_KEYS
            // covers (FabricToken*, etc.). Skip the credential-bearing keys
            // a second time at the AzureConfigKey enum level.
            if matches!(
                key,
                object_store::azure::AzureConfigKey::AccessKey
                    | object_store::azure::AzureConfigKey::SasKey
                    | object_store::azure::AzureConfigKey::ClientSecret
                    | object_store::azure::AzureConfigKey::Token
                    | object_store::azure::AzureConfigKey::FederatedTokenFile
            ) {
                continue;
            }
            builder = builder.with_config(key, value);
        }

        // Plug our credential provider in last so it wins over anything
        // `with_url` / `with_config` may have inferred.
        let credential_provider: Arc<dyn CredentialProvider<Credential = AzureCredential>> =
            Arc::new(CrossTenantAzureCredentialProvider::new(self.cache.clone()));
        builder = builder.with_credentials(credential_provider);

        let store = builder.build().map_err(|e| LanceError::IO {
            source: Box::new(e),
            location: location!(),
        })?;
        let inner = Arc::new(store) as Arc<dyn OSObjectStore>;

        Ok(ObjectStore::new(
            inner,
            base_path,
            Some(block_size),
            params.object_store_wrapper.clone(),
            params.use_constant_size_upload_parts,
            // Azure list is lexically ordered (matches stock AzureBlobStoreProvider).
            true,
            DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count,
            params.storage_options.as_ref(),
        ))
    }
}
