# GCP Cross-Tenant Access via Service Account Impersonation — Design Doc

**Status:** Landed (commit `4e3a7e9`)
**Author:** jiaqizho
**Date:** 2026-04-21
**Supersedes the working notes in:**
- `docs/cross-tenant-cloud-access.md` (cross-cloud survey)
- `docs/gcp-multi-identity-design-cn.md` (C++ multi-identity refactor)
- `docs/iceberg-gcp-impersonation-analysis.md` (iceberg two-path analysis)

---

## 1. Background

In the External Table scenario the service reads data objects owned by
customer GCP projects. Neither the customer's HMAC keys nor their JSON
service-account keys are acceptable to hand over; the industry-standard answer
across the three majors is **temporary, token-based cross-tenant access**
driven entirely by pre-configured customer-side authorization:

- **AWS** — STS `AssumeRole`, keyed on a role ARN.
- **GCP** — **Service Account Impersonation**, keyed on a target service
  account email. Our VM's default SA calls
  `iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/<target>:generateAccessToken`
  and receives a short-lived OAuth2 bearer for the target SA. The customer
  grants our VM SA `roles/iam.serviceAccountTokenCreator` on the target SA;
  no keys cross the tenant boundary.
- **Azure** — Multi-Tenant App + Federated Credential (not implemented here).

AWS AssumeRole is already supported end-to-end in milvus-storage. This commit
adds the analogous GCP path. Azure is a separate workstream, intentionally out
of scope.

## 2. Goals

Support GCP target-SA impersonation for External Table reads:

- In **both** read paths of both formats:

  | Format  | Metadata / plan path | Data read path |
  |---------|----------------------|----------------|
  | Lance   | Rust `lance-io` (object_store GCS) | Rust `lance-io` (same) |
  | Iceberg | Rust `iceberg-rust` (opendal GCS) | **C++** `ParquetFormatReader` over Arrow S3 against GCS S3-compat |

- Allow **multiple GCP identities in a single process** — distinct `fs.*` and
  `extfs.<ns>.*` slots can each use a different credential mode (VM default
  SA, a target SA, or HMAC AK/SK).
- Customer-facing surface area is one new property:
  `fs.gcp_target_service_account`. `fs.load_frequency` is reused as the
  requested token lifetime.

## 3. Non-goals

- **Azure cross-tenant.** Planned as a follow-up; see
  `cross-tenant-cloud-access.md` §6.1 for the target shape.
- **AWS changes.** AssumeRole already works; we do not touch it.
- **Write-side impersonation.** The E2E test writes test data with HMAC
  over S3-compat; production writer paths still use whatever credential the
  producer already supported. A writer-side impersonation feature would be a
  separate design.
- **Upstream iceberg-rust PR** to add a proper `CustomGcsTokenLoader`. We work
  around the missing hook in our bridge; the upstream fix is out of scope
  here.
- **Global TLS floor / log level.** Preserved as pre-refactor "first-Make
  wins" behavior; treated as a separate problem.
- **URI syntax / extfs matching logic.** Unchanged.

## 4. Architecture overview

A single customer-supplied identifier (`gcp_target_service_account`) flows
through three independent runtime paths, one per language × format × role:

```
Property: fs.gcp_target_service_account
   └── ArrowFileSystemConfig.gcp_target_service_account
         ├── (C++ FS producer) ──┐
         │                        ├─ Iceberg:  iceberg_common.cpp  → gcs.service-account
         ├── (Lance format)   ───┤              (bridge-private; intercepted by Rust)
         │                        └─ Lance:     lance_common.cpp   → gcp_target_service_account
         │                                      + gcp_credential_refresh_secs
         │                                      (bridge-private; intercepted by Rust)
         │
         └── Three runtime paths:
              (a) C++ parquet data reads against GCS S3-compat
                  → GcpFileSystemProducer
                  → GcpCredentialRegistry
                  → IamImpersonateProvider
              (b) Rust Lance (open_dataset / write_dataset)
                  → lance_bridgeimpl extracts bridge-private keys
                  → Session with ObjectStoreRegistry override for "gs"
                  → ImpersonatingGcsStoreProvider
                  → ImpersonatingGcsCredentialProvider (cached, refreshable)
              (c) Rust Iceberg (plan_files)
                  → iceberg_bridgeimpl intercepts "gcs.service-account"
                  → fetch_impersonated_bearer (one-shot, no cache)
                  → swap for "gcs.oauth2.token" before FileIOBuilder::build()
```

Each path performs the same underlying exchange:
`VM default SA (metadata.google.internal)` →
`iamcredentials.generateAccessToken(target_sa)` → impersonated bearer. The
implementations differ because each library's extension surface differs and
because the token-lifetime requirements differ between long-lived data scans
and transient metadata reads (see §7.3 and §8.3).

## 5. Properties and configuration

### 5.1 New property

```c++
#define PROPERTY_FS_GCP_TARGET_SERVICE_ACCOUNT "fs.gcp_target_service_account"
```

Registered as a standard string property (default `""`, no validator).
Available in both `fs.<key>` and `extfs.<ns>.<key>` forms.

### 5.2 Config field

```c++
// cpp/include/milvus-storage/filesystem/fs.h
struct ArrowFileSystemConfig {
  ...
  // Target service account email for impersonation (empty = no impersonation).
  std::string gcp_target_service_account = "";

  // Cross-provider token-lifetime / refresh-interval knob, in seconds.
  //   AWS STS AssumeRole: session length, STS clamps to [900, 43200]
  //   GCP IAM impersonation: generateAccessToken lifetime, IAM caps at 3600
  //                          unless org policy raises the cap
  // Lance/iceberg readers refresh ahead of expiry using this value as TTL.
  int32_t load_frequency = 900;
  ...
};
```

`load_frequency` is reused across providers. For GCP impersonation it is
passed through to `AccessTokenLifetimeOption` on the C++ side and to
`gcp_credential_refresh_secs` on the Rust side; providers that do not mint
temporary credentials ignore it.

### 5.3 Cache-key invariant

`GetCacheKey()` remains `"{address}/{bucket_name}"`. Identity fields are
deliberately omitted because a single `(address, bucket)` pair **cannot
legally be accessed by two different credentials in the same process**
(GCS permissions live at bucket granularity). Two configs resolving to the
same cache key therefore carry the same identity, and sharing the cached
filesystem instance is safe. The comment in `fs.h` states this invariant
explicitly because it is load-bearing for both filesystem caching and the
`GcpCredentialRegistry` "idempotent register, no conflict check" design
(§6.3).

## 6. C++ side: multi-identity GCP filesystem producer

### 6.1 The problem

Baseline `GcpFileSystemProducer::InitS3Compat` (commit `b5f8eef`, the
"move-out-gcp-producer" change) captured `use_iam`, `access_key_id`,
`access_key_value`, `tls_min_version` inside a `std::call_once` lambda and
baked **one** identity into `GoogleHttpClientFactory`. That factory was then
installed process-globally by `Aws::InitializeS3`. Result: the first GCP
`Make()` in a process permanently determines the identity used for every
subsequent GCP request — impossible to have `fs.*` and an `extfs.<ns>.*`
slot use different credentials at the same time.

S3 has no equivalent problem: its `TlsHttpClientFactory` is credential-free
and each `S3Client` carries its own credential chain. The issue is GCP-only
because GCP hides two credential-aware responsibilities inside the HTTP
factory: OAuth2 `Authorization: Bearer` injection, and GOOG4-HMAC-SHA256
re-signing of conditional writes (which GCS refuses to accept in SigV4 form).

### 6.2 Design: stateless factory + per-request URI routing

The fix is to take the identity out of the factory. The factory becomes a
thin URI dispatcher; a per-process registry maps `(endpoint_host, bucket_name)`
to a `GcpCredentialProvider`; each outbound request looks up its own provider.

```
┌─ GcpCredentialProvider (interface) ──────────────────────────────┐
│   AuthorizationHeader()  → optional<{name, value}>               │
│     IAM/Impersonate → Bearer; HMAC → nullopt                     │
│   MaybeSignConditionalWrite(request) → Status                    │
│     HMAC → GOOG4 re-sign; IAM/Impersonate → OK                   │
│                                                                    │
│   Implementations (all internal to the .cpp):                     │
│     IamVmProvider         (MakeGoogleDefaultCredentials)          │
│     IamImpersonateProvider(MakeImpersonateServiceAccount)         │
│     HmacProvider          (AKSK + auth_signer::googv4::SignRequest)│
└──────────────────┬───────────────────────────────────────────────┘
                   │ register / lookup by (endpoint_host, bucket_name)
                   ▼
┌─ GcpCredentialRegistry (process singleton, mutex-guarded) ────────┐
│   unordered_map<GcpBucketKey, shared_ptr<Provider>>                │
│   Register(key, provider)   // idempotent silent replace           │
│   Lookup(request_uri)       // tries path-style, then vhost-style  │
└──────────────────┬───────────────────────────────────────────────┘
                   ▼
┌─ GoogleHttpClientFactory (stateless — only holds tls_min_version) ┐
│   CreateHttpRequest(uri, …):                                       │
│     req = StandardHttpRequest{…}                                   │
│     if provider = Registry::Lookup(uri):                           │
│       if header = provider->AuthorizationHeader():                 │
│         req.SetHeader(header.first, header.second)                 │
│     return req                                                      │
│                                                                      │
│   CreateHttpClient(cfg):                                            │
│     return new GoogleHttpClientDelegator(cfg, tls_min_version)     │
└──────────────────┬───────────────────────────────────────────────┘
                   ▼
┌─ GoogleHttpClientDelegator (stateless) ──────────────────────────┐
│   MakeRequest(req):                                                │
│     provider = Registry::Lookup(req.uri)                           │
│     if !provider: return 403 SignatureFailed XML                   │
│     if !provider->MaybeSignConditionalWrite(req): return 403 XML   │
│     return underlying_.MakeRequest(req)                            │
└────────────────────────────────────────────────────────────────────┘
```

### 6.3 GcpCredentialRegistry

- **Key.** `GcpBucketKey{endpoint_host, bucket_name}`. `endpoint_host` is the
  bare host (scheme stripped, trailing `/` trimmed, lower-cased);
  `NormalizeGcpEndpointHost` does the normalization at register time, and
  `Lookup` lower-cases the request's authority before comparison.

- **Dual-interpretation lookup.** GCS S3-compat supports both URL styles; the
  lookup tries them in order:

  | Style            | Host                               | Path            | Resolves to              |
  |------------------|------------------------------------|-----------------|--------------------------|
  | Path-style       | `storage.googleapis.com`           | `/bucket/key`   | `{host, first_segment}`  |
  | Virtual-host     | `bucket.storage.googleapis.com`    | `/key`          | `{rest, subdomain}`      |

  The first hit wins. This mirrors exactly how the AWS SDK emits URIs under
  the two addressing modes.

- **Idempotent Register.** A second `Register` with the same key silently
  replaces the prior provider. This is correct because of the cache-key
  invariant (§5.3) — same `(endpoint, bucket)` always means same identity.
  The prior "conflict detection" draft was explicitly dropped as defensive
  code against a configuration that cannot happen.

### 6.4 Three provider implementations

All three live in the `.cpp` file; only `GcpCredentialProvider` and
`BuildGcpProviderFromConfig` are exposed in the header.

- **`IamVmProvider`** — VM / ADC identity. Built from
  `google::cloud::MakeGoogleDefaultCredentials()`, wrapped via
  `google::cloud::rest_internal::MapCredentials` into an
  `oauth2_internal::Credentials`. `AuthorizationHeader()` calls the
  google-cloud-cpp helper of the same name, which handles caching and refresh
  internally.

- **`IamImpersonateProvider`** — built from
  `google::cloud::MakeImpersonateServiceAccountCredentials(base,
  target_sa, opts)`. `load_frequency` becomes
  `AccessTokenLifetimeOption`, clamped to `kMaxImpersonationTokenLifetime =
  3600s` because GCP IAM rejects longer lifetimes without an org-policy
  override. Token caching/refresh are handled by the google-cloud-cpp
  internal credential wrapper — no bespoke cache needed. This is a **cleaner
  outcome than the "build a custom cache around `ComputeEngineCredentials` +
  REST POST to `iamcredentials`" plan in
  `iceberg-gcp-impersonation-analysis.md`** — the multi-identity refactor
  made it possible to lean on the public google-cloud-cpp API.

- **`HmacProvider`** — `AuthorizationHeader()` returns `nullopt` (AWS SDK's
  SigV4 signer runs as usual for non-conditional requests). For conditional
  writes (`x-goog-if-generation-match`), `MaybeSignConditionalWrite`:
  1. Deletes AWS SigV4 headers (`Authorization`, `x-amz-date`,
     `x-amz-content-sha256`, `x-amz-security-token`, `x-amz-api-version`).
  2. Renames `x-amz-meta-*` headers to `x-goog-meta-*` (case-insensitive key
     scan, stable rewrite).
  3. Calls `auth_signer::googv4::SignRequest(request, access_key, secret_key)`.
  This code used to live inline in `GoogleHttpClientDelegator` — it moved
  verbatim into `HmacProvider` as part of the refactor, no behavior change.

`BuildGcpProviderFromConfig` dispatches:
`use_iam && !target_sa.empty()` → impersonation; `use_iam` → VM/ADC;
both AK and SK set → HMAC; otherwise a detailed `Invalid` status naming the
three mutually exclusive valid combinations.

### 6.5 Make() flow

```cpp
arrow::Result<ArrowFileSystemPtr> GcpFileSystemProducer::Make() {
  ARROW_RETURN_NOT_OK(InitS3Compat(config_));     // call_once: stateless factory + InitializeS3
  ARROW_RETURN_NOT_OK(RegisterIdentity(config_)); // per-call: build + register provider
  ARROW_ASSIGN_OR_RAISE(auto s3_options, CreateS3Options());
  ARROW_ASSIGN_OR_RAISE(auto fs, S3FileSystem::Make(s3_options));
  return std::make_shared<FileSystemProxy>(config_.bucket_name, fs);
}
```

`call_once` stays because `Aws::InitializeS3` can only be called once per
process — that is an AWS SDK hard constraint, not something our refactor is
trying to relax. But identity is no longer frozen by it: it now lives in the
registry, which `RegisterIdentity` writes to on every `Make()`.

The first-Make-wins settings under `call_once` are now narrowed to TLS min
version and log level. This is deliberately not fixed in this commit; a
global-TLS-floor policy is a separate design.

### 6.6 Missing-lookup = fail fast

Both the factory (`CreateHttpRequest`) and the delegator (`MakeRequest`) treat
a registry miss as a bug:

- Factory logs an error and lets `MakeRequest` terminate the request.
- `MakeRequest` fabricates an AWS-compatible XML error body:

  ```xml
  <Error>
    <Code>SignatureFailed</Code>
    <Message>No GcpCredentialProvider registered for URI: …</Message>
  </Error>
  ```
  This shape is picked so the AWS SDK's `ErrorMarshaller` extracts the
  message cleanly and the upstream Arrow caller sees a usable diagnostic.

Invariant: every GCP URI reaching the HTTP stack must have been registered
by `RegisterIdentity` first. A miss indicates a URI-normalization bug in the
registry or a request path that bypassed `Make()`, both of which should
surface loudly instead of silently hitting the server with an unsigned
request and triggering a puzzling 403.

### 6.7 Switch to public google-cloud-cpp APIs

Baseline used `google::cloud::oauth2_internal::ComputeEngineCredentials`
directly and manually constructed a REST client. This commit switches to:

- `google::cloud::MakeGoogleDefaultCredentials(opts)` — VM / ADC path.
- `google::cloud::MakeImpersonateServiceAccountCredentials(base, target_sa, opts)` — impersonation.
- `google::cloud::rest_internal::MapCredentials(*public)` — bridge to the
  internal `oauth2_internal::Credentials` type that
  `google::cloud::oauth2_internal::AuthorizationHeader` accepts.

Direct includes of `oauth2_internal/oauth2_google_credentials.h`,
`rest_client.h`, `storage/oauth2/compute_engine_credentials.h` are removed;
only the `AuthorizationHeader` helper and the `MapCredentials` bridge remain
as internal-namespace dependencies, and those are one step away from being
public in upstream google-cloud-cpp.

## 7. Rust side: impersonation for Lance and Iceberg

### 7.1 Why no upstream key path works

- **`object_store::gcp`** (used by lance-io) hand-rolls its own GCP auth;
  all token-fetch code is crate-private. The public builder keys are
  `google_service_account` / `_path` / `_key` / `google_application_credentials`
  / `google_storage_token` — none accept "VM-SA impersonating target-SA".
- **`reqsign::google`** (used by opendal, used by iceberg-rust) has a
  `GoogleTokenLoader` that reads the metadata server, but
  `GoogleToken::access_token` is `pub(crate)` by design (upstream comment:
  *"don't allow get token from reqsign"*) — we cannot extract the bearer to
  reuse it. Its `ImpersonatedServiceAccount` JSON variant is the
  authorized-user (refresh-token) flow, not VM-SA source.
- **reqsign's `external_account`** is Workload Identity Federation for non-GCP
  identity providers — STS will not federate a Google-issued token back into
  Google.
- **iceberg-rust** could in principle accept a custom token loader via the
  opendal `GcsBuilder::customized_token_loader` hook, but iceberg-rust builds
  operators with `Operator::from_config(cfg).finish()` and has no callback
  surface for GCS (the S3 analogue `CustomAwsCredentialLoader` exists, the
  GCS one does not).

So the business logic (metadata server call + IAM REST POST) has to live
locally. Interface types (`CredentialProvider`, `GcpCredential`,
`GoogleCloudStorageBuilder`, `ObjectStoreProvider`) are reused as-is.

### 7.2 Shared core

`cpp/src/format/bridge/rust/src/gcp_impersonation.rs` centralizes the two
HTTPS calls:

- `GET http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token`
  (with `Metadata-Flavor: Google`).
- `POST https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/<target>:generateAccessToken`
  (body: `{"scope": ["https://www.googleapis.com/auth/cloud-platform"],
  "lifetime": "<secs>s"}`).

Both are invoked by `fetch_impersonated_access_token`, which returns the raw
`accessToken` + RFC3339 `expireTime`. Two callers use it:

- `fetch_impersonated_bearer(target_sa, lifetime)` — one-shot, no cache,
  returns just the bearer string. Consumed by the iceberg bridge.
- `ImpersonatingGcsCredentialProvider` — cached, refreshable.

Pulling in `google-cloud-iam-credentials-v1` would remove the hand-rolled
URLs but drag in the full gRPC stack (`tonic` / `prost`) for two JSON POSTs;
not a good tradeoff.

### 7.3 Lance — refreshable credential provider

Lance dataset opens can hold readers for the entire scan lifetime, which
routinely exceeds the 1-hour IAM token cap. A one-shot bearer is not enough;
we need `CredentialProvider::get_credential()` to transparently refresh
ahead of expiry.

**`ImpersonatingGcsCredentialProvider`** implements
`object_store::CredentialProvider<Credential = GcpCredential>` with:

- A single cached token behind an async `RwLock`.
- Fast path: read-lock, return cached credential if
  `now + REFRESH_OFFSET_SECS < expires_at`.
- Refresh path: `try_write` — on miss, return `None` so the caller backs off
  10ms and retries (keeps concurrent refreshes from stampeding the IAM
  endpoint). Pattern mirrors lance-io's AWS-side
  `DynamicStorageOptionsCredentialProvider`.
- Double-check after acquiring the write lock: another task may have just
  refreshed.
- `REFRESH_OFFSET_SECS = 300` — matches the AWS path.
- **Expiry is read from IAM's `expireTime`, not `now + lifetime`.** Clock
  skew between us and Google's auth servers cannot push us into
  stale-but-thinks-fresh.

**`ImpersonatingGcsStoreProvider`** implements
`lance_io::object_store::ObjectStoreProvider` for the `gs` scheme. It:

- Forwards non-credential config keys from `storage_options` to
  `GoogleCloudStorageBuilder` (endpoint overrides, retry knobs, etc.).
- **Explicitly filters out** credential keys (`google_service_account`,
  `google_storage_token`, etc.) so they cannot race with our
  provider-supplied credential.
- Installs the credential provider via `with_credentials`.
- Mirrors lance-io's otherwise-crate-private defaults: block size 64 KiB,
  3 download retries, `DEFAULT_CLOUD_IO_PARALLELISM`, lexically ordered
  listing.

### 7.4 Wiring into lance-io via a per-call Session

Instead of replacing lance-io's global registry, `lance_bridgeimpl`:

1. `GcpImpersonationConfig::extract` pulls bridge-private keys
   `gcp_target_service_account` and `gcp_credential_refresh_secs` out of the
   `storage_options` map (consuming them with `remove`, so they cannot leak
   onward to lance-io / object_store where they are unrecognized).
2. `build_gcp_impersonation_session(config)` creates a fresh
   `Session::new(0, 0, ObjectStoreRegistry)` whose registry overrides `gs`
   with a new `ImpersonatingGcsStoreProvider`.
3. The session is passed to `DatasetBuilder::with_session(...)` in
   `BlockingDataset::open`, and to `WriteParams::session` in `write_dataset`.

A fresh session per call means two concurrent opens with different target SAs
cannot collide on a shared registry.

### 7.5 Iceberg — one-shot pre-fetch + static bearer

Iceberg-rust's `gcs_config_parse` recognizes exactly five GCS keys
(`gcs.credentials-json`, `gcs.oauth2.token`, `gcs.service.path`,
`gcs.no-auth`, `gcs.allow-anonymous`) and silently drops everything else,
including `gcs.service-account`. Without intervention the effective identity
collapses to the VM's default SA and requests 403 against the customer
bucket.

`iceberg_bridgeimpl::iceberg_plan_files` pre-processes the props dict before
passing it to `FileIOBuilder`:

```rust
if let Some(target_sa) = props.remove("gcs.service-account") {
    if !target_sa.is_empty() {
        let bearer = fetch_impersonated_bearer(
            &target_sa,
            Duration::from_secs(DEFAULT_TOKEN_LIFETIME_SECS),
        ).await?;
        props.insert("gcs.oauth2.token".to_string(), bearer);
    }
}
```

`gcs.oauth2.token` maps to opendal `GcsConfig.token`, which is baked into
`GcsCore.token` with `usize::MAX` expiry — i.e. no refresh. That is fine
here: `plan_files` performs a handful of metadata/manifest reads in seconds,
and a 1-hour token is never at risk of expiring mid-flight. The much heavier
full-refresh machinery of `ImpersonatingGcsCredentialProvider` is
deliberately not wired in — it would require the `customized_token_loader`
hook iceberg-rust does not expose.

This asymmetry (cached + refreshable for Lance, one-shot static for iceberg)
is a direct function of how long each path keeps the token alive, not a
design preference.

### 7.6 Cargo additions

- `object_store` gains feature `gcp`.
- `async-trait = "0.1"` (for `CredentialProvider` / `ObjectStoreProvider`
  impls).
- `reqwest = "0.12"` with `default-features = false` +
  `features = ["json", "rustls-tls"]` — rustls matches the workspace TLS
  choice; already transitively present via lance-io / iceberg / reqsign, so
  no new transitive weight.
- `url = "2"` — kept as a direct dep because `url::Url` appears in the
  `ObjectStoreProvider::new_store` signature we implement.

## 8. Format-layer glue

### 8.1 Iceberg

`cpp/src/format/iceberg/iceberg_common.cpp`:

```cpp
} else if (provider == kCloudProviderGCP) {
  if (!config.gcp_target_service_account.empty()) {
    // Bridge-private key; consumed by iceberg_bridgeimpl::iceberg_plan_files
    // which swaps it for gcs.oauth2.token via fetch_impersonated_bearer.
    set("gcs.service-account", config.gcp_target_service_account);
  }
  // Otherwise default credentials via VM metadata.
}
```

The key `gcs.service-account` is not a standard iceberg-rust key — it is an
in-band signal between our C++ format layer and our Rust bridge. Iceberg-rust
itself would silently drop it; our bridge intercepts it first.

### 8.2 Lance

`cpp/src/format/lance/lance_common.cpp`:

```cpp
} else if (provider == kCloudProviderGCP) {
  if (!config.gcp_target_service_account.empty()) {
    set("gcp_target_service_account", config.gcp_target_service_account);
    if (config.load_frequency > 0) {
      options["gcp_credential_refresh_secs"] = std::to_string(config.load_frequency);
    }
  }
}
```

Again, both keys are bridge-private; `lance_bridgeimpl::open_dataset` and
`write_dataset` strip them from `storage_options` before forwarding to
lance-io.

## 9. Test infrastructure: cross-scheme metadata rewrite

This section documents scaffolding that is **pure test support**, not a
product feature. It is not in any of the three working docs because it only
became necessary once the E2E test was implementable end-to-end.

### 9.1 The problem

The GCP impersonation E2E test needs to:

1. **Write** the test table to a customer GCS bucket. The Rust GCS backends
   (`object_store::gcp`, `opendal::services::gcs`) both reject HMAC AK/SK —
   the only way to write with HMAC is through the AWS SDK against the GCS
   S3-compat endpoint (cloud_provider=aws + address=storage.googleapis.com).
   So the physical write goes via `s3://bucket/...`.
2. **Read** the table back via the feature under test — native `gs://` with
   SA impersonation.

Iceberg-rust bakes the absolute write-time URI into every level of a table's
metadata tree (`v1.metadata.json` → manifest list → manifests → data file
paths) and never rewrites on read. A top-level scheme flip on the
metadata-location string alone does not work because `plan_files` follows
the chain and any embedded `s3://...` reference under a `gs://` FileIO
produces `DataInvalid`.

### 9.2 The fix — `rewrite_iceberg_scheme`

After `iceberg_create_test_table` finishes writing, if the caller passed a
non-empty `record_scheme_override`, a helper walks the four metadata files
it just produced (`v1.metadata.json`, `snap-{id}-manifest-list.avro`,
`manifest-data-0.avro`, and optionally `manifest-deletes-0.avro`) and
byte-rewrites every occurrence of `<from>://` to `<to>://`.

Correctness conditions:

- **Equal byte length** required. `s3` and `gs` are both 2 bytes, so
  `s3://` and `gs://` are 5 bytes each — JSON stays valid and AVRO string
  length varints (which encode byte count, not codepoint count) don't need
  recomputing. The helper asserts this with `anyhow::bail!` if violated, to
  protect future callers.
- File names are fully determined by the write flow; we don't need to list
  the directory.
- opendal-backed FileIO replaces on `new_output().write()`, no explicit
  delete needed.

### 9.3 Surface

`CreateTestTable` gains a trailing `std::string record_scheme_override = ""`
parameter (empty string = no rewrite; cxx can't express `Option` across the
FFI, so the empty-string convention stands in). Production code never sets
it; only `ExternalTableGcpImpersonationTest::CreateIcebergTable` does, with
value `"gs"`.

## 10. Testing

### 10.1 Unit tests (storage options mapping)

`cpp/test/format/iceberg/iceberg_storage_options_test.cpp`:
- `GcpImpersonation` — expects `gcs.service-account` = target SA.
- `GcpDefaultCredentials` — expects empty output (fallback to VM default).

`cpp/test/format/lance/lance_storage_options_test.cpp`:
- `GcpImpersonation` — expects bridge-private
  `gcp_target_service_account` + `gcp_credential_refresh_secs`.
- `GcpDefaultCredentials` — expects empty output.

### 10.2 Unit tests (Rust)

`gcp_impersonation.rs` has three inline unit tests:
- `parse_rfc3339_basic` — RFC3339 → epoch ms.
- `impersonation_url_format` — hand-rolled URL matches the spec.
- `needs_refresh_when_empty` / `needs_refresh_within_offset` /
  `no_refresh_when_fresh` — refresh window boundary conditions.

### 10.3 Integration tests (GCP-backed; require live env)

`cpp/test/format/external_table_test.cpp` — `GcpS3CompatWriteTest`:

A prerequisite fixture that validates "can we even write to GCS via
S3-compat?" before the impersonation test piles identity swapping on top.
Uses the same env vars as the impersonation test:

- `LanceWriteAndRead` — full Lance round-trip (write via object_store S3,
  read via C++ FormatReader).
- `IcebergWriteAndPlanFiles` — opendal-backed write + `PlanFiles`. The data
  read is **intentionally not** exercised here: it would go through the C++
  Arrow S3 filesystem under cloud_provider=aws, which does not apply the
  GCS response-checksum workaround (that workaround keys on
  cloud_provider=gcp). The impersonation test covers the full round-trip
  under cloud_provider=gcp.

`cpp/test/format/external_table_arn_test.cpp` — `ExternalTableGcpImpersonationTest`:

Parameterized over `LOON_FORMAT_LANCE_TABLE` and `LOON_FORMAT_ICEBERG_TABLE`.
Flow, per parameter:

| Step | Props used | Credential | Target |
|------|------------|------------|--------|
| 1. Write test data | `write_props_` | HMAC AK/SK | customer bucket (S3-compat) |
| 2. Build LoonProperties | our-side `fs.*` (IAM) + `extfs.gcpsa.*` (target SA) | — | — |
| 3. `loon_exttable_explore` | step-2 props | VM SA → manifest bucket; **target SA → customer bucket (Rust)** | both |
| 4. Read manifest | step-2 props | VM SA | our bucket |
| 5. `FormatReader::create` | `read_props_` (`extfs.gcpsa.*` only) | **target SA (C++)** | customer bucket |

Notable fixture decisions:

- **Two separate buckets.** `OUR_TEST_ENV_*` supplies our-side manifest
  storage with IAM; `GCP_IMP_TEST_ENV_*` supplies the customer-side bucket
  with HMAC (for write) + target SA (for read). This matches production
  topology and avoids IAM/HMAC collisions on one address.
- **No `GetFileSystem()` in the GCP impersonation fixture.** Calling it with
  `cloud_provider=aws` would install the AWS S3 HttpClientFactory, which
  collides with the GCP factory installed later by `loon_exttable_explore`
  ("one cloud provider per process" — see commit `b5f8eef`). Test data is
  left in-bucket; rely on bucket lifecycle rules.
- **`region=auto`** for the S3-compat write side. GCS S3-compat ignores the
  region value but opendal's S3 Builder rejects empty strings.
- **Iceberg write uses `record_scheme_override = "gs"`** (§9).
- **`use_iam=true` is mandatory for impersonation reads**, otherwise
  `BuildGcpProviderFromConfig` falls through to `HmacProvider` with empty
  AK/SK and requests ship unsigned, returning 403.

## 11. Risks and open items

1. **Token refresh parity between C++ and Rust paths.** The C++
   `IamImpersonateProvider` relies on google-cloud-cpp's internal
   credential wrapper for caching and refresh; the Rust
   `ImpersonatingGcsCredentialProvider` uses our bespoke double-checked
   cache with `REFRESH_OFFSET_SECS = 300`. If E2E scans start seeing 401s
   just before expiry on one side only, check refresh-offset alignment
   first.

2. **Registry URI-normalization coverage.** A lookup miss degrades to a
   403 with an explicit error message (§6.6), but the integration tests
   only exercise the two common URI shapes. Additional unit tests for the
   registry's URI parsing (case sensitivity, port suffix, path vs
   vhost edge cases) would raise confidence; none landed in this commit.

3. **TLS floor / log level are still first-Make-wins.** Preserved
   pre-refactor behavior; not a new risk, but worth calling out
   alongside this change because the multi-identity refactor made it more
   visible.

4. **Bridge-private key names are an implicit API between the C++ format
   layer and the Rust bridge.** `gcp_target_service_account`,
   `gcp_credential_refresh_secs`, and `gcs.service-account` are not
   lance-io / object_store / opendal keys; all three sides must move
   together when any of them is renamed. Keeping the split in one place
   (the format-layer `ToStorageOptions` functions) is how we make that
   tractable today.
   carry a real subject and reference this design doc.
