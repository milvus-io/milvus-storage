# Endpoint Scheme from `use_ssl` for Lance and Iceberg

## Context

Lance and Iceberg convert `ArrowFileSystemConfig::address` into backend storage
endpoint options through `StorageUri::BuildEndpointUrl`. Today that helper keeps
an explicit URL scheme if present, but prepends `https://` to bare addresses.
Because `lance_common.cpp` and `iceberg_common.cpp` do not pass
`ArrowFileSystemConfig::use_ssl` into that helper, a config with
`use_ssl=false` and `address=localhost:9000` still produces an HTTPS endpoint.

This differs from the regular S3 and GCP filesystem producers, which derive the
transport scheme from `use_ssl`.

## Decision

Extend `StorageUri::BuildEndpointUrl` so it accepts `use_ssl`:

```cpp
static std::string BuildEndpointUrl(const std::string& address, bool use_ssl = true);
```

The helper will keep explicit schemes authoritative:

- `address=""` returns `""`.
- `address="http://localhost:9000"` returns `http://localhost:9000`.
- `address="https://example.com"` returns `https://example.com`.
- Bare addresses use `use_ssl`: `true` becomes `https://...`, `false` becomes
  `http://...`.

The default argument preserves the old standalone helper behavior for any
callers that do not opt into `use_ssl`.

## Scope

Update the shared endpoint helper and the two format option builders that depend
on it:

- `StorageUri::BuildEndpointUrl` declaration and implementation.
- `lance::ToStorageOptions`, specifically the common endpoint setter used by
  AWS, GCP, and OSS-style endpoints.
- `iceberg::ToStorageOptions`, specifically the common endpoint setter used by
  AWS S3 and Aliyun OSS endpoints.

Do not change Azure endpoint handling. Lance Azure already uses
`BuildAzureEndpointAddress(..., use_ssl)`, and Iceberg Azure passes an endpoint
suffix rather than a full HTTP endpoint.

Do not change regular S3, GCP, or Azure filesystem producers. Their direct
`use_ssl` handling already matches this design.

## Data Flow

For Lance and Iceberg, endpoint construction becomes:

1. The filesystem config is resolved from normal properties or `extfs.*`
   properties.
2. `ToStorageOptions(config)` passes `config.address` and `config.use_ssl` to
   `BuildEndpointUrl`.
3. If the resulting endpoint starts with `http://`, the format storage options
   include `allow_http=true`.
4. The Rust-backed Lance/Iceberg storage layer receives a final endpoint URL
   whose scheme already reflects either the explicit address scheme or
   `use_ssl`.

This means callers that use a bare address and want HTTPS must set
`use_ssl=true`. With the current `ArrowFileSystemConfig` default of
`use_ssl=false`, a bare address intentionally resolves to HTTP after this
change.

## Error Handling

No new validation is added. Existing behavior for malformed or unsupported
endpoint strings remains unchanged: the helper only decides whether to preserve
an explicit scheme or prepend one.

Explicit URL schemes take precedence over `use_ssl`, so conflicting inputs such
as `address=https://host` with `use_ssl=false` remain HTTPS rather than being
rewritten.

## Testing

Add or update focused unit tests for both Lance and Iceberg storage options:

- Bare address plus `use_ssl=false` produces an `http://` endpoint and sets
  `allow_http=true`.
- Bare address plus `use_ssl=true` produces an `https://` endpoint and does not
  set `allow_http`.
- Explicit `http://` is preserved and sets `allow_http=true`.
- Explicit `https://` is preserved even if `use_ssl=false` and does not set
  `allow_http`.

Adjust existing tests that expect HTTPS from a bare address to set
`use_ssl=true` explicitly.
