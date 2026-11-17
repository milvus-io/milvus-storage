# Narrow Azure SDK callback exception allowance

This is an explicit change to the ratchet's original **any library throw is a
violation** policy. It does not raise the existing baseline or exempt a file.
The existing per-category comparison with the PR base remains unchanged.

Azure's `TokenCredential::GetToken` returns `AccessToken`, not `Status` or
`Result`; its failure channel is `AuthenticationException`. Returning an empty
or expired token would instead send an invalid credential to the SDK consumer.
The following two reviewed interface boundaries are the only allowances:

| Fixed source file | Boundary | Exact exception expression | Maximum |
| --- | --- | --- | --- |
| `cpp/src/filesystem/azure/refreshing_token_credential.cpp` | `RefreshingAzureCredential::GetToken` | `Azure::Core::Credentials::AuthenticationException("Azure credential refresh failed; no usable cached token")` | 1 |
| Same file | `AzureStorageCredential::GetToken` | `Azure::Storage::StorageException(error.what())` | 1 |

The classifier checks the exact path, namespace/class/member/catch scopes and
statement tokens. A nested lambda, local class or other new brace scope cannot
inherit an enclosing method's allowance. Additional sites, other types, rethrows, changed expressions,
or moving either site to another boundary remain violations. Comments and
string literals cannot supply a declaration or allowed statement. Duplicating
a permitted statement invalidates its allowance, including duplicates on one
line. There is no opt-in annotation or unrestricted per-file exclusion.

## Failure contract and consumer containment

The raw refresh wrapper validates the fetched token after the fetch completes,
keeps an old token only while it is valid, and uses the SDK's typed error when
no usable token remains. Its message is local and sanitized. A forced refresh
following a server rejection must not return the rejected token.

AzureFS historically catches `StorageException` or its `RequestFailedException`
base, whereas `AuthenticationException` directly derives from `std::exception`.
The storage-only adapter converts the sanitized raw error to `StorageException`
before it reaches AzureFS. Its HTTP status stays `None` and its raw response is
null: it does not invent a server response. AzureFS then returns an Arrow error,
including inside Arrow executor tasks. Legacy Milvus keeps the raw SDK contract;
its `AzureChunkManager` already catches `std::exception` and translates failures
to its existing error boundary.

Checked-in evidence:

- **G1, provider contract:** `cpp/test/filesystem/azure_refreshing_credential_test.cpp`
  tests valid-old fallback, invalid candidate rejection, expiry at completion,
  scope/tenant isolation, recovery and forced invalidation.
- **G2, real consumer boundary:**
  `TestAzureOptions.CredentialFailureReturnsArrowStatusForSyncAndAsyncReads` in
  `cpp/test/filesystem/azure_file_system_test.cpp` runs actual AzureFS, the Azure
  Blob SDK, and Arrow's executor with throwing, empty and expired credentials.
  `GetSize`, `ReadAt` and `ReadAsync` must return Arrow `IOError` without leaking
  an exception or a secret. The credential failure occurs before HTTP transport.
  Background writes use the same adapter and their existing `StorageException`
  catch; this is a source trace, not a background-write runtime claim.
- `cpp/scripts/test_error_handling_sdk_exceptions.py` tests the two allowances
  and rejects extra, moved, untyped, duplicate and comment-masked sites. CI runs
  this test before the unchanged baseline/base-total gates.

Any additional SDK contract exception needs a separate reviewed policy change
with its own consuming-boundary evidence; adding a throw or raising the baseline
does not authorize it.
