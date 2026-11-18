# Object-storage request-ID validation

## Real endpoint probes

On 2026-11-18 at 08:48–08:55 UTC, sent 34 anonymous HTTPS HEAD/GET requests
with certificate verification enabled. No credentials, Authorization header,
resource creation or object writes were used. Redirects were not followed.
All 34 responses carried a server request/diagnostic ID.

For S3-compatible providers, tested both the service root and an object path
under a randomly generated nonexistent bucket. Azure uses Microsoft's documented
public `azureopendatastorage` account; tested the service root, a random nonexistent
container, and a random nonexistent blob in the existing public `nyctlc` container.

Representative real HEAD responses:

| Service | Endpoint | HTTP | Header | Real ID |
|---|---|---:|---|---|
| aws-s3 | `s3.us-east-1.amazonaws.com` | 404 | `x-amz-request-id` | `7YRZMG17SF1MR7J2` |
| tencent-cos | `cos.ap-beijing.myqcloud.com` | 404 | `x-cos-request-id` | `NmFmZDY2NTVfMTIyOTY1MDlfOGUzMF9mYTVhNjZh` |
| aliyun-oss | `oss-cn-beijing.aliyuncs.com` | 404 | `x-oss-request-id` | `6AFD6655A5D144323554D0AD` |
| huawei-obs | `obs.cn-north-4.myhuaweicloud.com` | 404 | `x-obs-request-id` | `000001A1EDD7C02144D020DC4B5D8EA7` |
| volcengine-tos-native | `tos-cn-beijing.volces.com` | 404 | `x-tos-request-id` | `709307fd66565e176afd6656-b22fb3b-1xTbL4-HO-cb-tos-1az-front-azc-1` |
| volcengine-tos-s3 | `tos-s3-cn-beijing.volces.com` | 404 | `x-amz-request-id` | `7c1007fd66567c6e6afd6656-b4496e1-1xTbL4-HO-cb-tos-1az-front-azc-1` |
| gcp-gcs-xml | `storage.googleapis.com` | 404 | `x-guploader-uploadid` | `AFy3sg2I73AAXCtsDYr2leUezZRlcFOubQzfF_psCBR4lSCj6KW-iTzQJxs84PWYILoveh41xzKRqSA` |
| Azure Blob | `azureopendatastorage.blob.core.windows.net` | 404 | `x-ms-request-id` | `69ba9dd5-301e-003e-233a-771ec4000000` |

Tencent COS, Alibaba OSS and Huawei OBS service-root GETs returned 403
`AccessDenied`, with their respective provider request-ID headers. GCS's root
returned 400 `MissingSecurityHeader`, with `x-guploader-uploadid`.

Volcengine's S3 **object** requests returned `x-amz-request-id`, while its native
TOS endpoint returned `x-tos-request-id`. The S3 service-root HEAD also returned
`x-tos-request-id`. GCS XML responses returned `x-guploader-uploadid`; Azure Blob
responses returned `x-ms-request-id`.

Reproduction examples (use a new random bucket/container/blob name per run):

```sh
curl -q --head --proto '=https' --connect-timeout 8 --max-time 20 \
  'https://<random-nonexistent-bucket>.tos-s3-cn-beijing.volces.com/request-id-probe/missing-object'
curl -q --head --proto '=https' --connect-timeout 8 --max-time 20 \
  'https://<random-nonexistent-bucket>.storage.googleapis.com/request-id-probe/missing-object'
curl -q --head --proto '=https' --connect-timeout 8 --max-time 20 \
  'https://azureopendatastorage.blob.core.windows.net/nyctlc/<random-nonexistent-blob>'
```

These are **anonymous HTTP probes**, not authenticated AKSK/IAM tests or a full
C++ SDK-to-status cloud integration run. The nonexistent-bucket requests do not
validate existing-bucket permissions or authentication. The Azure existing-container
probe validates a missing blob in that public container.

## Formatter regression tests

- `S3ErrorRequestIdTest.*`: SDK field precedence; explicit cloud headers, including
  TOS, GCS and Azure; case-insensitive matching; explicit headers ahead of the
  generic `*-request-id` fallback; empty values; unrelated headers; and transport
  errors without a response.
- `AzureErrorRequestIdTest.*`: SDK `RequestId` precedence; `x-ms-request-id` fallback;
  preservation in tagged and untagged Arrow errors; missing IDs; and transport
  classification. The client's `x-ms-client-request-id` is not substituted for
  the Azure server's ID.

The cloud request-ID header shapes above are covered by formatter tests.
The newly extended full gtest suite must pass CI before claiming an SDK/Arrow
integration pass for this revision.

References:

- [TOS S3 compatibility and endpoint selection](https://docs.volcengine.com/docs/torch-object-storage/compatibility-with-the-amazon-s-3-protocol?lang=en)
- [GCS XML response headers](https://docs.cloud.google.com/storage/docs/xml-api/reference-headers#xguploaderuploadid)
- [Azure Storage troubleshooting and request IDs](https://learn.microsoft.com/en-us/rest/api/storageservices/troubleshooting-api-operations)
- [Microsoft's documented public Azure dataset account](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-taxi-yellow)
