# Tencent COS role-ARN integration test

`ExternalTableTencentArnTest.ReadTwoParquetFilesWithArnRole` reuses the
Parquet flow from `ExternalTableAliyunArnTest` in
`cpp/test/format/external_table_arn_test.cpp`.

It writes two Parquet files with setup credentials, then uses only the target
role ARN (and optional ExternalId) to discover the files, inspect their row
counts and read all 100 rows. It checks every ID, name and value. The read path
uses TKE OIDC to obtain the workload role, then assumes the distinct target
role. Setup credentials are not supplied to the external read filesystem.

## Environment

Run in a TKE test workload with the patched native library. Use test buckets.
The workload identity must provide these four nonempty variables:

- `TKE_REGION`
- `TKE_ROLE_ARN`
- `TKE_WEB_IDENTITY_TOKEN_FILE` (readable projected OIDC token)
- `TKE_PROVIDER_ID`

Configure the customer-side COS test bucket:

| Variable | Value |
| --- | --- |
| `TENCENT_ARN_TEST_ENV_ADDRESS` | COS endpoint without scheme or bucket, e.g. `cos.ap-shanghai.myqcloud.com` |
| `TENCENT_ARN_TEST_ENV_REGION` | Bucket region, e.g. `ap-shanghai` |
| `TENCENT_ARN_TEST_ENV_BUCKET` | Full bucket name including numeric APPID suffix |
| `TENCENT_ARN_TEST_ENV_ACCESS_KEY` | Setup SecretId with permission to write, list and delete test objects |
| `TENCENT_ARN_TEST_ENV_SECRET_KEY` | Setup SecretKey |
| `TENCENT_ARN_TEST_ENV_ROLE_ARN` | Target CAM role ARN, distinct from `TKE_ROLE_ARN` |
| `TENCENT_ARN_TEST_ENV_EXTERNAL_ID` | Optional ExternalId required by the target role trust policy |

The workload role must be allowed to assume the target role, and the target
role must trust that workload role. Give the target role list/read access to
the test prefix and keep the workload role itself without direct access to the
customer bucket, so successful reads demonstrate the second authorization hop.

The manifest uses the existing `OUR_TEST_ENV_ADDRESS`, `OUR_TEST_ENV_BUCKET`,
`OUR_TEST_ENV_REGION`, `OUR_TEST_ENV_CLOUD_PROVIDER`,
`OUR_TEST_ENV_ACCESS_KEY` and `OUR_TEST_ENV_SECRET_KEY` variables. These can
point to the same COS test bucket with cloud provider `tencent`. They use setup
credentials for manifest operations, independently of the external role-based
reads. Supply credentials through the test environment; do not commit them.

Each run creates a unique `zc/tencent-arn-test-*` prefix. Cleanup removes the
Parquet files from the customer bucket and the manifest subdirectory from the
manifest bucket. Both setup identities need the corresponding cleanup rights.

## Run

Build `milvus_test` with the normal C++ build, then run:

```bash
./cpp/build/Release/test/milvus_test \
  --gtest_filter=ExternalTableTencentArnTest.ReadTwoParquetFilesWithArnRole \
  --gtest_output=xml:tencent-cos-arn.xml
```

Missing environment configuration produces `SKIPPED`, not a successful COS
validation. Acceptance requires one executed, passing test and no skipped
cases. Ordinary CI without TKE/COS credentials only compiles this test and
checks its skip behavior. Existing protocol unit tests separately cover
signature generation, refresh, token rotation and failure handling.
