# Rust TLS in Milvus FIPS builds

Tracking issue [milvus-io/milvus#55083](https://github.com/milvus-io/milvus/issues/55083). The Rust Storage bridge uses the existing `MILVUS_FIPS_ENABLED=ON` build switch. Milvus passes the switch through the build environment. Standalone Storage builds can set the same environment variable for `make build` or configure CMake with `-DMILVUS_FIPS_ENABLED=ON`.

## TLS and signing

The FIPS build keeps Lance, Iceberg, Delta, Paimon, Vortex, and Talon available. Both versions of reqwest use a Rustls FIPS `CryptoProvider` and reject a client configuration that fails the Rustls FIPS check. Talon uses the same provider for its control-plane mutual TLS and the AWS SDK uses Smithy's `AwsLcFips` connector, including Lance's default credential chain. The bridge checks the actual module version and mode before starting its Rust runtime or opening a Talon client. [Rustls FIPS guide](https://docs.rs/rustls/0.23.46/rustls/manual/_06_fips/index.html)

The FIPS-only lock and patches are applied in the build tree. The ordinary build uses the original dependency sources. The graph check rejects `ring`, ordinary `aws-lc-sys`, and the Rust native OpenSSL connectors. Cloud request signing in `object_store`, Reqsign, Talon, and the AWS SDK uses the same AWS-LC binding on the migrated paths. Approved use depends on the operation and key parameters. OSS HMAC-SHA1 request authentication requires a key that meets the module policy. Azure's SHA-1 `x5t` field identifies a certificate, and the JWT uses RS256.

## Module and link boundary

The builder checks the checksum of the source committed under `1276fc7d9cd0f509000294e9657914c3a23e65d2` and builds the AWS-LC FIPS 4.0.0 static module with the options specified in section 11.1. The Rust `aws-lc-fips-sys` 0.14.2 binding uses that separately built archive and generates bindings from those headers. This build requires a native Linux GNU target, Python 3, Go, a C and C++ toolchain, GNU binutils, and libclang for bindgen. [CMVP certificate #5596 and security policy](https://csrc.nist.gov/projects/cryptographic-module-validation-program/certificate/5596)

AWS-LC and OpenSSL have overlapping C symbols. The build resolves the Rust archive's AWS-LC references and localizes its definitions before linking `libmilvus-storage.so`. C++ can then resolve its crypto calls against OpenSSL. The final link checks for private AWS-LC FIPS symbols and rejects ordinary AWS-LC and Ring symbols. The certified module's source build is separate from this consumer link step.

## Release verification

1. Run the normal and FIPS builds for Linux arm64 and x86-64. Confirm the normal configuration keeps its existing provider and the FIPS artifact passes its symbol checks.
2. Use the final Storage and Milvus binaries to exercise TLS 1.2, TLS 1.3, certificate rejection, Talon mTLS, and cloud requests with credential refresh. Verify both C++ OpenSSL and Rust AWS-LC within the same process.
3. Review the final link procedure and the intended runtime OS and processor against the module security policy. The local Amazon Linux 2023 arm64 container verifies functionality. The policy lists the environments covered by the certificate.
4. After the Storage change is available upstream, update Milvus's Storage revision and verify the FIPS image. Milvus's separately built Tantivy analyzer has its own remote TLS build setting and must be reviewed in that image.

## Updating the lock

Run `prepare.py --update-lock` with the Cargo toolchain used by the Storage build. Update each affected patch against its version in the ordinary lock, then run the locked graph check on both supported architectures. The program stops when a patch does not apply or when an unapproved TLS backend enters the resolved build graph.
