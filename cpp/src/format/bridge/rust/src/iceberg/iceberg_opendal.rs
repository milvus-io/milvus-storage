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

//! Iceberg `FileIO` adapter over the generic filesystem-backed OpenDAL access.
//!
//! This module owns the Iceberg-specific boundary: strict credential-free read
//! options, absolute-URI validation, filesystem identity binding, and the
//! `StorageFactory` / `Storage` implementations required by iceberg-rust.
//!
//! Every Iceberg URI is reduced to a relative object path only after its scheme
//! family and bucket/container (or local root) match the filesystem selected by
//! C++. The adapter is intentionally read-only and never falls back to a native
//! Rust cloud provider.

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use bytes::Bytes;
use iceberg::{Error, ErrorKind, Result};
use opendal::Operator;

use crate::filesystem_opendal::FilesystemAccess;

const FS_IS_LOCAL: &str = "milvus_fs_is_local";
const FS_ROOT_PATH: &str = "milvus_fs_root_path";
const FS_BUCKET: &str = "milvus_fs_bucket";

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::DataInvalid, message)
}

// -----------------------------------------------------------------------------
// Credential-free reader options and URI identity binding
// -----------------------------------------------------------------------------

/// Filesystem identity passed by C++ for one planning call.
///
/// These options describe only the already-selected filesystem. Provider
/// endpoints and credentials are intentionally invalid here because the Rust
/// adapter must not create a second cloud client or credential lifecycle.
#[derive(Clone, Debug)]
pub(crate) enum FilesystemReadOptions {
    Local { root: PathBuf },
    Remote { bucket: String },
}

impl FilesystemReadOptions {
    /// Parse the complete option map and reject every unconsumed key.
    ///
    /// Strict consumption prevents a future caller from accidentally restoring
    /// the old Rust credential path. Error messages name only the key, never its
    /// value, so a rejected credential cannot be logged.
    pub(crate) fn parse(mut options: HashMap<String, String>) -> Result<Self> {
        let is_local = options
            .remove(FS_IS_LOCAL)
            .ok_or_else(|| {
                invalid(format!(
                    "missing required filesystem read option: {FS_IS_LOCAL}"
                ))
            })?
            .parse::<bool>()
            .map_err(|_| invalid(format!("invalid filesystem read option: {FS_IS_LOCAL}")))?;

        let identity = if is_local {
            let root = PathBuf::from(options.remove(FS_ROOT_PATH).ok_or_else(|| {
                invalid(format!(
                    "missing required filesystem read option: {FS_ROOT_PATH}"
                ))
            })?);
            if !root.is_absolute()
                || root
                    .components()
                    .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
            {
                return Err(invalid(format!(
                    "filesystem read option {FS_ROOT_PATH} must be an absolute normalized path"
                )));
            }
            Self::Local { root }
        } else {
            let bucket = options
                .remove(FS_BUCKET)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    invalid(format!(
                        "missing required filesystem read option: {FS_BUCKET}"
                    ))
                })?;
            Self::Remote { bucket }
        };

        if let Some(key) = options.keys().min() {
            return Err(invalid(format!(
                "unsupported filesystem read option: {key}"
            )));
        }

        Ok(identity)
    }
}

/// URI aliases that may safely address the same bound filesystem.
///
/// Aliases are grouped by storage semantics, not merely by URL syntax. A table
/// bound from S3 may reference `s3a`, for example, but may never switch to GCS.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SchemeFamily {
    S3,
    Gcs,
    Azure,
    Oss,
}

fn scheme_family(scheme: &str) -> Option<SchemeFamily> {
    match scheme {
        "s3" | "s3a" => Some(SchemeFamily::S3),
        "gs" | "gcs" => Some(SchemeFamily::Gcs),
        "azure" | "abfs" | "abfss" | "wasb" | "wasbs" => Some(SchemeFamily::Azure),
        "oss" => Some(SchemeFamily::Oss),
        _ => None,
    }
}

fn validate_raw_uri_path(uri: &str) -> Result<()> {
    let scheme_end = uri
        .find("://")
        .ok_or_else(|| invalid("Iceberg path must be an absolute URI"))?;
    let rest = &uri[scheme_end + 3..];
    let path_start = rest
        .find('/')
        .ok_or_else(|| invalid("Iceberg URI must contain an object path"))?;
    match object_store::path::Path::from_url_path(&rest[path_start..]) {
        Ok(_) => Ok(()),
        Err(object_store::path::Error::BadSegment { path, .. }) => {
            if path.split('/').any(|segment| segment == "..") {
                return Err(invalid("Iceberg object path must not contain parent traversal"));
            }
            let without_current = path
                .split('/')
                .filter(|segment| *segment != ".")
                .collect::<Vec<_>>()
                .join("/");
            object_store::path::Path::parse(without_current)
                .map(|_| ())
                .map_err(|error| invalid(format!("invalid Iceberg object path: {error}")))
        }
        Err(error) => Err(invalid(format!("invalid Iceberg object path: {error}"))),
    }
}

/// Parse a remote Iceberg URI into its scheme family, logical authority, and
/// relative object path.
///
/// The logical Azure authority is the container before `@`; any recorded
/// account endpoint is metadata, not permission to escape the C++ filesystem.
/// Non-Azure user information, queries, and fragments are rejected to keep path
/// mapping unambiguous.
fn parse_remote_uri(uri: &str) -> Result<(SchemeFamily, String, String)> {
    let scheme_end = uri
        .find("://")
        .ok_or_else(|| invalid("remote Iceberg path must be an absolute URI"))?;
    let scheme = uri[..scheme_end].to_ascii_lowercase();
    let family = scheme_family(&scheme)
        .ok_or_else(|| invalid(format!("unsupported Iceberg URI scheme: {scheme}")))?;
    let url =
        url::Url::parse(uri).map_err(|error| invalid(format!("invalid Iceberg URI: {error}")))?;
    if url.query().is_some() || url.fragment().is_some() {
        return Err(invalid("Iceberg URI must not contain a query or fragment"));
    }
    let rest = &uri[scheme_end + 3..];
    let authority_end = rest.find('/').unwrap_or(rest.len());
    let raw_authority = &rest[..authority_end];
    if raw_authority.is_empty() {
        return Err(invalid("remote Iceberg URI must contain an authority"));
    }
    let authority = if family == SchemeFamily::Azure {
        raw_authority.split('@').next().unwrap_or(raw_authority)
    } else {
        if raw_authority.contains('@') {
            return Err(invalid(
                "non-Azure Iceberg URI must not contain user information",
            ));
        }
        raw_authority
    };

    validate_raw_uri_path(uri)?;
    let path = object_store::path::Path::from_url_path(url.path())
        .map_err(|error| invalid(format!("invalid Iceberg object path: {error}")))?
        .to_string();
    if path.is_empty() {
        return Err(invalid("Iceberg URI must contain an object path"));
    }
    Ok((family, authority.to_string(), path))
}

/// Identity constraint applied to every absolute URI followed by iceberg-rust.
///
/// The metadata location establishes the allowed scheme family plus
/// bucket/container, or the allowed local root. Manifest lists, manifests,
/// delete files, and data files must all resolve through this same binding.
/// This prevents table metadata from redirecting reads to another tenant.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum IcebergUriBinding {
    Local {
        root: PathBuf,
    },
    Remote {
        family: SchemeFamily,
        authority: String,
    },
}

impl IcebergUriBinding {
    /// Establish the binding from the initial metadata location and validate
    /// that the location itself is reachable through the selected filesystem.
    pub(crate) fn new(metadata_location: &str, options: &FilesystemReadOptions) -> Result<Self> {
        let binding = match options {
            FilesystemReadOptions::Local { root } => Self::Local { root: root.clone() },
            FilesystemReadOptions::Remote { bucket } => {
                let (family, authority, _) = parse_remote_uri(metadata_location)?;
                if authority != *bucket {
                    return Err(invalid("Iceberg URI does not match the bound filesystem"));
                }
                Self::Remote { family, authority }
            }
        };
        binding.relative_path(metadata_location)?;
        Ok(binding)
    }

    /// Validate an Iceberg absolute URI and return only the relative path that
    /// the generic filesystem-backed OpenDAL access is allowed to receive.
    pub(crate) fn relative_path(&self, uri: &str) -> Result<String> {
        match self {
            Self::Remote { family, authority } => {
                let (candidate_family, candidate_authority, path) = parse_remote_uri(uri)?;
                if candidate_family != *family || candidate_authority != *authority {
                    return Err(invalid("Iceberg URI does not match the bound filesystem"));
                }
                Ok(path)
            }
            Self::Local { root } => local_relative_path(root, uri),
        }
    }
}

/// Map an absolute or file URI under `root` to a normalized object path.
/// Parent traversal is rejected before `strip_prefix` so relative paths cannot
/// escape the bound local filesystem either lexically or through URI syntax.
fn local_relative_path(root: &Path, uri: &str) -> Result<String> {
    let path = if uri.contains("://") {
        let url = url::Url::parse(uri)
            .map_err(|error| invalid(format!("invalid local Iceberg URI: {error}")))?;
        if url.scheme() != "file" || url.query().is_some() || url.fragment().is_some() {
            return Err(invalid(
                "local Iceberg URI must use file scheme without query or fragment",
            ));
        }
        validate_raw_uri_path(uri)?;
        url.to_file_path()
            .map_err(|_| invalid("local Iceberg URI is not a valid file path"))?
    } else {
        PathBuf::from(uri)
    };
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(invalid(
            "local Iceberg path must not contain parent traversal",
        ));
    }
    let relative = if path.is_absolute() {
        path.strip_prefix(root)
            .map_err(|_| invalid("local Iceberg path is outside the bound filesystem root"))?
            .to_path_buf()
    } else {
        path
    };
    let relative = relative
        .to_str()
        .ok_or_else(|| invalid("local Iceberg path is not valid UTF-8"))?
        .replace(std::path::MAIN_SEPARATOR, "/");
    object_store::path::Path::parse(relative)
        .map(|path| path.to_string())
        .map_err(|error| invalid(format!("invalid local Iceberg object path: {error}")))
}

/// Preserve the OpenDAL error as the source while crossing into Iceberg's error
/// type. The C++ bridge later recovers detailed filesystem status from that
/// source chain.
fn from_filesystem_opendal_error(error: opendal::Error) -> Error {
    Error::new(ErrorKind::Unexpected, "filesystem-backed OpenDAL I/O failed").with_source(error)
}

/// Return one consistent Iceberg error for every mutation entry point.
fn read_only_error(operation: &str) -> Error {
    Error::new(
        ErrorKind::FeatureUnsupported,
        format!("filesystem-backed Iceberg storage does not support {operation}"),
    )
}

// -----------------------------------------------------------------------------
// iceberg-rust StorageFactory and read-only Storage adapter
// -----------------------------------------------------------------------------

/// Iceberg factory carrying the validated URI binding and runtime filesystem
/// lease into each Storage instance.
///
/// `StorageFactory` is typetag-serializable, but a CXX shared pointer is not.
/// Serde therefore keeps only the binding. A deserialized factory fails in
/// `build` instead of silently constructing a native cloud provider.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct IcebergFilesystemStorageFactory {
    binding: IcebergUriBinding,
    #[serde(skip)]
    filesystem: Option<cxx::SharedPtr<crate::lance_ffi::FileSystemWrapper>>,
}

impl fmt::Debug for IcebergFilesystemStorageFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Do not print the runtime filesystem or bound tenant identity.
        formatter
            .debug_struct("IcebergFilesystemStorageFactory")
            .finish_non_exhaustive()
    }
}

impl IcebergFilesystemStorageFactory {
    /// Construct a runtime factory. This is the only path that supplies the
    /// serde-skipped filesystem lease.
    fn new(
        filesystem: cxx::SharedPtr<crate::lance_ffi::FileSystemWrapper>,
        binding: IcebergUriBinding,
    ) -> Self {
        Self {
            binding,
            filesystem: Some(filesystem),
        }
    }
}

#[typetag::serde(name = "IcebergFilesystemStorageFactory")]
impl iceberg::io::StorageFactory for IcebergFilesystemStorageFactory {
    fn build(
        &self,
        _config: &iceberg::io::StorageConfig,
    ) -> Result<Arc<dyn iceberg::io::Storage>> {
        let filesystem = self.filesystem.clone().ok_or_else(|| {
            invalid("filesystem-backed Iceberg factory is missing its runtime filesystem")
        })?;
        // Build a fresh lightweight operator per FileIO while sharing the same
        // C++ filesystem/client through the cloned CXX shared pointer.
        let operator = FilesystemAccess::new(filesystem)
            .map_err(from_filesystem_opendal_error)?
            .into_operator();
        Ok(Arc::new(IcebergFilesystemStorage {
            binding: self.binding.clone(),
            operator: Some(operator),
        }))
    }
}

/// Iceberg storage that validates absolute URIs before delegating relative
/// paths to the generic OpenDAL operator.
///
/// The operator is skipped by serde for the same reason as the filesystem in
/// the factory. Cloning this storage is cheap and retains the operator's shared
/// accessor; missing runtime state is always an explicit error.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct IcebergFilesystemStorage {
    binding: IcebergUriBinding,
    #[serde(skip)]
    operator: Option<Operator>,
}

impl fmt::Debug for IcebergFilesystemStorage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Avoid leaking the bound authority or operator internals.
        formatter
            .debug_struct("IcebergFilesystemStorage")
            .finish_non_exhaustive()
    }
}

impl IcebergFilesystemStorage {
    /// Access runtime-only state, failing explicitly after deserialization.
    fn operator(&self) -> Result<&Operator> {
        self.operator.as_ref().ok_or_else(|| {
            invalid("filesystem-backed Iceberg storage is missing its runtime operator")
        })
    }

    /// Enforce the table's URI binding before the path reaches OpenDAL.
    fn relative_path(&self, path: &str) -> Result<String> {
        self.binding.relative_path(path)
    }
}

#[typetag::serde(name = "IcebergFilesystemStorage")]
#[async_trait::async_trait]
/// Read methods always validate and map the absolute Iceberg URI first.
/// Mutation methods fail immediately: production planning must never write
/// through the filesystem lease, even when metadata contains a malicious path.
impl iceberg::io::Storage for IcebergFilesystemStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        let relative = self.relative_path(path)?;
        self.operator()?
            .exists(&relative)
            .await
            .map_err(from_filesystem_opendal_error)
    }

    async fn metadata(&self, path: &str) -> Result<iceberg::io::FileMetadata> {
        let relative = self.relative_path(path)?;
        let metadata = self
            .operator()?
            .stat(&relative)
            .await
            .map_err(from_filesystem_opendal_error)?;
        Ok(iceberg::io::FileMetadata {
            size: metadata.content_length(),
        })
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        let relative = self.relative_path(path)?;
        Ok(self
            .operator()?
            .read(&relative)
            .await
            .map_err(from_filesystem_opendal_error)?
            .to_bytes())
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn iceberg::io::FileRead>> {
        let relative = self.relative_path(path)?;
        let reader = self
            .operator()?
            .reader(&relative)
            .await
            .map_err(from_filesystem_opendal_error)?;
        Ok(Box::new(IcebergOpenDalReader(reader)))
    }

    async fn write(&self, _path: &str, _bytes: Bytes) -> Result<()> {
        Err(read_only_error("write"))
    }

    async fn writer(&self, _path: &str) -> Result<Box<dyn iceberg::io::FileWrite>> {
        Err(read_only_error("writer"))
    }

    async fn delete(&self, _path: &str) -> Result<()> {
        Err(read_only_error("delete"))
    }

    async fn delete_prefix(&self, _path: &str) -> Result<()> {
        Err(read_only_error("delete_prefix"))
    }

    fn new_input(&self, path: &str) -> Result<iceberg::io::InputFile> {
        // InputFile retains the absolute URI for Iceberg's public contract. The
        // cloned storage validates it when a concrete read operation begins.
        Ok(iceberg::io::InputFile::new(
            Arc::new(self.clone()),
            path.to_string(),
        ))
    }

    fn new_output(&self, _path: &str) -> Result<iceberg::io::OutputFile> {
        Err(read_only_error("new_output"))
    }
}

/// Iceberg's range-reader facade over the OpenDAL reader returned after URI
/// binding. The OpenDAL layer owns range semantics and filesystem error mapping.
struct IcebergOpenDalReader(opendal::Reader);

#[async_trait::async_trait]
impl iceberg::io::FileRead for IcebergOpenDalReader {
    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        Ok(opendal::Reader::read(&self.0, range)
            .await
            .map_err(from_filesystem_opendal_error)?
            .to_bytes())
    }
}

/// Build the FileIO used by production planning.
///
/// Parsing and binding happen once here. All Storage instances created later by
/// iceberg-rust inherit the same filesystem identity and cannot fall back to
/// provider options embedded in table metadata. The caller also receives the
/// binding so it can validate planned data-file URIs, which FileIO does not read.
pub(crate) fn build_filesystem_file_io(
    filesystem: cxx::SharedPtr<crate::lance_ffi::FileSystemWrapper>,
    metadata_location: &str,
    read_options: HashMap<String, String>,
) -> Result<(iceberg::io::FileIO, IcebergUriBinding)> {
    let options = FilesystemReadOptions::parse(read_options)?;
    let binding = IcebergUriBinding::new(metadata_location, &options)?;
    let factory = IcebergFilesystemStorageFactory::new(filesystem, binding.clone());
    let file_io = iceberg::io::FileIOBuilder::new(Arc::new(factory)).build();
    Ok((file_io, binding))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn remote_binding_accepts_all_documented_scheme_aliases() {
        for aliases in [
            &["s3", "s3a"][..],
            &["gs", "gcs"][..],
            &["azure", "abfs", "abfss", "wasb", "wasbs"][..],
            &["oss"][..],
        ] {
            let options = FilesystemReadOptions::parse(remote_options("bucket")).unwrap();
            for metadata_scheme in aliases {
                let metadata = format!("{metadata_scheme}://bucket/table/metadata/v1.json");
                let binding = IcebergUriBinding::new(&metadata, &options).unwrap();
                for referenced_scheme in aliases {
                    let referenced =
                        format!("{referenced_scheme}://bucket/table/metadata/m.avro");
                    assert_eq!(
                        binding.relative_path(&referenced).unwrap(),
                        "table/metadata/m.avro"
                    );
                }
            }
        }
    }

    fn remote_options(bucket: &str) -> HashMap<String, String> {
        HashMap::from([
            ("milvus_fs_is_local".into(), "false".into()),
            ("milvus_fs_bucket".into(), bucket.into()),
        ])
    }

    fn local_options(root: &str) -> HashMap<String, String> {
        HashMap::from([
            ("milvus_fs_is_local".into(), "true".into()),
            ("milvus_fs_root_path".into(), root.into()),
        ])
    }

    #[test]
    fn reader_options_reject_credentials_and_unknown_keys() {
        let mut options = remote_options("bucket");
        options.insert("s3.access-key-id".into(), "secret".into());

        let error = FilesystemReadOptions::parse(options).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("unsupported filesystem read option")
        );
        assert!(!error.to_string().contains("secret"));
    }

    #[test]
    fn reader_options_require_complete_identity() {
        let cases = [
            HashMap::new(),
            HashMap::from([(FS_IS_LOCAL.into(), "invalid".into())]),
            HashMap::from([(FS_IS_LOCAL.into(), "true".into())]),
            HashMap::from([(FS_IS_LOCAL.into(), "false".into())]),
            HashMap::from([
                (FS_IS_LOCAL.into(), "false".into()),
                (FS_BUCKET.into(), String::new()),
            ]),
        ];

        for options in cases {
            assert!(FilesystemReadOptions::parse(options).is_err());
        }
    }

    #[test]
    fn reader_options_reject_mixed_local_and_remote_identity() {
        let mut local = local_options("/data/root");
        local.insert(FS_BUCKET.into(), "bucket".into());
        let mut remote = remote_options("bucket");
        remote.insert(FS_ROOT_PATH.into(), "/data/root".into());
        let relative_local = local_options("relative/root");

        for options in [local, remote, relative_local] {
            assert!(FilesystemReadOptions::parse(options).is_err());
        }
    }

    #[test]
    fn remote_binding_rejects_other_identity() {
        let options = FilesystemReadOptions::parse(remote_options("bucket")).unwrap();
        let binding =
            IcebergUriBinding::new("s3://bucket/table/metadata/v1.json", &options).unwrap();

        assert!(
            binding
                .relative_path("gs://bucket/table/metadata/m.avro")
                .is_err()
        );
        assert!(
            binding
                .relative_path("s3://other/table/metadata/m.avro")
                .is_err()
        );
    }

    #[test]
    fn remote_binding_rejects_parent_traversal() {
        let options = FilesystemReadOptions::parse(remote_options("bucket")).unwrap();
        let binding =
            IcebergUriBinding::new("s3://bucket/table/metadata/v1.json", &options).unwrap();

        for uri in [
            "s3://bucket/table/../secret.avro",
            "s3://bucket/table/%2E%2E/secret.avro",
            "s3://bucket/table%2F%2E%2E%2Fsecret.avro",
        ] {
            assert!(binding.relative_path(uri).is_err());
        }
    }

    #[test]
    fn remote_binding_normalizes_current_directory_segments() {
        let options = FilesystemReadOptions::parse(remote_options("bucket")).unwrap();
        let binding =
            IcebergUriBinding::new("s3://bucket/table/metadata/v1.json", &options).unwrap();

        for uri in [
            "s3://bucket/table/./metadata/m.avro",
            "s3://bucket/table/%2E/metadata/m.avro",
        ] {
            assert_eq!(
                binding.relative_path(uri).unwrap(),
                "table/metadata/m.avro"
            );
        }
    }

    #[test]
    fn remote_binding_rejects_query_and_fragment() {
        let options = FilesystemReadOptions::parse(remote_options("bucket")).unwrap();
        let binding =
            IcebergUriBinding::new("s3://bucket/table/metadata/v1.json", &options).unwrap();

        for uri in [
            "s3://bucket/table/metadata/m.avro?version=1",
            "s3://bucket/table/metadata/m.avro#manifest",
        ] {
            assert!(binding.relative_path(uri).is_err());
        }
    }

    #[test]
    fn remote_binding_rejects_non_azure_user_information() {
        let options = FilesystemReadOptions::parse(remote_options("bucket")).unwrap();
        let binding =
            IcebergUriBinding::new("s3://bucket/table/metadata/v1.json", &options).unwrap();

        assert!(
            binding
                .relative_path("s3://user@bucket/table/metadata/m.avro")
                .is_err()
        );
    }

    #[test]
    fn azure_binding_accepts_simple_and_recorded_authorities() {
        let options = FilesystemReadOptions::parse(remote_options("container")).unwrap();
        let binding =
            IcebergUriBinding::new("abfss://container/table/metadata/v1.json", &options).unwrap();

        for scheme in ["azure", "abfs", "abfss", "wasb", "wasbs"] {
            let uri = format!(
                "{scheme}://container@account.blob.core.windows.net/table/metadata/m.avro"
            );
            assert_eq!(
                binding.relative_path(&uri).unwrap(),
                "table/metadata/m.avro"
            );
        }
    }

    #[test]
    fn local_binding_rejects_paths_outside_root() {
        let options = FilesystemReadOptions::parse(local_options("/data/root")).unwrap();
        let binding =
            IcebergUriBinding::new("/data/root/table/metadata/v1.json", &options).unwrap();

        assert_eq!(
            binding
                .relative_path("file:///data/root/table/metadata/m.avro")
                .unwrap(),
            "table/metadata/m.avro"
        );
        assert!(binding.relative_path("/data/other/m.avro").is_err());
        assert!(binding.relative_path("../escape.avro").is_err());
    }

    #[test]
    fn local_binding_rejects_parent_traversal_in_file_uri() {
        let options = FilesystemReadOptions::parse(local_options("/data/root")).unwrap();
        let binding =
            IcebergUriBinding::new("/data/root/table/metadata/v1.json", &options).unwrap();

        for uri in [
            "file:///data/root/table/../secret.avro",
            "file:///data/root/table/%2E%2E/secret.avro",
            "file:///data/root/table%2F%2E%2E%2Fsecret.avro",
        ] {
            assert!(binding.relative_path(uri).is_err());
        }
    }

    #[test]
    fn local_binding_normalizes_current_directory_segments() {
        let options = FilesystemReadOptions::parse(local_options("/data/root")).unwrap();
        let binding =
            IcebergUriBinding::new("/data/root/table/metadata/v1.json", &options).unwrap();

        for uri in [
            "file:///data/root/table/./metadata/m.avro",
            "file:///data/root/table/%2E/metadata/m.avro",
        ] {
            assert_eq!(
                binding.relative_path(uri).unwrap(),
                "table/metadata/m.avro"
            );
        }
    }

    #[test]
    fn local_binding_accepts_relative_paths() {
        let options = FilesystemReadOptions::parse(local_options("/data/root")).unwrap();
        let binding =
            IcebergUriBinding::new("/data/root/table/metadata/v1.json", &options).unwrap();

        assert_eq!(
            binding.relative_path("table/metadata/m.avro").unwrap(),
            "table/metadata/m.avro"
        );
    }

    #[tokio::test]
    async fn iceberg_storage_rejects_mutations_without_mapping_the_path() {
        let options = FilesystemReadOptions::parse(local_options("/data/root")).unwrap();
        let binding = IcebergUriBinding::new("/data/root/metadata/v1.json", &options).unwrap();
        let storage = IcebergFilesystemStorage {
            binding,
            operator: None,
        };

        let write_error = iceberg::io::Storage::write(
            &storage,
            "/outside/root",
            Bytes::from_static(b"data"),
        )
        .await
        .unwrap_err();
        let output_error = iceberg::io::Storage::new_output(&storage, "/outside/root").unwrap_err();

        assert_eq!(write_error.kind(), ErrorKind::FeatureUnsupported);
        assert_eq!(output_error.kind(), ErrorKind::FeatureUnsupported);
    }
}
