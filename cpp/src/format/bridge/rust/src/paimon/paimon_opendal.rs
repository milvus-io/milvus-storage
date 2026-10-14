// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Paimon uses OpenDAL 0.58, while the shared filesystem adapter uses 0.56.
//! Adapt the read-only interfaces; credentials and retries remain owned by C++.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use opendal_paimon::raw::*;
use opendal_paimon::{
    BytesRange, Capability, EntryMode, Error, ErrorKind, Metadata, OperationContext, Operator,
    Result,
};
use tokio::sync::OnceCell;

use crate::filesystem_opendal::FilesystemAccess;
use crate::iceberg_opendal::{FilesystemReadOptions, SchemeFamily};
use crate::paimon_ffi::PaimonFileSystemWrapper;

fn translate_error(error: opendal::Error) -> Error {
    let kind = match error.kind() {
        opendal::ErrorKind::NotFound => ErrorKind::NotFound,
        opendal::ErrorKind::PermissionDenied => ErrorKind::PermissionDenied,
        opendal::ErrorKind::Unsupported => ErrorKind::Unsupported,
        opendal::ErrorKind::ConfigInvalid => ErrorKind::ConfigInvalid,
        opendal::ErrorKind::RateLimited => ErrorKind::RateLimited,
        opendal::ErrorKind::ConditionNotMatch => ErrorKind::ConditionNotMatch,
        opendal::ErrorKind::RangeNotSatisfied => ErrorKind::RangeNotSatisfied,
        opendal::ErrorKind::IsADirectory => ErrorKind::IsADirectory,
        opendal::ErrorKind::NotADirectory => ErrorKind::NotADirectory,
        _ => ErrorKind::Unexpected,
    };
    let temporary = error.is_temporary();
    let result = Error::new(kind, "filesystem-backed I/O failed").set_source(error);
    if temporary {
        result.set_temporary()
    } else {
        result
    }
}

fn metadata(value: &opendal::Metadata) -> Result<Metadata> {
    let mode = if value.mode().is_dir() {
        EntryMode::DIR
    } else {
        EntryMode::FILE
    };
    let mut result = Metadata::new(mode);
    result.set_content_length(value.content_length());
    if let Some(modified) = value.last_modified() {
        let modified = modified.into_inner();
        result.set_last_modified(Timestamp::new(
            modified.as_second(),
            modified.subsec_nanosecond(),
        )?);
    }
    Ok(result)
}

#[derive(Debug)]
struct FilesystemService(opendal::Operator);

impl Service for FilesystemService {
    type Reader = oio::StreamReader<FilesystemReader>;
    type Writer = ();
    type Lister = FilesystemLister;
    type Deleter = ();
    type Copier = ();

    fn info(&self) -> ServiceInfo {
        ServiceInfo::new("milvus-filesystem", "/", "milvus-filesystem")
    }

    fn capability(&self) -> Capability {
        Capability {
            stat: true,
            read: true,
            list: true,
            list_with_recursive: true,
            ..Default::default()
        }
    }

    async fn stat(&self, _: &OperationContext, path: &str, _: OpStat) -> Result<RpStat> {
        let value = self.0.stat(path).await.map_err(translate_error)?;
        Ok(RpStat::new(metadata(&value)?))
    }

    fn read(&self, _: &OperationContext, path: &str, _: OpRead) -> Result<Self::Reader> {
        Ok(oio::StreamReader::new(FilesystemReader {
            operator: self.0.clone(),
            path: path.to_string(),
            reader: OnceCell::new(),
        }))
    }

    fn list(&self, _: &OperationContext, path: &str, args: OpList) -> Result<Self::Lister> {
        Ok(FilesystemLister {
            operator: self.0.clone(),
            path: path.to_string(),
            recursive: args.recursive(),
            entries: None,
        })
    }

    async fn create_dir(
        &self,
        _: &OperationContext,
        _: &str,
        _: OpCreateDir,
    ) -> Result<RpCreateDir> {
        Err(read_only())
    }

    fn write(&self, _: &OperationContext, _: &str, _: OpWrite) -> Result<Self::Writer> {
        Err(read_only())
    }

    fn delete(&self, _: &OperationContext) -> Result<Self::Deleter> {
        Err(read_only())
    }

    fn copy(
        &self,
        _: &OperationContext,
        _: &str,
        _: &str,
        _: OpCopy,
        _: OpCopier,
    ) -> Result<Self::Copier> {
        Err(read_only())
    }

    async fn rename(
        &self,
        _: &OperationContext,
        _: &str,
        _: &str,
        _: OpRename,
    ) -> Result<RpRename> {
        Err(read_only())
    }

    async fn presign(&self, _: &OperationContext, _: &str, _: OpPresign) -> Result<RpPresign> {
        Err(read_only())
    }
}

fn read_only() -> Error {
    Error::new(
        ErrorKind::Unsupported,
        "Paimon filesystem adapter is read-only",
    )
}

struct FilesystemReader {
    operator: opendal::Operator,
    path: String,
    reader: OnceCell<opendal::Reader>,
}

impl oio::StreamRead for FilesystemReader {
    async fn open(&self, range: BytesRange) -> Result<(RpRead, Box<dyn oio::ReadStreamDyn>)> {
        let reader = self
            .reader
            .get_or_try_init(|| async {
                self.operator
                    .reader(&self.path)
                    .await
                    .map_err(translate_error)
            })
            .await?;
        let content = match range.size() {
            Some(size) => {
                let end = range.offset().checked_add(size).ok_or_else(|| {
                    Error::new(ErrorKind::RangeNotSatisfied, "range end overflow")
                })?;
                reader.read(range.offset()..end).await
            }
            None => reader.read(range.offset()..).await,
        }
        .map_err(translate_error)?;
        // Both versions use bytes::Bytes; retain the chunks without copying.
        let content: opendal_paimon::Buffer = content.into_iter().collect();
        Ok((RpRead::default(), Box::new(content)))
    }
}

struct FilesystemLister {
    operator: opendal::Operator,
    path: String,
    recursive: bool,
    entries: Option<std::vec::IntoIter<opendal::Entry>>,
}

impl oio::List for FilesystemLister {
    async fn next(&mut self) -> Result<Option<oio::Entry>> {
        if self.entries.is_none() {
            self.entries = Some(
                self.operator
                    .list_with(&self.path)
                    .recursive(self.recursive)
                    .await
                    .map_err(translate_error)?
                    .into_iter(),
            );
        }
        self.entries
            .as_mut()
            .unwrap()
            .next()
            .map(|entry| Ok(oio::Entry::new(entry.path(), metadata(entry.metadata())?)))
            .transpose()
    }
}

#[derive(Debug)]
struct BoundOperator {
    operator: Operator,
    binding: PaimonUriBinding,
}

fn invalid_binding(error: impl std::fmt::Display) -> paimon::Error {
    paimon::Error::ConfigInvalid {
        message: format!("invalid Paimon filesystem binding: {error}"),
    }
}

#[derive(Clone, Debug)]
pub(crate) enum PaimonUriBinding {
    Local {
        root: PathBuf,
    },
    Remote {
        family: SchemeFamily,
        bucket: String,
    },
}

// Paimon partition paths contain literal escapes such as `p=a%2Fb`. Match
// paimon-rust's bucket_and_relative_path: parse the identity, but take the key
// from the original string. FileIO::list_status also requires this byte suffix.
fn remote_path(uri: &str) -> paimon::Result<(SchemeFamily, &str, &str)> {
    let url = url::Url::parse(uri).map_err(invalid_binding)?;
    let family = match url.scheme() {
        "s3" | "s3a" => SchemeFamily::S3,
        "gs" | "gcs" => SchemeFamily::Gcs,
        "oss" => SchemeFamily::Oss,
        "azure" | "az" | "azdfs" | "azdls" | "abfs" | "abfss" | "wasb" | "wasbs" => {
            SchemeFamily::Azure
        }
        _ => return Err(invalid_binding("unsupported URI scheme")),
    };
    if url.password().is_some() {
        return Err(invalid_binding("URI must not contain a password"));
    }
    let (authority, path) = uri
        .split_once("://")
        .map(|(_, rest)| rest.split_once('/').unwrap_or((rest, "")))
        .ok_or_else(|| invalid_binding("URI must contain an authority"))?;
    let (bucket, path) = if family == SchemeFamily::Azure {
        if let Some((container, _)) = authority.split_once('@') {
            (container, path)
        } else if authority.contains('.') {
            // Paimon also accepts account-host/container/key Azure paths.
            path.split_once('/').unwrap_or((path, ""))
        } else {
            (authority, path)
        }
    } else {
        if authority.contains('@') {
            return Err(invalid_binding(
                "non-Azure URI must not contain user information",
            ));
        }
        (authority, path)
    };
    if bucket.is_empty() || url.host_str().is_none() {
        return Err(invalid_binding("missing bucket or container"));
    }
    // Unlike from_url_path, parse validates segments without percent decoding.
    object_store::path::Path::parse(path).map_err(invalid_binding)?;
    Ok((family, bucket, path))
}

impl PaimonUriBinding {
    fn new(location: &str, options: &FilesystemReadOptions) -> paimon::Result<Self> {
        let binding = match options {
            FilesystemReadOptions::Local { root } => Self::Local { root: root.clone() },
            FilesystemReadOptions::Remote { bucket } => {
                let (family, _, _) = remote_path(location)?;
                Self::Remote {
                    family,
                    bucket: bucket.clone(),
                }
            }
        };
        binding.relative_path(location)?;
        Ok(binding)
    }

    pub(crate) fn relative_path(&self, uri: &str) -> paimon::Result<String> {
        let path = match self {
            Self::Remote { family, bucket } => {
                let (candidate_family, candidate_bucket, path) = remote_path(uri)?;
                if candidate_family != *family || candidate_bucket != bucket {
                    return Err(invalid_binding("URI does not match the bound filesystem"));
                }
                path
            }
            Self::Local { root } => {
                // These are Paimon Path strings, not percent-encoded file URLs.
                let path = if let Some(path) = uri.strip_prefix("file://") {
                    if !path.starts_with('/') {
                        return Err(invalid_binding(
                            "local file URI must not contain an authority",
                        ));
                    }
                    path
                } else if let Some(path) = uri.strip_prefix("file:") {
                    if !path.starts_with('/') {
                        return Err(invalid_binding("local file URI must be absolute"));
                    }
                    path
                } else if uri.contains("://") {
                    return Err(invalid_binding("local filesystem requires a local path"));
                } else {
                    uri
                };
                let relative = if path.starts_with('/') {
                    let root = root
                        .to_str()
                        .ok_or_else(|| invalid_binding("local root is not UTF-8"))?;
                    path.strip_prefix(root.trim_end_matches('/'))
                        .filter(|suffix| suffix.is_empty() || suffix.starts_with('/'))
                        .ok_or_else(|| {
                            invalid_binding("path is outside the bound filesystem root")
                        })?
                        .trim_start_matches('/')
                } else {
                    path
                };
                object_store::path::Path::parse(relative).map_err(invalid_binding)?;
                relative
            }
        };
        Ok(path.to_string())
    }
}

#[async_trait::async_trait]
impl paimon::io::FileIOProvider for BoundOperator {
    async fn create(&self, uri: &str) -> paimon::Result<(Operator, String)> {
        let path = self.binding.relative_path(uri)?;
        Ok((self.operator.clone(), path))
    }
}

pub(crate) fn file_io(
    filesystem: cxx::SharedPtr<PaimonFileSystemWrapper>,
    location: &str,
    options: HashMap<String, String>,
) -> paimon::Result<(paimon::io::FileIO, PaimonUriBinding)> {
    // Reuse Iceberg's credential-free filesystem identity options.
    let options = FilesystemReadOptions::parse(options).map_err(invalid_binding)?;
    let binding = PaimonUriBinding::new(location, &options)?;
    let access = FilesystemAccess::new(filesystem).map_err(translate_error)?;
    let operator = Operator::from_parts(
        OperationContext::default(),
        Arc::new(FilesystemService(access.into_operator())),
    );
    let file_io = paimon::io::FileIOBuilder::new("milvus-filesystem")
        .with_provider(Arc::new(BoundOperator { operator, binding: binding.clone() }))
        .build()?;
    Ok((file_io, binding))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filesystem_opendal::tests::LocalOperatorFixture;

    #[test]
    fn remote_binding_preserves_paimon_paths() {
        let options = FilesystemReadOptions::Remote {
            bucket: "bucket".to_string(),
        };
        for prefix in [
            "s3://bucket",
            "s3a://bucket",
            "oss://bucket",
            "gs://bucket",
            "gcs://bucket",
            "azure://bucket",
            "abfss://bucket@account.dfs.core.windows.net",
            "abfs://account.dfs.core.windows.net/bucket",
        ] {
            let binding = PaimonUriBinding::new(&format!("{prefix}/table"), &options).unwrap();
            assert_eq!(binding.relative_path(prefix).unwrap(), "");
            assert_eq!(binding.relative_path(&format!("{prefix}/")).unwrap(), "");
            for key in [
                "table/p=a%2Fb/data.parquet",
                "table/p=a%25b/data.parquet",
                "table/p=a%3Db/data.parquet",
                "table/p=%252F/data.parquet",
                "table/p=中文 空格/data.parquet",
                "table/p=a?b#c/data.parquet",
                "table/%2e%2e/data.parquet",
                "table/p=a%2Fb/",
            ] {
                let uri = format!("{prefix}/{key}");
                let relative = binding.relative_path(&uri).unwrap();
                assert_eq!(relative, key, "{uri}");
                assert!(uri.ends_with(&relative));
            }
        }
    }

    #[test]
    fn remote_binding_checks_identity_and_path() {
        let options = FilesystemReadOptions::Remote {
            bucket: "bucket".to_string(),
        };
        let binding = PaimonUriBinding::new("s3://bucket/table", &options).unwrap();
        assert_eq!(
            binding.relative_path("s3a://bucket/table/schema").unwrap(),
            "table/schema"
        );
        assert!(PaimonUriBinding::new("s3://other/table", &options).is_err());
        for uri in [
            "s3://other/table",
            "gs://bucket/table",
            "s3://user@bucket/table",
            "s3://bucket/table/../schema",
            "s3://bucket/table/./schema",
            "s3:///table",
            "https://bucket/table",
            "file:///table",
        ] {
            assert!(binding.relative_path(uri).is_err(), "{uri}");
        }
        let binding = PaimonUriBinding::new(
            "abfss://bucket@account.dfs.core.windows.net/table",
            &options,
        )
        .unwrap();
        for uri in [
            "abfss://other@account.dfs.core.windows.net/table",
            "abfss://account.dfs.core.windows.net/other/table",
            "abfss://bucket:secret@account.dfs.core.windows.net/table",
        ] {
            assert!(binding.relative_path(uri).is_err(), "{uri}");
        }
    }

    #[test]
    fn local_binding_preserves_paths_under_root() {
        let options = FilesystemReadOptions::Local {
            root: PathBuf::from("/data/root"),
        };
        let binding = PaimonUriBinding::new("file:/data/root/table", &options).unwrap();
        for prefix in ["/data/root/", "file:/data/root/", "file:///data/root/", ""] {
            for key in [
                "table/p=a%2Fb%25c%3Dd%252Fe/中文 空格.parquet",
                "table/p=a%2Fb/",
            ] {
                let uri = format!("{prefix}{key}");
                let relative = binding.relative_path(&uri).unwrap();
                assert_eq!(relative, key, "{uri}");
                assert!(uri.ends_with(&relative));
                // Mirror FileIO's list prefix recovery and map a returned entry.
                let base = &uri[..uri.len() - relative.len()];
                let listed = format!("{base}table/p=a%2Fb/schema");
                assert_eq!(
                    binding.relative_path(&listed).unwrap(),
                    "table/p=a%2Fb/schema"
                );
            }
        }
        assert_eq!(binding.relative_path("file:///data/root/").unwrap(), "");
        for uri in [
            "/data/other/table",
            "/data/root-other/table",
            "../table",
            "table/../schema",
            "file:///data/root/../table",
            "file://host/data/root/table",
            "file:relative",
            "s3://bucket/table",
        ] {
            assert!(binding.relative_path(uri).is_err(), "{uri}");
        }
    }

    #[tokio::test]
    async fn listed_paimon_paths_can_be_read_back() {
        let fixture = LocalOperatorFixture::new();
        let root = &fixture.root;
        let key = "table/p=a%2Fb%25c%3Dd%252Fe/中文 空格/data";
        let path = root.path().join(key);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"paimon-path").unwrap();
        for (prefix, options) in [
            (
                "s3://bucket".to_string(),
                FilesystemReadOptions::Remote {
                    bucket: "bucket".to_string(),
                },
            ),
            (
                format!("file://{}", root.path().display()),
                FilesystemReadOptions::Local {
                    root: root.path().to_path_buf(),
                },
            ),
        ] {
            let operator = Operator::from_parts(
                OperationContext::default(),
                Arc::new(FilesystemService(fixture.operator.clone())),
            );
            let binding = PaimonUriBinding::new(&format!("{prefix}/table"), &options).unwrap();
            let file_io = paimon::io::FileIOBuilder::new("milvus-filesystem")
                .with_provider(Arc::new(BoundOperator { operator, binding }))
                .build()
                .unwrap();
            let directory = format!("{prefix}/{}", key.rsplit_once('/').unwrap().0);
            let listed = file_io.list_status(&directory).await.unwrap();
            let recursive = file_io
                .list_status_recursive(&format!("{prefix}/table"))
                .await
                .unwrap();
            for entries in [listed, recursive] {
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].path, format!("{prefix}/{key}"));
                assert_eq!(entries[0].size, 11);
                assert!(!entries[0].is_dir);
                let input = file_io.new_input(&entries[0].path).unwrap();
                assert_eq!(input.read().await.unwrap().as_ref(), b"paimon-path");
            }
        }
    }

    #[tokio::test]
    async fn filesystem_adapter_preserves_read_boundaries_and_errors() {
        let fixture = LocalOperatorFixture::new();
        std::fs::write(fixture.root.path().join("payload"), b"0123456789").unwrap();
        std::fs::write(fixture.root.path().join("empty"), b"").unwrap();
        let operator = Operator::from_parts(
            OperationContext::default(),
            Arc::new(FilesystemService(fixture.operator.clone())),
        );
        let info = operator.stat("payload").await.unwrap();
        assert!(info.mode().is_file());
        assert_eq!(info.content_length(), 10);
        let reader = operator.reader("payload").await.unwrap();
        for (range, expected) in [(2..6, b"2345".as_slice()), (8..10, b"89"), (10..10, b"")] {
            assert_eq!(reader.read(range).await.unwrap().to_bytes().as_ref(), expected);
        }
        assert!(operator.read("empty").await.unwrap().is_empty());
        // The shared OpenDAL layer must not silently return a truncated range.
        assert!(reader.read(8..12).await.is_err());
        assert_eq!(reader.read(11..12).await.unwrap_err().kind(), ErrorKind::RangeNotSatisfied);
        for error in [
            operator.stat("missing").await.unwrap_err(),
            operator.read("missing").await.unwrap_err(),
        ] {
            assert_eq!(error.kind(), ErrorKind::NotFound);
            assert!(format!("{error:?}").contains(crate::filesystem_c::LOON_FFI_ERRCODE_MARKER));
        }
    }

    #[test]
    fn errors_preserve_category_temporary_and_source() {
        for (old, new) in [
            (opendal::ErrorKind::NotFound, ErrorKind::NotFound),
            (
                opendal::ErrorKind::PermissionDenied,
                ErrorKind::PermissionDenied,
            ),
            (opendal::ErrorKind::RateLimited, ErrorKind::RateLimited),
            (
                opendal::ErrorKind::ConditionNotMatch,
                ErrorKind::ConditionNotMatch,
            ),
            (
                opendal::ErrorKind::RangeNotSatisfied,
                ErrorKind::RangeNotSatisfied,
            ),
        ] {
            let result = translate_error(opendal::Error::new(old, "source-detail").set_temporary());
            assert_eq!(result.kind(), new);
            assert!(result.is_temporary());
            assert!(format!("{result:?}").contains("source-detail"));
        }
    }
}
