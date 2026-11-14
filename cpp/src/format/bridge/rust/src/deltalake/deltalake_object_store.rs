// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Kernel uses absolute file-URL paths, while our C++ filesystem is already
//! rooted. Keep that translation private to Delta so Lance's path contract is
//! unchanged. All actual I/O still uses the shared filesystem adapter.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Result, bail, ensure};
use async_trait::async_trait;
use cxx::SharedPtr;
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use url::Url;

use crate::deltalake_ffi::{DeltaFilesystemBinding, DeltaStorageKind};
use crate::filesystem_object_store::{FFIPathMapper, FilesystemObjectStore};
use crate::lance_ffi::FileSystemWrapper;

#[derive(Clone, Debug)]
pub(crate) struct DeltaPathBinding {
    mapper: FFIPathMapper,
    local_prefix: Option<Path>,
}

impl DeltaPathBinding {
    pub(crate) fn new(options: &DeltaFilesystemBinding, table_url: &Url) -> Result<Self> {
        let (mapper, local_prefix) = match options.kind {
            DeltaStorageKind::Local => {
                ensure!(
                    table_url.scheme() == "file",
                    "local Delta table must use a file URL"
                );
                let root = PathBuf::from(&options.local_root);
                ensure!(root.is_absolute(), "Delta filesystem root must be absolute");
                let prefix = Path::from_absolute_path(&root)?;
                (FFIPathMapper::local(root), Some(prefix))
            }
            DeltaStorageKind::Remote => {
                ensure!(
                    !options.authority.is_empty(),
                    "Delta filesystem bucket is empty"
                );
                (
                    FFIPathMapper::remote(&options.scheme, &options.authority),
                    None,
                )
            }
            _ => bail!("invalid Delta filesystem kind"),
        };
        let binding = Self {
            mapper,
            local_prefix,
        };
        binding.validate_url(table_url)?;
        Ok(binding)
    }

    pub(crate) fn validate_url(&self, url: &Url) -> Result<()> {
        ensure!(
            url.query().is_none()
                && url.fragment().is_none()
                && url.username().is_empty()
                && url.password().is_none()
                && url.port().is_none(),
            "Delta source URL has an unsupported authority, query, or fragment: {url}"
        );
        self.mapper.from_url(url)?;
        Ok(())
    }

    fn to_inner(&self, path: &Path) -> object_store::Result<Path> {
        let Some(prefix) = &self.local_prefix else {
            return Ok(path.clone());
        };
        if prefix.as_ref().is_empty() {
            return Ok(path.clone());
        }
        if path == prefix {
            return Ok(Path::ROOT);
        }
        let relative = path
            .as_ref()
            .strip_prefix(prefix.as_ref())
            .and_then(|tail| tail.strip_prefix('/'))
            .ok_or_else(|| object_store::Error::Generic {
                store: "delta-filesystem",
                source: "Kernel path is outside the bound local filesystem".into(),
            })?;
        Path::parse(relative).map_err(Into::into)
    }

    fn to_outer(&self, path: &Path) -> object_store::Result<Path> {
        match &self.local_prefix {
            Some(prefix) if !prefix.as_ref().is_empty() => {
                Path::parse(format!("{prefix}/{path}")).map_err(Into::into)
            }
            _ => Ok(path.clone()),
        }
    }

    fn outer_meta(&self, mut meta: ObjectMeta) -> object_store::Result<ObjectMeta> {
        meta.location = self.to_outer(&meta.location)?;
        Ok(meta)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DeltaObjectStore {
    inner: FilesystemObjectStore,
    binding: DeltaPathBinding,
}

impl DeltaObjectStore {
    pub(crate) fn new(filesystem: SharedPtr<FileSystemWrapper>, binding: DeltaPathBinding) -> Self {
        let inner = FilesystemObjectStore::new(
            filesystem,
            binding.mapper.clone(),
            Arc::new(AtomicBool::new(false)),
        );
        Self { inner, binding }
    }

    fn read_only<T>(operation: &str) -> object_store::Result<T> {
        Err(object_store::Error::NotImplemented {
            operation: operation.into(),
            implementer: "DeltaObjectStore".into(),
        })
    }
}

impl fmt::Display for DeltaObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DeltaObjectStore(filesystem_c)")
    }
}

#[async_trait]
impl ObjectStore for DeltaObjectStore {
    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let mut result = self
            .inner
            .get_opts(&self.binding.to_inner(location)?, options)
            .await?;
        result.meta = self.binding.outer_meta(result.meta)?;
        Ok(result)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        let prefix = match prefix.map(|path| self.binding.to_inner(path)).transpose() {
            Ok(prefix) => prefix,
            Err(error) => return stream::once(async { Err(error) }).boxed(),
        };
        let binding = self.binding.clone();
        self.inner
            .list(prefix.as_ref())
            .map(move |meta| binding.outer_meta(meta?))
            .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        let prefix = prefix.map(|path| self.binding.to_inner(path)).transpose()?;
        let result = self.inner.list_with_delimiter(prefix.as_ref()).await?;
        Ok(ListResult {
            objects: result
                .objects
                .into_iter()
                .map(|meta| self.binding.outer_meta(meta))
                .collect::<Result<_, _>>()?,
            common_prefixes: result
                .common_prefixes
                .iter()
                .map(|path| self.binding.to_outer(path))
                .collect::<Result<_, _>>()?,
        })
    }

    async fn put_opts(
        &self,
        _: &Path,
        _: PutPayload,
        _: PutOptions,
    ) -> object_store::Result<PutResult> {
        Self::read_only("put_opts")
    }

    async fn put_multipart_opts(
        &self,
        _: &Path,
        _: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        Self::read_only("put_multipart_opts")
    }

    fn delete_stream(
        &self,
        _: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        stream::once(async { Self::read_only("delete_stream") }).boxed()
    }

    async fn copy_opts(&self, _: &Path, _: &Path, _: CopyOptions) -> object_store::Result<()> {
        Self::read_only("copy_opts")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_binding() -> DeltaPathBinding {
        DeltaPathBinding {
            mapper: FFIPathMapper::local("/tmp/table%2F"),
            local_prefix: Some(Path::from_absolute_path("/tmp/table%2F").unwrap()),
        }
    }

    #[test]
    fn local_paths_round_trip_without_decoding_literal_percent() {
        let binding = local_binding();
        let url = Url::parse("file:///tmp/table%252F/p%252F/file.parquet").unwrap();
        binding.validate_url(&url).unwrap();
        let outer = Path::from_absolute_path(url.to_file_path().unwrap()).unwrap();
        let inner = binding.to_inner(&outer).unwrap();
        assert_eq!(inner.as_ref(), "p%2F/file.parquet");
        assert_eq!(binding.to_outer(&inner).unwrap(), outer);
    }

    #[test]
    fn local_binding_rejects_sibling_roots() {
        assert!(
            local_binding()
                .to_inner(&Path::parse("tmp/table%2F-other/file").unwrap())
                .is_err()
        );
    }

    #[test]
    fn remote_binding_preserves_keys_and_rejects_other_filesystems() {
        let binding = DeltaPathBinding {
            mapper: FFIPathMapper::remote("s3", "bucket"),
            local_prefix: None,
        };
        let url = Url::parse("s3://bucket/table/p%252F/file.parquet").unwrap();
        binding.validate_url(&url).unwrap();
        let path = Path::from_url_path(url.path()).unwrap();
        assert_eq!(
            binding.to_inner(&path).unwrap().as_ref(),
            "table/p%2F/file.parquet"
        );
        for invalid in [
            "s3://other/table/file.parquet",
            "gs://bucket/table/file.parquet",
            "s3://bucket/table/file.parquet?versionId=1",
        ] {
            assert!(binding.validate_url(&Url::parse(invalid).unwrap()).is_err());
        }
    }
}
