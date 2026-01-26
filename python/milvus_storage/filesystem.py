"""
Filesystem classes for milvus-storage.

Provides access to local and cloud storage backends (S3, GCS, Azure, etc.).
"""

from typing import Dict, List, Optional, Tuple

from ._ffi import check_result, get_ffi, get_library
from .exceptions import InvalidArgumentError, ResourceError
from .properties import Properties


class FileInfo:
    """
    Information about a file or directory.

    Attributes:
        path: File or directory path
        is_dir: True if this is a directory
        size: File size in bytes (0 for directories)
        mtime_ns: Modification time in nanoseconds since epoch
    """

    def __init__(self, path: str, is_dir: bool, size: int, mtime_ns: int):
        self.path = path
        self.is_dir = is_dir
        self.size = size
        self.mtime_ns = mtime_ns

    def __repr__(self) -> str:
        type_str = "dir" if self.is_dir else "file"
        return f"FileInfo(path={self.path!r}, type={type_str}, size={self.size})"


class FilesystemMetrics:
    """
    Filesystem operation metrics snapshot.

    Contains counters for various filesystem operations.

    Attributes:
        read_count: Number of read operations
        write_count: Number of write operations
        read_bytes: Total bytes read
        write_bytes: Total bytes written
        get_file_info_count: Number of file info queries
        create_dir_count: Number of directory creations
        delete_dir_count: Number of directory deletions
        delete_file_count: Number of file deletions
        move_count: Number of move operations
        copy_file_count: Number of copy operations
        failed_count: Number of failed operations
        multi_part_upload_created: Number of multipart uploads created (S3)
        multi_part_upload_finished: Number of multipart uploads completed (S3)
    """

    def __init__(
        self,
        read_count: int = 0,
        write_count: int = 0,
        read_bytes: int = 0,
        write_bytes: int = 0,
        get_file_info_count: int = 0,
        create_dir_count: int = 0,
        delete_dir_count: int = 0,
        delete_file_count: int = 0,
        move_count: int = 0,
        copy_file_count: int = 0,
        failed_count: int = 0,
        multi_part_upload_created: int = 0,
        multi_part_upload_finished: int = 0,
    ):
        self.read_count = read_count
        self.write_count = write_count
        self.read_bytes = read_bytes
        self.write_bytes = write_bytes
        self.get_file_info_count = get_file_info_count
        self.create_dir_count = create_dir_count
        self.delete_dir_count = delete_dir_count
        self.delete_file_count = delete_file_count
        self.move_count = move_count
        self.copy_file_count = copy_file_count
        self.failed_count = failed_count
        self.multi_part_upload_created = multi_part_upload_created
        self.multi_part_upload_finished = multi_part_upload_finished

    @classmethod
    def _from_c(cls, c_metrics) -> "FilesystemMetrics":
        """Create FilesystemMetrics from C structure."""
        return cls(
            read_count=c_metrics.read_count,
            write_count=c_metrics.write_count,
            read_bytes=c_metrics.read_bytes,
            write_bytes=c_metrics.write_bytes,
            get_file_info_count=c_metrics.get_file_info_count,
            create_dir_count=c_metrics.create_dir_count,
            delete_dir_count=c_metrics.delete_dir_count,
            delete_file_count=c_metrics.delete_file_count,
            move_count=c_metrics.move_count,
            copy_file_count=c_metrics.copy_file_count,
            failed_count=c_metrics.failed_count,
            multi_part_upload_created=c_metrics.multi_part_upload_created,
            multi_part_upload_finished=c_metrics.multi_part_upload_finished,
        )

    def __repr__(self) -> str:
        return (
            f"FilesystemMetrics(reads={self.read_count}, writes={self.write_count}, "
            f"read_bytes={self.read_bytes}, write_bytes={self.write_bytes})"
        )


class FilesystemWriter:
    """
    Output stream for writing to a file.

    Provides buffered writes to a filesystem path.

    Example:
        >>> fs = Filesystem.get()
        >>> with fs.open_writer("/path/to/file") as writer:
        ...     writer.write(b"Hello, World!")
    """

    def __init__(self, handle, lib, ffi):
        """
        Initialize FilesystemWriter (internal use).

        Args:
            handle: C handle to output stream
            lib: Library instance
            ffi: FFI instance
        """
        self._handle = handle
        self._lib = lib
        self._ffi = ffi
        self._closed = False

    def write(self, data: bytes) -> None:
        """
        Write data to the file.

        Args:
            data: Bytes to write

        Raises:
            ResourceError: If writer is closed
            FFIError: If write fails
        """
        if self._closed:
            raise ResourceError("FilesystemWriter is closed")

        if not data:
            return

        data_ptr = self._ffi.from_buffer(data)
        result = self._lib.loon_filesystem_writer_write(self._handle, data_ptr, len(data))
        check_result(result)

    def flush(self) -> None:
        """
        Flush buffered data to storage.

        Raises:
            ResourceError: If writer is closed
            FFIError: If flush fails
        """
        if self._closed:
            raise ResourceError("FilesystemWriter is closed")

        result = self._lib.loon_filesystem_writer_flush(self._handle)
        check_result(result)

    def close(self) -> None:
        """Close the writer and release resources."""
        if not self._closed and self._handle is not None:
            result = self._lib.loon_filesystem_writer_close(self._handle)
            check_result(result)
            self._lib.loon_filesystem_writer_destroy(self._handle)
            self._handle = None
            self._closed = True

    def __del__(self):
        """Cleanup on destruction."""
        if not self._closed and self._handle is not None:
            try:
                # Try to close properly first (flush data)
                self._lib.loon_filesystem_writer_close(self._handle)
            except Exception:
                pass
            try:
                self._lib.loon_filesystem_writer_destroy(self._handle)
            except Exception:
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class FilesystemReader:
    """
    Input stream for reading from a file.

    Provides random access reads from a filesystem path.

    Example:
        >>> fs = Filesystem.get()
        >>> with fs.open_reader("/path/to/file") as reader:
        ...     data = reader.read_at(0, 1024)  # Read first 1KB
    """

    def __init__(self, handle, lib, ffi):
        """
        Initialize FilesystemReader (internal use).

        Args:
            handle: C handle to input stream
            lib: Library instance
            ffi: FFI instance
        """
        self._handle = handle
        self._lib = lib
        self._ffi = ffi
        self._closed = False

    def read_at(self, offset: int, nbytes: int) -> bytes:
        """
        Read data from a specific offset.

        Args:
            offset: Byte offset to start reading from
            nbytes: Number of bytes to read

        Returns:
            Bytes read from the file

        Raises:
            ResourceError: If reader is closed
            InvalidArgumentError: If offset or nbytes is negative
            FFIError: If read fails
        """
        if self._closed:
            raise ResourceError("FilesystemReader is closed")

        if offset < 0:
            raise InvalidArgumentError(f"offset must be non-negative, got {offset}")
        if nbytes < 0:
            raise InvalidArgumentError(f"nbytes must be non-negative, got {nbytes}")
        if nbytes == 0:
            return b""

        buffer = self._ffi.new(f"uint8_t[{nbytes}]")
        result = self._lib.loon_filesystem_reader_readat(self._handle, offset, nbytes, buffer)
        check_result(result)

        return bytes(self._ffi.buffer(buffer, nbytes))

    def close(self) -> None:
        """Close the reader and release resources."""
        if not self._closed and self._handle is not None:
            result = self._lib.loon_filesystem_reader_close(self._handle)
            check_result(result)
            self._lib.loon_filesystem_reader_destroy(self._handle)
            self._handle = None
            self._closed = True

    def __del__(self):
        """Cleanup on destruction."""
        if not self._closed and self._handle is not None:
            try:
                # Try to close properly first
                self._lib.loon_filesystem_reader_close(self._handle)
            except Exception:
                pass
            try:
                self._lib.loon_filesystem_reader_destroy(self._handle)
            except Exception:
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class Filesystem:
    """
    Filesystem interface for storage operations.

    Provides access to local and cloud storage backends including:
    - Local filesystem
    - AWS S3
    - Google Cloud Storage
    - Azure Blob Storage
    - Aliyun OSS
    - Tencent COS
    - Huawei OBS

    Example:
        >>> from milvus_storage import Filesystem
        >>>
        >>> # Get local filesystem
        >>> fs = Filesystem.get()
        >>>
        >>> # Get S3 filesystem
        >>> fs = Filesystem.get(properties={
        ...     "fs.storage_type": "s3",
        ...     "fs.bucket_name": "my-bucket",
        ...     "fs.access_key_id": "...",
        ...     "fs.access_key_value": "...",
        ... })
        >>>
        >>> # Read/write files
        >>> fs.write_file("/path/to/file", b"data")
        >>> data = fs.read_file_all("/path/to/file")
    """

    def __init__(self, handle):
        """
        Initialize Filesystem (internal use).

        Use Filesystem.get() to obtain a filesystem instance.

        Args:
            handle: C handle to filesystem
        """
        self._lib = get_library().lib
        self._ffi = get_ffi()
        self._handle = handle
        self._closed = False

    @classmethod
    def get(
        cls,
        properties: Optional[Dict[str, str]] = None,
        path: Optional[str] = None,
    ) -> "Filesystem":
        """
        Get a filesystem instance from cache or create a new one.

        Uses a cache internally to reuse filesystem instances with the same
        configuration.

        Args:
            properties: Optional configuration properties (e.g., S3 credentials)
            path: Optional path to determine filesystem type. If path has a
                  scheme (e.g., "s3://bucket/key"), it will resolve the
                  appropriate filesystem.

        Returns:
            Filesystem instance

        Raises:
            FFIError: If filesystem creation fails
        """
        lib = get_library().lib
        ffi = get_ffi()

        props = Properties(properties) if properties else Properties()

        path_bytes = path.encode("utf-8") if path else ffi.NULL
        path_len = len(path) if path else 0

        handle = ffi.new("FileSystemHandle*")
        result = lib.loon_filesystem_get(props._get_c_properties(), path_bytes, path_len, handle)
        check_result(result)

        return cls(handle[0])

    @staticmethod
    def close_all() -> None:
        """
        Close all cached filesystem instances.

        Clears the internal filesystem cache. Useful for cleanup or testing.
        """
        lib = get_library().lib
        lib.loon_close_filesystems()

    def get_file_size(self, path: str) -> int:
        """
        Get the size of a file.

        Args:
            path: File path

        Returns:
            File size in bytes

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        path_bytes = path.encode("utf-8")
        size = self._ffi.new("uint64_t*")

        result = self._lib.loon_filesystem_get_file_info(
            self._handle, path_bytes, len(path_bytes), size
        )
        check_result(result)

        return size[0]

    def read_file(self, path: str, offset: int, nbytes: int) -> bytes:
        """
        Read a portion of a file.

        Args:
            path: File path
            offset: Byte offset to start reading
            nbytes: Number of bytes to read

        Returns:
            Bytes read from the file

        Raises:
            ResourceError: If filesystem is closed
            InvalidArgumentError: If offset or nbytes is negative
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        if offset < 0:
            raise InvalidArgumentError(f"offset must be non-negative, got {offset}")
        if nbytes < 0:
            raise InvalidArgumentError(f"nbytes must be non-negative, got {nbytes}")
        if nbytes == 0:
            return b""

        path_bytes = path.encode("utf-8")
        buffer = self._ffi.new(f"uint8_t[{nbytes}]")

        result = self._lib.loon_filesystem_read_file(
            self._handle, path_bytes, len(path_bytes), offset, nbytes, buffer
        )
        check_result(result)

        return bytes(self._ffi.buffer(buffer, nbytes))

    def read_file_all(self, path: str) -> bytes:
        """
        Read entire file content.

        Args:
            path: File path

        Returns:
            Complete file contents as bytes

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        path_bytes = path.encode("utf-8")
        data_ptr = self._ffi.new("uint8_t**")
        size = self._ffi.new("uint64_t*")

        result = self._lib.loon_filesystem_read_file_all(
            self._handle, path_bytes, len(path_bytes), data_ptr, size
        )
        check_result(result)

        # Copy data to Python bytes
        data = bytes(self._ffi.buffer(data_ptr[0], size[0]))

        # Free the C-allocated buffer using loon_free_cstr (internally calls free())
        self._lib.loon_free_cstr(self._ffi.cast("char*", data_ptr[0]))

        return data

    def write_file(
        self,
        path: str,
        data: bytes,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Write data to a file.

        Args:
            path: File path
            data: Data to write
            metadata: Optional key-value metadata

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        path_bytes = path.encode("utf-8")
        data_ptr = self._ffi.from_buffer(data) if data else self._ffi.NULL

        # Build metadata array
        if metadata:
            meta_array = self._ffi.new("LoonFileSystemMeta[]", len(metadata))
            key_buffers = []
            val_buffers = []
            for i, (k, v) in enumerate(metadata.items()):
                key_c = self._ffi.new("char[]", k.encode("utf-8"))
                val_c = self._ffi.new("char[]", v.encode("utf-8"))
                key_buffers.append(key_c)
                val_buffers.append(val_c)
                meta_array[i].key = key_c
                meta_array[i].value = val_c
            meta_count = len(metadata)
        else:
            meta_array = self._ffi.NULL
            meta_count = 0

        result = self._lib.loon_filesystem_write_file(
            self._handle,
            path_bytes,
            len(path_bytes),
            data_ptr,
            len(data) if data else 0,
            meta_array,
            meta_count,
        )
        check_result(result)

    def delete_file(self, path: str) -> None:
        """
        Delete a file.

        Args:
            path: File path to delete

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        path_bytes = path.encode("utf-8")
        result = self._lib.loon_filesystem_delete_file(self._handle, path_bytes, len(path_bytes))
        check_result(result)

    def get_path_info(self, path: str) -> Tuple[bool, bool, int]:
        """
        Get path information.

        Args:
            path: Path to check

        Returns:
            Tuple of (exists, is_dir, mtime_ns)

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        path_bytes = path.encode("utf-8")
        exists = self._ffi.new("bool*")
        is_dir = self._ffi.new("bool*")
        mtime_ns = self._ffi.new("int64_t*")

        result = self._lib.loon_filesystem_get_path_info(
            self._handle, path_bytes, len(path_bytes), exists, is_dir, mtime_ns
        )
        check_result(result)

        return exists[0], is_dir[0], mtime_ns[0]

    def exists(self, path: str) -> bool:
        """
        Check if a path exists.

        Args:
            path: Path to check

        Returns:
            True if path exists

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        exists, _, _ = self.get_path_info(path)
        return exists

    def is_directory(self, path: str) -> bool:
        """
        Check if a path is a directory.

        Args:
            path: Path to check

        Returns:
            True if path is a directory

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        exists, is_dir, _ = self.get_path_info(path)
        return exists and is_dir

    def create_dir(self, path: str, recursive: bool = True) -> None:
        """
        Create a directory.

        Args:
            path: Directory path to create
            recursive: If True, create parent directories as needed

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        path_bytes = path.encode("utf-8")
        result = self._lib.loon_filesystem_create_dir(
            self._handle, path_bytes, len(path_bytes), recursive
        )
        check_result(result)

    def list_dir(self, path: str, recursive: bool = False) -> List[FileInfo]:
        """
        List directory contents.

        Args:
            path: Directory path to list
            recursive: If True, list recursively

        Returns:
            List of FileInfo objects

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        path_bytes = path.encode("utf-8")
        file_list = self._ffi.new("LoonFileInfoList*")

        result = self._lib.loon_filesystem_list_dir(
            self._handle, path_bytes, len(path_bytes), recursive, file_list
        )
        check_result(result)

        # Convert to Python objects
        files = []
        for i in range(file_list.count):
            entry = file_list.entries[i]
            file_path = self._ffi.string(entry.path, entry.path_len).decode("utf-8")
            files.append(FileInfo(file_path, entry.is_dir, entry.size, entry.mtime_ns))

        # Free C list
        self._lib.loon_filesystem_free_file_info_list(file_list)

        return files

    def open_writer(
        self,
        path: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> FilesystemWriter:
        """
        Open a file for writing.

        Args:
            path: File path to write
            metadata: Optional key-value metadata

        Returns:
            FilesystemWriter for writing data

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        path_bytes = path.encode("utf-8")

        # Build metadata array
        if metadata:
            meta_array = self._ffi.new("LoonFileSystemMeta[]", len(metadata))
            key_buffers = []
            val_buffers = []
            for i, (k, v) in enumerate(metadata.items()):
                key_c = self._ffi.new("char[]", k.encode("utf-8"))
                val_c = self._ffi.new("char[]", v.encode("utf-8"))
                key_buffers.append(key_c)
                val_buffers.append(val_c)
                meta_array[i].key = key_c
                meta_array[i].value = val_c
            meta_count = len(metadata)
        else:
            meta_array = self._ffi.NULL
            meta_count = 0

        handle = self._ffi.new("FileSystemWriterHandle*")
        result = self._lib.loon_filesystem_open_writer(
            self._handle, path_bytes, len(path_bytes), meta_array, meta_count, handle
        )
        check_result(result)

        return FilesystemWriter(handle[0], self._lib, self._ffi)

    def open_reader(self, path: str) -> FilesystemReader:
        """
        Open a file for reading.

        Args:
            path: File path to read

        Returns:
            FilesystemReader for reading data

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        path_bytes = path.encode("utf-8")

        handle = self._ffi.new("FileSystemReaderHandle*")
        result = self._lib.loon_filesystem_open_reader(
            self._handle, path_bytes, len(path_bytes), handle
        )
        check_result(result)

        return FilesystemReader(handle[0], self._lib, self._ffi)

    def get_file_stats(self, path: str) -> Tuple[int, Dict[str, str]]:
        """
        Get file statistics including size and metadata.

        Args:
            path: File path

        Returns:
            Tuple of (size, metadata_dict)

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        path_bytes = path.encode("utf-8")
        size = self._ffi.new("uint64_t*")
        meta_array = self._ffi.new("LoonFileSystemMeta**")
        meta_count = self._ffi.new("uint32_t*")

        result = self._lib.loon_filesystem_get_file_stats(
            self._handle, path_bytes, len(path_bytes), size, meta_array, meta_count
        )
        check_result(result)

        # Convert metadata to dict
        metadata = {}
        for i in range(meta_count[0]):
            key = self._ffi.string(meta_array[0][i].key).decode("utf-8")
            value = self._ffi.string(meta_array[0][i].value).decode("utf-8")
            metadata[key] = value

        # Free metadata array
        if meta_count[0] > 0:
            self._lib.loon_filesystem_free_meta_array(meta_array[0], meta_count[0])

        return size[0], metadata

    def get_metrics(self) -> FilesystemMetrics:
        """
        Get filesystem operation metrics.

        Returns:
            FilesystemMetrics snapshot

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails or metrics not supported
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        metrics = self._ffi.new("LoonFilesystemMetricsSnapshot*")
        result = self._lib.loon_filesystem_get_metrics(self._handle, metrics)
        check_result(result)

        return FilesystemMetrics._from_c(metrics)

    def reset_metrics(self) -> None:
        """
        Reset filesystem operation metrics.

        Raises:
            ResourceError: If filesystem is closed
            FFIError: If operation fails or metrics not supported
        """
        if self._closed:
            raise ResourceError("Filesystem is closed")

        result = self._lib.loon_filesystem_reset_metrics(self._handle)
        check_result(result)

    def close(self) -> None:
        """Close the filesystem and release resources."""
        if not self._closed and self._handle is not None:
            self._lib.loon_filesystem_destroy(self._handle)
            self._handle = None
            self._closed = True

    def __del__(self):
        """Cleanup on destruction."""
        if not self._closed and self._handle is not None:
            try:
                self._lib.loon_filesystem_destroy(self._handle)
            except Exception:
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @property
    def is_closed(self) -> bool:
        """Check if filesystem is closed."""
        return self._closed

    def __repr__(self) -> str:
        """String representation."""
        status = "closed" if self._closed else "open"
        return f"Filesystem(status={status})"


class FilesystemSingleton:
    """
    Global filesystem singleton.

    Provides a shared filesystem instance for the application.

    Example:
        >>> from milvus_storage import FilesystemSingleton
        >>>
        >>> # Initialize with properties
        >>> FilesystemSingleton.init({"fs.storage_type": "local"})
        >>>
        >>> # Get the singleton
        >>> fs = FilesystemSingleton.get()
    """

    _fs: Optional[Filesystem] = None

    @classmethod
    def init(cls, properties: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize the filesystem singleton.

        Args:
            properties: Configuration properties

        Raises:
            FFIError: If initialization fails
        """
        lib = get_library().lib

        props = Properties(properties) if properties else Properties()

        result = lib.loon_initialize_filesystem_singleton(props._get_c_properties())
        check_result(result)

    @classmethod
    def get(cls) -> Filesystem:
        """
        Get the filesystem singleton.

        Returns:
            Filesystem singleton instance

        Raises:
            FFIError: If singleton not initialized
        """
        lib = get_library().lib
        ffi = get_ffi()

        handle = ffi.new("FileSystemHandle*")
        result = lib.loon_get_filesystem_singleton_handle(handle)
        check_result(result)

        return Filesystem(handle[0])
