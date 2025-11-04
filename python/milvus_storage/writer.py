"""
Writer class for milvus-storage.
"""

from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore

from ._ffi import get_library, get_ffi, check_result
from .properties import Properties
from .exceptions import InvalidArgumentError, ResourceError


class Writer:
    """
    Writer for milvus-storage datasets.

    The Writer allows you to write Arrow RecordBatches to milvus-storage format.
    Data is buffered in memory and flushed to storage either manually or when
    the buffer is full.

    Example:
        >>> import pyarrow as pa
        >>> from milvus_storage import Writer
        >>>
        >>> schema = pa.schema([
        ...     pa.field("id", pa.int64()),
        ...     pa.field("vector", pa.list_(pa.float32(), 128)),
        ...     pa.field("metadata", pa.string())
        ... ])
        >>>
        >>> with Writer("/path/to/storage", schema) as writer:
        ...     batch = pa.record_batch([...], schema=schema)
        ...     writer.write(batch)
        ...     manifest = writer.close()
    """

    def __init__(
        self,
        path: str,
        schema: "pa.Schema",
        properties: Optional[Dict[str, str]] = None
    ):
        """
        Initialize a new Writer.

        Args:
            path: Base path where data will be written
            schema: PyArrow schema for the dataset
            properties: Optional configuration properties

        Raises:
            InvalidArgumentError: If arguments are invalid
            FFIError: If writer creation fails
        """
        # Load native library BEFORE importing pyarrow to avoid TLS conflicts
        self._lib = get_library().lib
        self._ffi = get_ffi()

        # Import pyarrow lazily to control load order
        import pyarrow as pa  # type: ignore

        if not isinstance(schema, pa.Schema):
            raise InvalidArgumentError(
                f"schema must be a pyarrow.Schema, got {type(schema).__name__}"
            )

        self._handle = None
        self._path = path
        self._schema = schema
        self._closed = False

        # Create properties
        self._props = Properties(properties) if properties else Properties()

        # Export schema to C using milvus-storage FFI
        c_schema = self._ffi.new("struct ArrowSchema*")
        schema._export_to_c(int(self._ffi.cast("uintptr_t", c_schema)))

        # Create writer
        handle = self._ffi.new("WriterHandle*")
        result = self._lib.writer_new(
            path.encode('utf-8'),
            c_schema,
            self._props._get_c_properties(),
            handle
        )
        check_result(result)

        self._handle = handle[0]

    def write(self, batch: "pa.RecordBatch") -> None:
        """
        Write a RecordBatch to the dataset.

        The batch schema must match the schema provided during Writer creation.

        Args:
            batch: PyArrow RecordBatch to write

        Raises:
            InvalidArgumentError: If batch schema doesn't match
            ResourceError: If writer is closed
            FFIError: If write operation fails
        """
        if self._closed or self._handle is None:
            raise ResourceError("Writer is closed")

        import pyarrow as pa  # type: ignore
        if not isinstance(batch, pa.RecordBatch):
            raise InvalidArgumentError(
                f"batch must be a pyarrow.RecordBatch, got {type(batch).__name__}"
            )

        if not batch.schema.equals(self._schema):
            raise InvalidArgumentError(
                f"Batch schema doesn't match writer schema.\n"
                f"Expected: {self._schema}\n"
                f"Got: {batch.schema}"
            )

        # Export batch to C using milvus-storage FFI
        c_array = self._ffi.new("struct ArrowArray*")
        batch._export_to_c(int(self._ffi.cast("uintptr_t", c_array)))

        # Write
        result = self._lib.writer_write(self._handle, c_array)
        check_result(result)

    def flush(self) -> None:
        """
        Flush buffered data to storage.

        This forces any buffered data to be written to storage files without
        closing the writer. You can continue writing after flushing.

        Raises:
            ResourceError: If writer is closed
            FFIError: If flush operation fails
        """
        if self._closed or self._handle is None:
            raise ResourceError("Writer is closed")

        result = self._lib.writer_flush(self._handle)
        check_result(result)

    def close(self) -> str:
        """
        Close the writer and return the manifest.

        This finalizes all writes and returns a JSON string containing the
        manifest metadata for the written dataset. The manifest is needed
        to read the data later.

        Returns:
            JSON string containing the dataset manifest

        Raises:
            ResourceError: If writer is already closed
            FFIError: If close operation fails
        """
        if self._closed or self._handle is None:
            raise ResourceError("Writer is already closed")

        manifest_ptr = self._ffi.new("char**")
        manifest_size = self._ffi.new("size_t*")

        result = self._lib.writer_close(
            self._handle,
            manifest_ptr,
            manifest_size
        )
        check_result(result)

        # Copy manifest to Python string
        manifest = self._ffi.buffer(manifest_ptr[0], manifest_size[0])[:].decode('utf-8')

        # Free C manifest
        self._lib.free_manifest(manifest_ptr[0])

        # Destroy writer
        self._lib.writer_destroy(self._handle)
        self._handle = None
        self._closed = True

        return manifest

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if not self._closed and self._handle is not None:
            try:
                self.close()
            except Exception:
                # Don't hide the original exception
                if exc_type is None:
                    raise

    def __del__(self):
        """Cleanup on destruction."""
        try:
            if hasattr(self, "_closed") and hasattr(self, "_handle") and hasattr(self, "_lib"):
                if not self._closed and self._handle is not None:
                    self._lib.writer_destroy(self._handle)
                    self._handle = None
        except Exception:
            # Suppress all exceptions in destructor
            pass

    @property
    def is_closed(self) -> bool:
        """Check if writer is closed."""
        return self._closed

    def __repr__(self) -> str:
        """String representation."""
        status = "closed" if self._closed else "open"
        return f"Writer(path={self._path!r}, status={status})"
