"""
Writer class for milvus-storage.
"""

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore

from ._ffi import check_result, get_ffi, get_library
from .exceptions import InvalidArgumentError, ResourceError
from .properties import Properties


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

    def __init__(self, path: str, schema: "pa.Schema", properties: Optional[Dict[str, str]] = None):
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
        handle = self._ffi.new("LoonWriterHandle*")
        result = self._lib.loon_writer_new(
            path.encode("utf-8"), c_schema, self._props._get_c_properties(), handle
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
        result = self._lib.loon_writer_write(self._handle, c_array)
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

        result = self._lib.loon_writer_flush(self._handle)
        check_result(result)

    def close(self):
        """
        Close the writer and return the column groups.

        This finalizes all writes and returns the column groups metadata.

        Returns:
            LoonColumnGroups pointer containing the dataset column groups

        Raises:
            ResourceError: If writer is already closed
            FFIError: If close operation fails
        """
        if self._closed or self._handle is None:
            raise ResourceError("Writer is already closed")

        column_groups_ptr = self._ffi.new("LoonColumnGroups**")

        # Pass NULL for meta_keys/vals/len as we don't support custom metadata yet
        result = self._lib.loon_writer_close(
            self._handle, self._ffi.NULL, self._ffi.NULL, 0, column_groups_ptr
        )
        check_result(result)

        column_groups = column_groups_ptr[0]

        # Destroy writer
        self._lib.loon_writer_destroy(self._handle)
        self._handle = None
        self._closed = True

        return column_groups

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
                    self._lib.loon_writer_destroy(self._handle)
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
