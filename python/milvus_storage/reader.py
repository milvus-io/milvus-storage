"""
Reader classes for milvus-storage.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore

from ._ffi import check_result, get_ffi, get_library
from .exceptions import InvalidArgumentError, ResourceError
from .properties import Properties


class ChunkReader:
    """
    Reader for accessing specific chunks within a column group.

    ChunkReaders provide random access to chunks (contiguous groups of rows)
    within a single column group. This is useful for efficient random access
    patterns.

    Example:
        >>> reader = Reader(manifest, schema)
        >>> chunk_reader = reader.get_chunk_reader(column_group_id=0)
        >>> chunk = chunk_reader.get_chunk(5)  # Get 5th chunk
    """

    def __init__(self, handle):
        """
        Initialize ChunkReader (internal use).

        Args:
            handle: C handle to chunk reader
        """
        self._lib = get_library().lib
        self._ffi = get_ffi()
        self._handle = handle
        self._closed = False

    def get_chunk(self, index: int) -> "pa.RecordBatch":
        """
        Retrieve a single chunk by index.

        Args:
            index: Zero-based chunk index

        Returns:
            RecordBatch containing the chunk data

        Raises:
            InvalidArgumentError: If index is negative
            ResourceError: If reader is closed
            FFIError: If read operation fails
        """
        if self._closed:
            raise ResourceError("ChunkReader is closed")

        if index < 0:
            raise InvalidArgumentError(f"Chunk index must be non-negative, got {index}")

        # Allocate Arrow C Data Interface structure using milvus-storage FFI
        c_array = self._ffi.new("struct ArrowArray*")
        result = self._lib.loon_get_chunk(self._handle, index, c_array)
        check_result(result)

        # Import to PyArrow
        import pyarrow as pa  # type: ignore

        return pa.RecordBatch._import_from_c(int(self._ffi.cast("uintptr_t", c_array)))

    def get_chunks(
        self, indices: Union[List[int], np.ndarray], parallelism: int = 1
    ) -> List["pa.RecordBatch"]:
        """
        Retrieve multiple chunks by their indices.

        Args:
            indices: List or array of chunk indices
            parallelism: Number of threads for parallel reading (default: 1)

        Returns:
            List of RecordBatches, one per requested chunk

        Raises:
            InvalidArgumentError: If indices are invalid
            ResourceError: If reader is closed
            FFIError: If read operation fails
        """
        if self._closed:
            raise ResourceError("ChunkReader is closed")

        if not indices:
            return []

        # Convert to numpy array
        indices_array = np.asarray(indices, dtype=np.int64)
        if indices_array.ndim != 1:
            raise InvalidArgumentError("indices must be 1-dimensional")

        # Create C array - use numpy's ctypes interop
        indices_ptr = indices_array.ctypes.data

        arrays_ptr_holder = self._ffi.new("struct ArrowArray**")
        num_arrays = self._ffi.new("size_t*")

        result = self._lib.loon_get_chunks(
            self._handle,
            self._ffi.cast("int64_t*", indices_ptr),
            len(indices_array),
            parallelism,
            arrays_ptr_holder,
            num_arrays,
        )
        check_result(result)

        # Import arrays to PyArrow
        import pyarrow as pa  # type: ignore

        batches = []
        arrays_ptr = arrays_ptr_holder[0]
        for i in range(num_arrays[0]):
            # Get pointer to the i-th ArrowArray in the array
            array_ptr = self._ffi.cast(
                "struct ArrowArray*",
                int(self._ffi.cast("uintptr_t", arrays_ptr))
                + i * self._ffi.sizeof("struct ArrowArray"),
            )
            batch = pa.RecordBatch._import_from_c(int(self._ffi.cast("uintptr_t", array_ptr)))
            batches.append(batch)

        # Free C arrays
        self._lib.loon_free_chunk_arrays(arrays_ptr, num_arrays[0])

        return batches

    def get_chunk_indices(self, row_indices: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Map row indices to their corresponding chunk indices.

        Args:
            row_indices: Global row indices to map

        Returns:
            Array of chunk indices

        Raises:
            ResourceError: If reader is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("ChunkReader is closed")

        # Convert to numpy array
        row_indices_array = np.asarray(row_indices, dtype=np.int64)
        row_indices_ptr = row_indices_array.ctypes.data

        chunk_indices_ptr = self._ffi.new("int64_t**")
        num_chunks = self._ffi.new("size_t*")

        result = self._lib.loon_get_chunk_indices(
            self._handle,
            self._ffi.cast("int64_t*", row_indices_ptr),
            len(row_indices_array),
            chunk_indices_ptr,
            num_chunks,
        )
        check_result(result)

        # Copy to numpy array
        chunk_indices = np.frombuffer(
            self._ffi.buffer(chunk_indices_ptr[0], num_chunks[0] * self._ffi.sizeof("int64_t")),
            dtype=np.int64,
        ).copy()

        # Free C array
        self._lib.loon_free_chunk_indices(chunk_indices_ptr[0])

        return chunk_indices

    def close(self) -> None:
        """Close the chunk reader and free resources."""
        if not self._closed and self._handle is not None:
            self._lib.loon_chunk_reader_destroy(self._handle)
            self._handle = None
            self._closed = True

    def __del__(self):
        """Cleanup on destruction."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class Reader:
    """
    Reader for milvus-storage datasets.

    The Reader provides multiple access patterns:
    - Full table scan with optional filtering
    - Random access via row indices
    - Chunk-based access for specific column groups

    Example:
        >>> from milvus_storage import Reader
        >>> import pyarrow as pa
        >>>
        >>> # Create reader
        >>> reader = Reader(column_groups_json, schema)
        >>>
        >>> # Scan entire dataset
        >>> for batch in reader.scan():
        ...     process(batch)
        >>>
        >>> # Random access
        >>> indices = [10, 100, 1000]
        >>> batch = reader.take(indices)
    """

    def __init__(
        self,
        column_groups,
        schema: "pa.Schema",
        columns: Optional[List[str]] = None,
        properties: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize a new Reader.

        Args:
            column_groups: Dataset column groups handle (from Writer.close)
            schema: PyArrow schema for the dataset
            columns: Optional list of column names to read (default: all)
            properties: Optional configuration properties

        Raises:
            InvalidArgumentError: If arguments are invalid
            FFIError: If reader creation fails
        """
        # Load native library BEFORE importing pyarrow to avoid TLS conflicts
        self._lib = get_library().lib
        self._ffi = get_ffi()

        # Import pyarrow lazily
        import pyarrow as pa  # type: ignore

        if not isinstance(schema, pa.Schema):
            raise InvalidArgumentError(
                f"schema must be a pyarrow.Schema, got {type(schema).__name__}"
            )

        self._handle = None
        self._schema = schema
        self._closed = False

        # Create properties
        self._props = Properties(properties) if properties else Properties()

        # Export schema to C using milvus-storage FFI
        c_schema = self._ffi.new("struct ArrowSchema*")
        schema._export_to_c(int(self._ffi.cast("uintptr_t", c_schema)))

        # Prepare columns array
        if columns:
            # Create cdata pointers for each column name
            columns_cdata = [self._ffi.new("char[]", c.encode("utf-8")) for c in columns]
            columns_array = self._ffi.new("char*[]", columns_cdata)
            num_columns = len(columns)
        else:
            columns_array = self._ffi.NULL
            num_columns = 0

        # Create reader
        handle = self._ffi.new("LoonReaderHandle*")
        result = self._lib.loon_reader_new(
            column_groups,
            c_schema,
            columns_array,
            num_columns,
            self._props._get_c_properties(),
            handle,
        )
        check_result(result)

        self._handle = handle[0]

    def scan(self, predicate: Optional[str] = None) -> "pa.RecordBatchReader":
        """
        Perform a full table scan with optional filtering.

        Args:
            predicate: Optional filter expression (e.g., "id > 100")

        Returns:
            PyArrow RecordBatchReader for streaming data

        Raises:
            ResourceError: If reader is closed
            FFIError: If scan operation fails

        Example:
            >>> for batch in reader.scan(predicate="age > 18"):
            ...     print(f"Got {len(batch)} rows")
        """
        if self._closed or self._handle is None:
            raise ResourceError("Reader is closed")

        # Allocate Arrow C Data Interface structure using milvus-storage FFI
        c_stream = self._ffi.new("struct ArrowArrayStream*")

        predicate_bytes = predicate.encode("utf-8") if predicate else self._ffi.NULL

        result = self._lib.loon_get_record_batch_reader(self._handle, predicate_bytes, c_stream)
        check_result(result)

        # Import to PyArrow
        import pyarrow as pa  # type: ignore

        return pa.RecordBatchReader._import_from_c(int(self._ffi.cast("uintptr_t", c_stream)))

    def take(self, indices: Union[List[int], np.ndarray], parallelism: int = 1) -> "pa.RecordBatch":
        """
        Extract specific rows by their global indices.

        Args:
            indices: Row indices to extract
            parallelism: Number of threads for parallel reading (default: 1)

        Returns:
            RecordBatch containing the requested rows

        Raises:
            InvalidArgumentError: If indices are invalid
            ResourceError: If reader is closed
            FFIError: If take operation fails

        Example:
            >>> batch = reader.take([0, 10, 100, 1000], parallelism=4)
        """
        if self._closed or self._handle is None:
            raise ResourceError("Reader is closed")

        if not indices:
            raise InvalidArgumentError("indices cannot be empty")

        # Convert to numpy array
        indices_array = np.asarray(indices, dtype=np.int64)
        if indices_array.ndim != 1:
            raise InvalidArgumentError("indices must be 1-dimensional")

        # Create C array - use numpy's ctypes interop
        indices_ptr = indices_array.ctypes.data

        # Allocate Arrow C Data Interface structure using milvus-storage FFI
        c_array = self._ffi.new("struct ArrowArray*")

        result = self._lib.loon_take(
            self._handle,
            self._ffi.cast("int64_t*", indices_ptr),
            len(indices_array),
            parallelism,
            c_array,
        )
        check_result(result)

        # Import to PyArrow
        import pyarrow as pa  # type: ignore

        return pa.RecordBatch._import_from_c(int(self._ffi.cast("uintptr_t", c_array)))

    def get_chunk_reader(self, column_group_id: int) -> ChunkReader:
        """
        Get a chunk reader for a specific column group.

        Args:
            column_group_id: ID of the column group

        Returns:
            ChunkReader for the specified column group

        Raises:
            InvalidArgumentError: If column_group_id is invalid
            ResourceError: If reader is closed
            FFIError: If operation fails
        """
        if self._closed or self._handle is None:
            raise ResourceError("Reader is closed")

        if column_group_id < 0:
            raise InvalidArgumentError(
                f"column_group_id must be non-negative, got {column_group_id}"
            )

        chunk_handle = self._ffi.new("LoonChunkReaderHandle*")
        result = self._lib.loon_get_chunk_reader(self._handle, column_group_id, chunk_handle)
        check_result(result)

        return ChunkReader(chunk_handle[0])

    def close(self) -> None:
        """Close the reader and free resources."""
        if not self._closed and self._handle is not None:
            self._lib.loon_reader_destroy(self._handle)
            self._handle = None
            self._closed = True

    def __del__(self):
        """Cleanup on destruction."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @property
    def is_closed(self) -> bool:
        """Check if reader is closed."""
        return self._closed

    @property
    def schema(self) -> "pa.Schema":
        """Get the schema."""
        return self._schema

    def __repr__(self) -> str:
        """String representation."""
        status = "closed" if self._closed else "open"
        return f"Reader(status={status})"
