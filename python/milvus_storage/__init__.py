"""
Python bindings for milvus-storage.

Milvus-storage is a high-performance storage engine using Apache Arrow Parquet
as the underlying format, optimized for analytical workloads.

Example:
    >>> import pyarrow as pa
    >>> from milvus_storage import Writer, Reader
    >>>
    >>> # Define schema
    >>> schema = pa.schema([
    ...     pa.field("id", pa.int64()),
    ...     pa.field("vector", pa.list_(pa.float32(), 128)),
    ...     pa.field("text", pa.string())
    ... ])
    >>>
    >>> # Write data
    >>> with Writer("/tmp/my_dataset", schema) as writer:
    ...     batch = pa.record_batch([[1, 2, 3], ...], schema=schema)
    ...     writer.write(batch)
    ...     column_groups = writer.close()
    >>>
    >>> # Read data
    >>> with Reader(column_groups, schema) as reader:
    ...     for batch in reader.scan():
    ...         print(batch)
"""

from .exceptions import (
    ArrowError,
    FFIError,
    InvalidArgumentError,
    MilvusStorageError,
    ResourceError,
)
from .properties import Properties
from .reader import ChunkReader, Reader
from .writer import Writer

__version__ = "0.1.0"

__all__ = [
    "Writer",
    "Reader",
    "ChunkReader",
    "Properties",
    "MilvusStorageError",
    "FFIError",
    "ArrowError",
    "InvalidArgumentError",
    "ResourceError",
]
