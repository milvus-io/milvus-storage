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
    ...     manifest = writer.close()
    >>>
    >>> # Read data
    >>> with Reader(manifest, schema) as reader:
    ...     for batch in reader.scan():
    ...         print(batch)
"""

from .writer import Writer
from .reader import Reader, ChunkReader
from .properties import Properties
from .exceptions import (
    MilvusStorageError,
    FFIError,
    ArrowError,
    InvalidArgumentError,
    ResourceError
)

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
