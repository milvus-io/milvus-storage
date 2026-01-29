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

from ._ffi import column_groups_debug_string, manifest_debug_string
from .common import ThreadPool
from .exceptions import (
    ArrowError,
    FFIError,
    InvalidArgumentError,
    MilvusStorageError,
    ResourceError,
)
from .exttable import ExternalTable
from .filesystem import (
    FileInfo,
    Filesystem,
    FilesystemMetrics,
    FilesystemReader,
    FilesystemSingleton,
    FilesystemWriter,
)
from .manifest import (
    ColumnGroup,
    ColumnGroupFile,
    ColumnGroups,
    DeltaLog,
    Manifest,
    StatEntry,
    destroy_column_groups,
)
from .properties import Properties, PropertyKeys
from .reader import ChunkMetadata, ChunkMetadataType, ChunkReader, Reader
from .transaction import ResolveStrategy, Transaction
from .writer import Writer

__version__ = "0.1.0"

__all__ = [
    # Writer/Reader
    "Writer",
    "Reader",
    "ChunkReader",
    "ChunkMetadata",
    "ChunkMetadataType",
    # Transaction
    "Transaction",
    "ResolveStrategy",
    # Manifest
    "Manifest",
    "ColumnGroup",
    "ColumnGroupFile",
    "ColumnGroups",
    "DeltaLog",
    "StatEntry",
    "destroy_column_groups",
    "manifest_debug_string",
    "column_groups_debug_string",
    # External Table
    "ExternalTable",
    # Filesystem
    "Filesystem",
    "FilesystemReader",
    "FilesystemWriter",
    "FilesystemMetrics",
    "FilesystemSingleton",
    "FileInfo",
    # Common
    "ThreadPool",
    "Properties",
    "PropertyKeys",
    # Exceptions
    "MilvusStorageError",
    "FFIError",
    "ArrowError",
    "InvalidArgumentError",
    "ResourceError",
]
