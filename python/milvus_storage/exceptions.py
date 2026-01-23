"""
Exception classes for milvus-storage Python package.
"""


class MilvusStorageError(Exception):
    """Base exception for milvus-storage errors."""

    pass


class FFIError(MilvusStorageError):
    """Error from C FFI layer."""

    pass


class ArrowError(MilvusStorageError):
    """Error during Arrow data conversion."""

    pass


class InvalidArgumentError(MilvusStorageError):
    """Invalid argument passed to API."""

    pass


class ResourceError(MilvusStorageError):
    """Resource management error (memory, handles, etc.)."""

    pass
