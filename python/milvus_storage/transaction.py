"""
Transaction classes for milvus-storage.
"""

from typing import Dict, List, Optional

from ._ffi import (
    LOON_TRANSACTION_RESOLVE_FAIL,
    LOON_TRANSACTION_RESOLVE_MERGE,
    LOON_TRANSACTION_RESOLVE_OVERWRITE,
    check_result,
    get_ffi,
    get_library,
)
from .exceptions import InvalidArgumentError, ResourceError
from .manifest import ColumnGroup, ColumnGroups, Manifest
from .properties import Properties


class ResolveStrategy:
    """
    Transaction conflict resolution strategies.

    When committing a transaction, conflicts may occur if another transaction
    has modified the same data. These strategies determine how to resolve conflicts.
    """

    FAIL = LOON_TRANSACTION_RESOLVE_FAIL
    """Fail the transaction on conflict."""

    MERGE = LOON_TRANSACTION_RESOLVE_MERGE
    """Merge changes on conflict."""

    OVERWRITE = LOON_TRANSACTION_RESOLVE_OVERWRITE
    """Overwrite existing data on conflict."""


class Transaction:
    """
    Transaction for atomic dataset modifications.

    Transactions provide ACID-like guarantees for dataset modifications.
    Changes are only visible after commit.

    Example:
        >>> from milvus_storage import Transaction
        >>>
        >>> with Transaction("/path/to/dataset") as txn:
        ...     manifest = txn.get_manifest()
        ...     # ... modify data ...
        ...     txn.append_files(new_column_groups)
        ...     version = txn.commit()
        ...     print(f"Committed version: {version}")
    """

    def __init__(
        self,
        base_path: str,
        properties: Optional[Dict[str, str]] = None,
        read_version: int = -1,
        retry_limit: int = 1,
    ):
        """
        Begin a new transaction.

        Args:
            base_path: Base path for the dataset
            properties: Optional configuration properties
            read_version: Version to read (-1 for latest)
            retry_limit: Maximum retries on commit conflicts (default: 1)

        Raises:
            InvalidArgumentError: If arguments are invalid
            FFIError: If transaction creation fails
        """
        self._lib = get_library().lib
        self._ffi = get_ffi()
        self._handle = None
        self._closed = False
        self._committed = False
        self._base_path = base_path

        # Create properties
        self._props = Properties(properties) if properties else Properties()

        # Begin transaction
        handle = self._ffi.new("LoonTransactionHandle*")
        result = self._lib.loon_transaction_begin(
            base_path.encode("utf-8"),
            self._props._get_c_properties(),
            read_version,
            retry_limit,
            handle,
        )
        check_result(result)

        self._handle = handle[0]

    def get_manifest(self) -> Manifest:
        """
        Get the current manifest for this transaction.

        Returns:
            Manifest containing column groups, delta logs, and stats

        Raises:
            ResourceError: If transaction is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Transaction is closed")

        manifest_ptr = self._ffi.new("LoonManifest**")
        result = self._lib.loon_transaction_get_manifest(self._handle, manifest_ptr)
        check_result(result)

        c_manifest = manifest_ptr[0]
        manifest = Manifest._from_c(c_manifest, self._ffi)

        # Free C manifest
        self._lib.loon_manifest_destroy(c_manifest)

        return manifest

    def get_read_version(self) -> int:
        """
        Get the read version of this transaction.

        Returns:
            Version number that was read

        Raises:
            ResourceError: If transaction is closed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Transaction is closed")

        version = self._ffi.new("int64_t*")
        result = self._lib.loon_transaction_get_read_version(self._handle, version)
        check_result(result)

        return version[0]

    def add_column_group(self, column_group: ColumnGroup) -> None:
        """
        Add a new column group to the transaction.

        Args:
            column_group: ColumnGroup to add

        Raises:
            ResourceError: If transaction is closed or committed
            InvalidArgumentError: If column_group is invalid
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Transaction is closed")
        if self._committed:
            raise ResourceError("Transaction is already committed")

        # Build C structure
        c_cg = self._ffi.new("LoonColumnGroup*")

        # Columns
        columns_c = [self._ffi.new("char[]", c.encode("utf-8")) for c in column_group.columns]
        columns_array = self._ffi.new("char*[]", columns_c)
        c_cg.columns = self._ffi.cast("const char**", columns_array)
        c_cg.num_of_columns = len(column_group.columns)

        # Format
        format_c = self._ffi.new("char[]", column_group.format.encode("utf-8"))
        c_cg.format = format_c

        # Files
        if column_group.files:
            c_files = self._ffi.new("LoonColumnGroupFile[]", len(column_group.files))
            path_buffers = []
            for i, f in enumerate(column_group.files):
                path_c = self._ffi.new("char[]", f.path.encode("utf-8"))
                path_buffers.append(path_c)
                c_files[i].path = path_c
                c_files[i].start_index = f.start_index
                c_files[i].end_index = f.end_index
                if f.metadata:
                    meta_c = self._ffi.new("uint8_t[]", f.metadata)
                    c_files[i].metadata = meta_c
                    c_files[i].metadata_size = len(f.metadata)
                else:
                    c_files[i].metadata = self._ffi.NULL
                    c_files[i].metadata_size = 0
            c_cg.files = c_files
            c_cg.num_of_files = len(column_group.files)
        else:
            c_cg.files = self._ffi.NULL
            c_cg.num_of_files = 0

        result = self._lib.loon_transaction_add_column_group(self._handle, c_cg)
        check_result(result)

    def append_files(self, column_groups: ColumnGroups) -> None:
        """
        Append files to existing column groups.

        Args:
            column_groups: ColumnGroups instance (from Writer.close or
                           ColumnGroups.create for external files)

        Raises:
            ResourceError: If transaction is closed or committed
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Transaction is closed")
        if self._committed:
            raise ResourceError("Transaction is already committed")

        # Get C pointer from ColumnGroups or use raw pointer for backward compatibility
        if isinstance(column_groups, ColumnGroups):
            c_column_groups = column_groups._get_c_ptr()
        else:
            # Legacy: raw C pointer
            c_column_groups = column_groups

        result = self._lib.loon_transaction_append_files(self._handle, c_column_groups)
        check_result(result)

    def add_delta_log(self, path: str, num_entries: int) -> None:
        """
        Add a delta log to the transaction.

        Args:
            path: Relative path to the delta log file
            num_entries: Number of entries in the delta log

        Raises:
            ResourceError: If transaction is closed or committed
            InvalidArgumentError: If arguments are invalid
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Transaction is closed")
        if self._committed:
            raise ResourceError("Transaction is already committed")

        if num_entries < 0:
            raise InvalidArgumentError(f"num_entries must be non-negative, got {num_entries}")

        result = self._lib.loon_transaction_add_delta_log(
            self._handle, path.encode("utf-8"), num_entries
        )
        check_result(result)

    def update_stat(self, key: str, files: List[str]) -> None:
        """
        Add or update a stat entry.

        Args:
            key: Stat key (e.g., "pk.delete", "bloomfilter", "bm25")
            files: List of file paths for this stat

        Raises:
            ResourceError: If transaction is closed or committed
            InvalidArgumentError: If arguments are invalid
            FFIError: If operation fails
        """
        if self._closed:
            raise ResourceError("Transaction is closed")
        if self._committed:
            raise ResourceError("Transaction is already committed")

        if not key:
            raise InvalidArgumentError("key cannot be empty")

        # Build C array
        files_c = [self._ffi.new("char[]", f.encode("utf-8")) for f in files]
        files_array = self._ffi.new("char*[]", files_c) if files else self._ffi.NULL

        result = self._lib.loon_transaction_update_stat(
            self._handle, key.encode("utf-8"), files_array, len(files)
        )
        check_result(result)

    def commit(self) -> int:
        """
        Commit the transaction.

        Returns:
            Committed version number

        Raises:
            ResourceError: If transaction is closed or already committed
            FFIError: If commit fails
        """
        if self._closed:
            raise ResourceError("Transaction is closed")
        if self._committed:
            raise ResourceError("Transaction is already committed")

        version = self._ffi.new("int64_t*")
        result = self._lib.loon_transaction_commit(self._handle, version)
        check_result(result)

        self._committed = True
        return version[0]

    def close(self) -> None:
        """Close the transaction and free resources."""
        if not self._closed and self._handle is not None:
            self._lib.loon_transaction_destroy(self._handle)
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
        """Check if transaction is closed."""
        return self._closed

    @property
    def is_committed(self) -> bool:
        """Check if transaction is committed."""
        return self._committed

    def __repr__(self) -> str:
        """String representation."""
        if self._committed:
            status = "committed"
        elif self._closed:
            status = "closed"
        else:
            status = "open"
        return f"Transaction(path={self._base_path!r}, status={status})"
