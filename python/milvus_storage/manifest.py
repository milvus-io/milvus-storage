"""
Manifest and column group structures for milvus-storage.
"""

from typing import List, Optional

from ._ffi import column_groups_debug_string, get_ffi, get_library


class ColumnGroupFile:
    """
    Represents a file within a column group.

    Attributes:
        path: File path
        start_index: Start row index (inclusive)
        end_index: End row index (exclusive)
        metadata: Optional metadata bytes
    """

    def __init__(
        self,
        path: str,
        start_index: int,
        end_index: int,
        metadata: Optional[bytes] = None,
    ):
        self.path = path
        self.start_index = start_index
        self.end_index = end_index
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"ColumnGroupFile(path={self.path!r}, "
            f"start_index={self.start_index}, end_index={self.end_index})"
        )


class ColumnGroup:
    """
    Represents a column group in the manifest.

    A column group contains a subset of columns stored together.

    Attributes:
        columns: List of column names
        format: Storage format (e.g., "parquet", "vortex")
        files: List of files in this column group
    """

    def __init__(
        self,
        columns: List[str],
        format: str,
        files: Optional[List[ColumnGroupFile]] = None,
    ):
        self.columns = columns
        self.format = format
        self.files = files or []

    def __repr__(self) -> str:
        return (
            f"ColumnGroup(columns={self.columns}, "
            f"format={self.format!r}, num_files={len(self.files)})"
        )


class ColumnGroups:
    """
    Wrapper around LoonColumnGroups C pointer.

    This class manages the lifecycle of the underlying C structure and provides
    Pythonic access to column groups data.

    Memory management:
        - When created from C FFI (e.g., Writer.close()), the C library owns the memory
          and loon_column_groups_destroy() must be called to free it.
        - When created from Python (e.g., from_list()), cffi manages the memory
          and no explicit destroy is needed.

    Example:
        >>> # Created by Writer.close() - C owns memory
        >>> with Writer(path, schema, properties=props) as writer:
        ...     writer.write(batch)
        ...     column_groups = writer.close()
        ...
        >>> # Created from Python list - Python/cffi owns memory
        >>> cg = ColumnGroups.from_list([
        ...     ColumnGroup(columns=["id", "vector"], format="parquet", files=[...])
        ... ])
        ...
        >>> # Use with Reader
        >>> with Reader(column_groups, schema, properties=props) as reader:
        ...     for batch in reader.scan():
        ...         print(batch)
    """

    def __init__(self, c_ptr, owned_by_c: bool = True):
        """
        Initialize from a C pointer.

        Args:
            c_ptr: LoonColumnGroups* pointer from C API
            owned_by_c: If True, memory is owned by C and must be freed with
                        loon_column_groups_destroy(). If False, memory is managed
                        by Python/cffi.
        """
        self._ffi = get_ffi()
        self._lib = get_library().lib
        self._ptr = c_ptr
        self._destroyed = False
        self._owned_by_c = owned_by_c
        # Keep references to Python-allocated buffers to prevent GC
        self._python_buffers = []

    @classmethod
    def from_list(cls, column_groups: List[ColumnGroup]) -> "ColumnGroups":
        """
        Create a ColumnGroups from a list of Python ColumnGroup objects.

        The memory is managed by Python/cffi, not by the C library.

        Args:
            column_groups: List of ColumnGroup objects

        Returns:
            ColumnGroups instance with Python-managed memory

        Example:
            >>> cg = ColumnGroups.from_list([
            ...     ColumnGroup(
            ...         columns=["id", "vector"],
            ...         format="parquet",
            ...         files=[
            ...             ColumnGroupFile("/data/file1.parquet", 0, 1000),
            ...             ColumnGroupFile("/data/file2.parquet", 1000, 2000),
            ...         ]
            ...     )
            ... ])
        """
        ffi = get_ffi()

        # Keep all allocated buffers alive
        buffers = []

        # Allocate main structure
        c_cgs = ffi.new("LoonColumnGroups*")
        buffers.append(c_cgs)

        num_cgs = len(column_groups)
        c_cgs.num_of_column_groups = num_cgs

        if num_cgs == 0:
            c_cgs.column_group_array = ffi.NULL
        else:
            # Allocate column group array
            cg_array = ffi.new("LoonColumnGroup[]", num_cgs)
            buffers.append(cg_array)
            c_cgs.column_group_array = cg_array

            for i, cg in enumerate(column_groups):
                c_cg = cg_array[i]

                # Columns
                num_cols = len(cg.columns)
                c_cg.num_of_columns = num_cols
                if num_cols > 0:
                    columns_ptrs = ffi.new("char*[]", num_cols)
                    buffers.append(columns_ptrs)
                    for j, col in enumerate(cg.columns):
                        col_buf = ffi.new("char[]", col.encode("utf-8"))
                        buffers.append(col_buf)
                        columns_ptrs[j] = col_buf
                    c_cg.columns = ffi.cast("const char**", columns_ptrs)
                else:
                    c_cg.columns = ffi.NULL

                # Format
                format_buf = ffi.new("char[]", cg.format.encode("utf-8"))
                buffers.append(format_buf)
                c_cg.format = format_buf

                # Files
                num_files = len(cg.files)
                c_cg.num_of_files = num_files
                if num_files > 0:
                    files_array = ffi.new("LoonColumnGroupFile[]", num_files)
                    buffers.append(files_array)
                    c_cg.files = files_array

                    for k, f in enumerate(cg.files):
                        c_file = files_array[k]

                        # Path
                        path_buf = ffi.new("char[]", f.path.encode("utf-8"))
                        buffers.append(path_buf)
                        c_file.path = path_buf

                        c_file.start_index = f.start_index
                        c_file.end_index = f.end_index

                        # Metadata
                        if f.metadata:
                            meta_buf = ffi.new("uint8_t[]", len(f.metadata))
                            ffi.memmove(meta_buf, f.metadata, len(f.metadata))
                            buffers.append(meta_buf)
                            c_file.metadata = meta_buf
                            c_file.metadata_size = len(f.metadata)
                        else:
                            c_file.metadata = ffi.NULL
                            c_file.metadata_size = 0
                else:
                    c_cg.files = ffi.NULL

        # Create instance with owned_by_c=False
        instance = cls(c_cgs, owned_by_c=False)
        instance._python_buffers = buffers
        return instance

    def to_list(self) -> List[ColumnGroup]:
        """
        Convert to a list of Python ColumnGroup objects.

        Returns:
            List of ColumnGroup objects

        Raises:
            ResourceError: If already destroyed
        """
        from .exceptions import ResourceError

        if self._destroyed:
            raise ResourceError("ColumnGroups has been destroyed")

        result = []
        for i in range(self._ptr.num_of_column_groups):
            cg = self._ptr.column_group_array[i]

            # Parse columns
            columns = []
            for j in range(cg.num_of_columns):
                col_name = self._ffi.string(cg.columns[j]).decode("utf-8")
                columns.append(col_name)

            # Parse format
            format_str = self._ffi.string(cg.format).decode("utf-8") if cg.format else ""

            # Parse files
            files = []
            for k in range(cg.num_of_files):
                f = cg.files[k]
                path = self._ffi.string(f.path).decode("utf-8") if f.path else ""

                metadata = None
                if f.metadata != self._ffi.NULL and f.metadata_size > 0:
                    metadata = bytes(self._ffi.buffer(f.metadata, f.metadata_size))

                files.append(ColumnGroupFile(path, f.start_index, f.end_index, metadata))

            result.append(ColumnGroup(columns, format_str, files))

        return result

    def debug_string(self) -> str:
        """
        Get a debug string representation.

        Returns:
            Debug string with detailed information about the column groups
        """
        if self._destroyed:
            return "ColumnGroups(destroyed)"
        return column_groups_debug_string(self._ptr)

    def destroy(self) -> None:
        """
        Manually destroy the underlying C structure.

        After calling this method, the ColumnGroups object cannot be used.

        Note:
            - If memory is owned by C (from Writer.close() etc.), this calls
              loon_column_groups_destroy() to free the memory.
            - If memory is owned by Python (from from_list()), this just clears
              the references and lets Python/cffi handle cleanup.
        """
        if not self._destroyed and self._ptr is not None:
            if self._owned_by_c:
                self._lib.loon_column_groups_destroy(self._ptr)
            # Clear Python buffer references to allow GC
            self._python_buffers = []
            self._ptr = None
            self._destroyed = True

    def _get_c_ptr(self):
        """
        Get the underlying C pointer.

        For internal use by Reader, Writer, Transaction.

        Returns:
            LoonColumnGroups* pointer

        Raises:
            ResourceError: If already destroyed
        """
        from .exceptions import ResourceError

        if self._destroyed:
            raise ResourceError("ColumnGroups has been destroyed")
        return self._ptr

    @property
    def is_destroyed(self) -> bool:
        """Check if the ColumnGroups has been destroyed."""
        return self._destroyed

    def __len__(self) -> int:
        """Return the number of column groups."""
        if self._destroyed:
            return 0
        return self._ptr.num_of_column_groups

    def __iter__(self):
        """Iterate over column groups as Python objects."""
        return iter(self.to_list())

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - destroy the column groups."""
        self.destroy()

    def __del__(self):
        """Destructor - clean up C resources."""
        try:
            self.destroy()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._destroyed:
            return "ColumnGroups(destroyed)"
        return f"ColumnGroups(num_of_column_groups={len(self)})"


class DeltaLog:
    """
    Represents a delta log entry.

    Attributes:
        path: Path to the delta log file
        num_entries: Number of entries in the log
    """

    def __init__(self, path: str, num_entries: int):
        self.path = path
        self.num_entries = num_entries

    def __repr__(self) -> str:
        return f"DeltaLog(path={self.path!r}, num_entries={self.num_entries})"


class StatEntry:
    """
    Represents a stat entry.

    Attributes:
        key: Stat key (e.g., "pk.delete", "bloomfilter", "bm25")
        files: List of file paths for this stat
        metadata: Dict of key-value metadata pairs
    """

    def __init__(self, key: str, files: List[str], metadata: Optional[dict] = None):
        self.key = key
        self.files = files
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"StatEntry(key={self.key!r}, "
            f"num_files={len(self.files)}, "
            f"num_metadata={len(self.metadata)})"
        )


class Manifest:
    """
    Dataset manifest containing column groups, delta logs, and stats.

    The manifest describes the structure and location of all data in the dataset.

    Attributes:
        column_groups: List of column groups
        delta_logs: List of delta logs
        stats: List of stat entries
    """

    def __init__(
        self,
        column_groups: Optional[List[ColumnGroup]] = None,
        delta_logs: Optional[List[DeltaLog]] = None,
        stats: Optional[List[StatEntry]] = None,
    ):
        self.column_groups = column_groups or []
        self.delta_logs = delta_logs or []
        self.stats = stats or []

    @classmethod
    def _from_c(cls, c_manifest, ffi) -> "Manifest":
        """Create Manifest from C structure."""
        # Parse column groups
        column_groups = []
        cg_array = c_manifest.column_groups
        for i in range(cg_array.num_of_column_groups):
            cg = cg_array.column_group_array[i]

            # Parse columns
            columns = []
            for j in range(cg.num_of_columns):
                col_name = ffi.string(cg.columns[j]).decode("utf-8")
                columns.append(col_name)

            # Parse format
            format_str = ffi.string(cg.format).decode("utf-8") if cg.format else ""

            # Parse files
            files = []
            for k in range(cg.num_of_files):
                f = cg.files[k]
                path = ffi.string(f.path).decode("utf-8") if f.path else ""

                metadata = None
                if f.metadata != ffi.NULL and f.metadata_size > 0:
                    metadata = bytes(ffi.buffer(f.metadata, f.metadata_size))

                files.append(ColumnGroupFile(path, f.start_index, f.end_index, metadata))

            column_groups.append(ColumnGroup(columns, format_str, files))

        # Parse delta logs
        delta_logs = []
        dl = c_manifest.delta_logs
        for i in range(dl.num_delta_logs):
            path = ffi.string(dl.delta_log_paths[i]).decode("utf-8")
            num_entries = dl.delta_log_num_entries[i]
            delta_logs.append(DeltaLog(path, num_entries))

        # Parse stats
        stats = []
        st = c_manifest.stats
        for i in range(st.num_stats):
            key = ffi.string(st.stat_keys[i]).decode("utf-8")
            files = []
            for j in range(st.stat_file_counts[i]):
                file_path = ffi.string(st.stat_files[i][j]).decode("utf-8")
                files.append(file_path)
            metadata = {}
            if st.stat_metadata_counts and st.stat_metadata_keys and st.stat_metadata_keys[i]:
                for j in range(st.stat_metadata_counts[i]):
                    mk = ffi.string(st.stat_metadata_keys[i][j]).decode("utf-8")
                    mv = ffi.string(st.stat_metadata_values[i][j]).decode("utf-8")
                    metadata[mk] = mv
            stats.append(StatEntry(key, files, metadata))

        return cls(column_groups, delta_logs, stats)

    def __repr__(self) -> str:
        return (
            f"Manifest(column_groups={len(self.column_groups)}, "
            f"delta_logs={len(self.delta_logs)}, stats={len(self.stats)})"
        )


def destroy_column_groups(column_groups) -> None:
    """
    Destroy a column groups structure and free memory.

    .. deprecated::
        Use ColumnGroups.destroy() or context manager instead.

    Args:
        column_groups: ColumnGroups instance or LoonColumnGroups pointer
    """
    if isinstance(column_groups, ColumnGroups):
        column_groups.destroy()
    else:
        # Legacy: raw C pointer
        lib = get_library().lib
        lib.loon_column_groups_destroy(column_groups)
