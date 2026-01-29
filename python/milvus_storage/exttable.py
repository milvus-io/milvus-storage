"""
External table utilities for milvus-storage.

Provides functions to work with external files and import them into datasets.
"""

from typing import Dict, List, Optional, Tuple

from ._ffi import check_result, get_ffi, get_library
from .manifest import Manifest
from .properties import Properties


class ExternalTable:
    """
    Utilities for working with external tables and files.

    External tables allow importing existing Parquet or other format files
    into a milvus-storage dataset without copying the data.

    Example:
        >>> from milvus_storage import ExternalTable
        >>>
        >>> # Explore a directory for parquet files
        >>> num_files, manifest_path = ExternalTable.explore(
        ...     columns=["id", "vector", "text"],
        ...     format="parquet",
        ...     base_dir="/data/dataset",
        ...     explore_dir="/data/dataset/files"
        ... )
        >>> print(f"Found {num_files} files, manifest at: {manifest_path}")
        >>>
        >>> # Get info about a specific file
        >>> num_rows = ExternalTable.get_file_info(
        ...     format="parquet",
        ...     file_path="/data/dataset/file.parquet"
        ... )
        >>> print(f"File has {num_rows} rows")
        >>>
        >>> # Read manifest from file
        >>> manifest = ExternalTable.read_manifest("/data/dataset/manifest.json")
    """

    @staticmethod
    def explore(
        columns: List[str],
        format: str,
        base_dir: str,
        explore_dir: str,
        properties: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str]:
        """
        Explore a directory for external files and generate column groups.

        Scans the specified directory for files matching the given format
        and generates a manifest file describing the column groups.

        Args:
            columns: List of column names to include
            format: File format (e.g., "parquet")
            base_dir: Base directory path for the dataset
            explore_dir: Directory to explore for files
            properties: Optional configuration properties (e.g., S3 credentials)

        Returns:
            Tuple of (number_of_files, manifest_file_path)

        Raises:
            InvalidArgumentError: If arguments are invalid
            FFIError: If operation fails

        Example:
            >>> num_files, path = ExternalTable.explore(
            ...     columns=["id", "vector"],
            ...     format="parquet",
            ...     base_dir="/data",
            ...     explore_dir="/data/parquet_files"
            ... )
        """
        lib = get_library().lib
        ffi = get_ffi()

        props = Properties(properties) if properties else Properties()

        # Build columns array
        columns_c = [ffi.new("char[]", c.encode("utf-8")) for c in columns]
        columns_array = ffi.new("char*[]", columns_c)

        format_bytes = format.encode("utf-8")
        base_dir_bytes = base_dir.encode("utf-8")
        explore_dir_bytes = explore_dir.encode("utf-8")

        num_files = ffi.new("uint64_t*")
        manifest_path = ffi.new("char**")

        result = lib.loon_exttable_explore(
            columns_array,
            len(columns),
            format_bytes,
            base_dir_bytes,
            explore_dir_bytes,
            props._get_c_properties(),
            num_files,
            manifest_path,
        )
        check_result(result)

        # Convert result
        path_str = ffi.string(manifest_path[0]).decode("utf-8")

        # Free the C string
        lib.loon_free_cstr(manifest_path[0])

        return num_files[0], path_str

    @staticmethod
    def get_file_info(
        format: str,
        file_path: str,
        properties: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Get information about an external file.

        Args:
            format: File format (e.g., "parquet")
            file_path: Path to the file
            properties: Optional configuration properties (e.g., S3 credentials)

        Returns:
            Number of rows in the file

        Raises:
            InvalidArgumentError: If arguments are invalid
            FFIError: If operation fails

        Example:
            >>> num_rows = ExternalTable.get_file_info(
            ...     format="parquet",
            ...     file_path="s3://bucket/data.parquet",
            ...     properties={"fs.access_key_id": "...", ...}
            ... )
        """
        lib = get_library().lib
        ffi = get_ffi()

        props = Properties(properties) if properties else Properties()

        format_bytes = format.encode("utf-8")
        file_path_bytes = file_path.encode("utf-8")

        num_rows = ffi.new("uint64_t*")

        result = lib.loon_exttable_get_file_info(
            format_bytes,
            file_path_bytes,
            props._get_c_properties(),
            num_rows,
        )
        check_result(result)

        return num_rows[0]

    @staticmethod
    def read_manifest(
        manifest_file_path: str,
        properties: Optional[Dict[str, str]] = None,
    ) -> Manifest:
        """
        Read a manifest from a file.

        Args:
            manifest_file_path: Path to the manifest file
            properties: Optional configuration properties (e.g., S3 credentials)

        Returns:
            Manifest object containing column groups, delta logs, and stats

        Raises:
            InvalidArgumentError: If arguments are invalid
            FFIError: If operation fails

        Example:
            >>> manifest = ExternalTable.read_manifest(
            ...     "/data/dataset/manifest.json"
            ... )
            >>> print(f"Column groups: {len(manifest.column_groups)}")
        """
        lib = get_library().lib
        ffi = get_ffi()

        props = Properties(properties) if properties else Properties()

        manifest_path_bytes = manifest_file_path.encode("utf-8")

        manifest_ptr = ffi.new("LoonManifest**")

        result = lib.loon_exttable_read_manifest(
            manifest_path_bytes,
            props._get_c_properties(),
            manifest_ptr,
        )
        check_result(result)

        c_manifest = manifest_ptr[0]
        manifest = Manifest._from_c(c_manifest, ffi)

        # Free C manifest
        lib.loon_manifest_destroy(c_manifest)

        return manifest
