"""
Fault Injection Utilities for milvus-storage.

This module provides Python wrappers for the libfiu-based fault injection
functionality. Fault injection is used for testing error handling and
recovery scenarios.

Example usage:
    from milvus_storage.fiu import FaultInjector

    # Create injector
    fiu = FaultInjector()

    # Check if fault injection is enabled
    if fiu.is_enabled():
        # Enable a fault point to fail once
        fiu.enable(FaultInjector.WRITER_FLUSH_FAIL, one_time=True)

        # The next flush will fail
        try:
            writer.flush()
        except IOError:
            pass

        # Retry should succeed (failnum exhausted)
        writer.flush()

        # Cleanup
        fiu.disable_all()

Available fault points (use FaultInjector class constants):
    Writer fault points:
    - FaultInjector.WRITER_WRITE_FAIL: Fail during Writer write batch operation
    - FaultInjector.WRITER_FLUSH_FAIL: Fail during Writer flush
    - FaultInjector.WRITER_CLOSE_FAIL: Fail during Writer close

    Reader fault points (low-level):
    - FaultInjector.COLUMN_GROUP_READ_FAIL: Fail during ColumnGroup get_chunk/get_chunks
    - FaultInjector.TAKE_ROWS_FAIL: Fail during take rows operation
    - FaultInjector.CHUNK_READER_READ_FAIL: Fail during ChunkReader read (FFI layer)
    - FaultInjector.READER_OPEN_FAIL: Fail during Reader open (FFI layer)

    Transaction/Manifest fault points:
    - FaultInjector.MANIFEST_COMMIT_FAIL: Fail during Transaction commit
    - FaultInjector.MANIFEST_READ_FAIL: Fail during manifest read
    - FaultInjector.MANIFEST_WRITE_FAIL: Fail during manifest write/serialize

    Filesystem fault points:
    - FaultInjector.FS_OPEN_OUTPUT_FAIL: Fail during filesystem open output stream
    - FaultInjector.FS_OPEN_INPUT_FAIL: Fail during filesystem open input file

    S3 Filesystem fault points:
    - FaultInjector.S3FS_CREATE_UPLOAD_FAIL: Fail during S3 CreateMultipartUpload
    - FaultInjector.S3FS_PART_UPLOAD_FAIL: Fail during S3 multipart upload UploadPart
    - FaultInjector.S3FS_COMPLETE_UPLOAD_FAIL: Fail during S3 CompleteMultipartUpload
    - FaultInjector.S3FS_READ_FAIL: Fail during S3 ObjectInputFile Read
    - FaultInjector.S3FS_READAT_FAIL: Fail during S3 ObjectInputFile ReadAt

    ColumnGroup fault points:
    - FaultInjector.COLUMN_GROUP_WRITE_FAIL: Fail during ColumnGroup write operation
"""

from ._ffi import _ffi, check_result, get_library


def _get_fiu_key(lib, name: str) -> str:
    """Get fault point key string from FFI library."""
    ptr = getattr(lib, name)
    return _ffi.string(ptr).decode("utf-8")


class FaultInjector:
    """
    Fault injection controller for testing error handling.

    This class provides methods to enable/disable fault injection points
    in the milvus-storage C++ library. Fault injection must be enabled
    at compile time with -DWITH_FIU=ON.
    """

    # Fault point keys are loaded from FFI library at class initialization time
    _keys_loaded = False

    # Writer fault points
    WRITER_WRITE_FAIL: str = ""
    WRITER_FLUSH_FAIL: str = ""
    WRITER_CLOSE_FAIL: str = ""

    # Reader fault points (low-level)
    COLUMN_GROUP_READ_FAIL: str = ""
    TAKE_ROWS_FAIL: str = ""
    CHUNK_READER_READ_FAIL: str = ""
    READER_OPEN_FAIL: str = ""

    # Transaction/Manifest fault points
    MANIFEST_COMMIT_FAIL: str = ""
    MANIFEST_READ_FAIL: str = ""
    MANIFEST_WRITE_FAIL: str = ""

    # Filesystem fault points
    FS_OPEN_OUTPUT_FAIL: str = ""
    FS_OPEN_INPUT_FAIL: str = ""

    # S3 Filesystem fault points
    S3FS_CREATE_UPLOAD_FAIL: str = ""
    S3FS_PART_UPLOAD_FAIL: str = ""
    S3FS_COMPLETE_UPLOAD_FAIL: str = ""
    S3FS_READ_FAIL: str = ""
    S3FS_READAT_FAIL: str = ""

    # ColumnGroup fault points
    COLUMN_GROUP_WRITE_FAIL: str = ""

    @classmethod
    def _load_keys(cls) -> None:
        """Load fault point keys from FFI library (called once)."""
        if cls._keys_loaded:
            return
        lib = get_library()
        # Writer fault points
        cls.WRITER_WRITE_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_writer_write_fail")
        cls.WRITER_FLUSH_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_writer_flush_fail")
        cls.WRITER_CLOSE_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_writer_close_fail")
        # Reader fault points
        cls.COLUMN_GROUP_READ_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_column_group_read_fail")
        cls.TAKE_ROWS_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_take_rows_fail")
        cls.CHUNK_READER_READ_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_chunk_reader_read_fail")
        cls.READER_OPEN_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_reader_open_fail")
        # Transaction/Manifest fault points
        cls.MANIFEST_COMMIT_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_manifest_commit_fail")
        cls.MANIFEST_READ_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_manifest_read_fail")
        cls.MANIFEST_WRITE_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_manifest_write_fail")
        # Filesystem fault points
        cls.FS_OPEN_OUTPUT_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_fs_open_output_fail")
        cls.FS_OPEN_INPUT_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_fs_open_input_fail")
        # S3 Filesystem fault points
        cls.S3FS_CREATE_UPLOAD_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_s3fs_create_upload_fail")
        cls.S3FS_PART_UPLOAD_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_s3fs_part_upload_fail")
        cls.S3FS_COMPLETE_UPLOAD_FAIL = _get_fiu_key(
            lib.lib, "loon_fiukey_s3fs_complete_upload_fail"
        )
        cls.S3FS_READ_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_s3fs_read_fail")
        cls.S3FS_READAT_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_s3fs_readat_fail")
        # ColumnGroup fault points
        cls.COLUMN_GROUP_WRITE_FAIL = _get_fiu_key(lib.lib, "loon_fiukey_column_group_write_fail")
        cls._keys_loaded = True

    def __init__(self):
        """Initialize the fault injector."""
        self._lib = get_library()
        # Load fault point keys from FFI library
        self._load_keys()

    def is_enabled(self) -> bool:
        """
        Check if fault injection support is compiled in.

        Returns:
            True if FIU is enabled, False otherwise.
        """
        return bool(self._lib.lib.loon_fiu_is_enabled())

    def enable(self, name: str, one_time: bool = True) -> None:
        """
        Enable a fault injection point.

        Args:
            name: The name of the fault point to enable.
            one_time: If True, the fault triggers only once then auto-disables.
                      If False, the fault triggers forever until explicitly disabled.

        Raises:
            RuntimeError: If FIU is not enabled or if enabling fails.
        """
        if not self.is_enabled():
            raise RuntimeError(
                "Fault injection is not enabled. " "Rebuild the C++ library with -DWITH_FIU=ON"
            )

        name_bytes = name.encode("utf-8")
        result = self._lib.lib.loon_fiu_enable(name_bytes, len(name_bytes), 1 if one_time else 0)
        check_result(result)

    def disable(self, name: str) -> None:
        """
        Disable a specific fault injection point.

        Args:
            name: The name of the fault point to disable.

        Raises:
            RuntimeError: If FIU is not enabled or if disabling fails.
        """
        if not self.is_enabled():
            raise RuntimeError(
                "Fault injection is not enabled. " "Rebuild the C++ library with -DWITH_FIU=ON"
            )

        name_bytes = name.encode("utf-8")
        result = self._lib.lib.loon_fiu_disable(name_bytes, len(name_bytes))
        check_result(result)

    def disable_all(self) -> None:
        """
        Disable all active fault injection points.

        This should be called in test cleanup to ensure all fault points
        are disabled after a test completes.
        """
        self._lib.lib.loon_fiu_disable_all()

    def __enter__(self) -> "FaultInjector":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - disable all fault points."""
        self.disable_all()
        return None


# Convenience function for quick access
def is_fiu_enabled() -> bool:
    """
    Check if fault injection support is compiled in.

    Returns:
        True if FIU is enabled, False otherwise.
    """
    lib = get_library()
    return bool(lib.lib.loon_fiu_is_enabled())
