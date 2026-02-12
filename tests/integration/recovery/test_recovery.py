"""
Recovery tests with fault injection.

Core philosophy:
  1. After a write/transaction failure, previously committed data stays intact.
  2. After a read failure, retrying the read should succeed.
  3. Tests should set up real committed data first, then inject faults.
  4. Use parametrize to cover multiple fault points with the same test logic.
  5. S3-specific fault keys are auto-skipped when backend is not S3-compatible.

Requires BUILD_WITH_FIU=ON.
"""

import pyarrow as pa
import pytest
from milvus_storage import PropertyKeys, Reader, Transaction, Writer
from milvus_storage.fiu import FaultInjector
from milvus_storage.manifest import ColumnGroups

# ---------------------------------------------------------------------------
# Fault key groups
# ---------------------------------------------------------------------------

# Write failure keys that trigger during Writer.write()
_WRITE_FAIL_KEYS = [
    "WRITER_WRITE_FAIL",
    # Note: FS_OPEN_OUTPUT_FAIL only works on local filesystem.
    # S3 backend uses its own CustomOutputStream that bypasses FileSystemProxy::OpenOutputStream.
    "COLUMN_GROUP_WRITE_FAIL",
]

# Filesystem-level keys that only work on local backend
_LOCAL_FS_FAIL_KEYS = [
    "FS_OPEN_OUTPUT_FAIL",
]

# S3 multipart upload failure keys - these trigger during multipart upload operations
# which may happen during write (when buffer fills) or close (final flush)
_S3_MULTIPART_FAIL_KEYS = [
    "S3FS_CREATE_UPLOAD_FAIL",
    "S3FS_PART_UPLOAD_FAIL",
    "S3FS_COMPLETE_UPLOAD_FAIL",
]

_COMMIT_FAIL_KEYS = [
    "MANIFEST_COMMIT_FAIL",
    "MANIFEST_WRITE_FAIL",
]

_CHUNK_READ_FAIL_KEYS = [
    "COLUMN_GROUP_READ_FAIL",
    "CHUNK_READER_READ_FAIL",
    # Note: S3FS_READ_FAIL is not included because Parquet Reader uses ReadAt (random access)
    # instead of Read (sequential), so S3FS_READ_FAIL would never trigger.
    "S3FS_READAT_FAIL",
]

_SCAN_READ_FAIL_KEYS = [
    "FS_OPEN_INPUT_FAIL",
    "READER_OPEN_FAIL",
]


_S3_FAULT_KEYS = {
    "S3FS_CREATE_UPLOAD_FAIL",
    "S3FS_PART_UPLOAD_FAIL",
    "S3FS_COMPLETE_UPLOAD_FAIL",
    "S3FS_READ_FAIL",
    "S3FS_READAT_FAIL",
}


def _is_s3_key(name: str) -> bool:
    """Check if a fault key name is S3-specific."""
    return name in _S3_FAULT_KEYS


def _get_fiu_key(name: str) -> str:
    """Resolve a FaultInjector attribute name to the actual fault key string."""
    return getattr(FaultInjector, name)


def _skip_if_not_s3(test_config, fault_key: str):
    """Skip the test if fault_key is S3-specific but backend is not S3-compatible."""
    if _is_s3_key(fault_key) and not test_config.is_s3_compatible:
        pytest.skip(f"{fault_key} requires S3-compatible backend")


@pytest.mark.fiu
class TestRecovery:
    """Test data integrity and recoverability under injected faults."""

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------

    def _commit_data(self, path, schema, batch_gen, props, rows=1000):
        """Write data and commit via transaction. Return ColumnGroups (C ptr)."""
        writer = Writer(path, schema, props)
        writer.write(batch_gen(rows, offset=0))
        cg = writer.close()

        txn = Transaction(path, props)
        txn.append_files(cg)
        txn.commit()
        txn.close()
        return cg

    def _read_all_rows_via_manifest(self, path, schema, props):
        """Open a fresh Transaction, read manifest, verify data. Return row count."""
        txn = Transaction(path, props)
        manifest = txn.get_manifest()
        txn.close()

        cg = ColumnGroups.from_list(manifest.column_groups)
        reader = Reader(cg, schema, properties=props)
        return sum(b.num_rows for b in reader.scan())

    # -----------------------------------------------------------------
    # 1. Write failures don't corrupt existing data
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("fault_key", _WRITE_FAIL_KEYS)
    def test_write_fail_preserves_committed_data(
        self,
        require_fiu: FaultInjector,
        test_config,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        fault_key: str,
    ):
        """After a write-path failure, previously committed data is intact."""
        _skip_if_not_s3(test_config, fault_key)

        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=500,
        )

        require_fiu.enable(_get_fiu_key(fault_key), one_time=True)
        writer2 = Writer(temp_case_path, simple_schema, default_properties)
        with pytest.raises(Exception):
            writer2.write(batch_generator(200, offset=500))

        assert (
            self._read_all_rows_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            == 500
        )

    @pytest.mark.parametrize("fault_key", _LOCAL_FS_FAIL_KEYS)
    def test_local_fs_fail_preserves_committed_data(
        self,
        require_fiu: FaultInjector,
        test_config,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        fault_key: str,
    ):
        """Local filesystem failures don't corrupt existing data.

        NOTE: These FIU keys only work on local filesystem backend.
        S3 uses CustomOutputStream which bypasses FileSystemProxy::OpenOutputStream.
        """
        if test_config.is_s3_compatible:
            pytest.skip(f"{fault_key} only works on local filesystem backend")

        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=500,
        )

        require_fiu.enable(_get_fiu_key(fault_key), one_time=True)
        writer2 = Writer(temp_case_path, simple_schema, default_properties)
        with pytest.raises(Exception):
            writer2.write(batch_generator(200, offset=500))

        assert (
            self._read_all_rows_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            == 500
        )

    @pytest.mark.parametrize("fault_key", _S3_MULTIPART_FAIL_KEYS)
    @pytest.mark.skip(
        reason="S3 multipart upload requires min 10MB data to trigger. "
        "These FIU points are tested in C++ unit tests with mock S3 client."
    )
    def test_s3_multipart_fail_preserves_committed_data(
        self,
        require_fiu: FaultInjector,
        test_config,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        fault_key: str,
    ):
        """S3 multipart upload failures don't corrupt existing data.

        NOTE: This test is skipped because:
        1. S3 multipart upload minimum part size is 5MB (AWS requirement)
        2. multi_part_upload_size property minimum is 10MB
        3. Writing 10MB+ data for each test is too slow for CI
        4. These FIU points (CREATE_UPLOAD, PART_UPLOAD, COMPLETE_UPLOAD)
           are properly tested in C++ unit tests with mock S3 client.
        """
        _skip_if_not_s3(test_config, fault_key)

        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=500,
        )

        # Minimum multipart upload size is 10MB
        s3_props = default_properties.copy()
        s3_props[PropertyKeys.FS_MULTI_PART_UPLOAD_SIZE] = str(10 * 1024 * 1024)

        require_fiu.enable(_get_fiu_key(fault_key), one_time=True)
        writer2 = Writer(temp_case_path, simple_schema, s3_props)

        # Would need to write 10MB+ to trigger multipart upload
        exception_raised = False
        try:
            # This would need many more rows to exceed 10MB threshold
            writer2.write(batch_generator(500, offset=500))
            writer2.close()
        except Exception:
            exception_raised = True

        assert exception_raised, f"Expected {fault_key} to raise exception"
        assert (
            self._read_all_rows_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            == 500
        )

    @pytest.mark.parametrize("fault_key", _COMMIT_FAIL_KEYS)
    def test_commit_fail_preserves_committed_data(
        self,
        require_fiu: FaultInjector,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        fault_key: str,
    ):
        """After a commit-path failure, previously committed data is intact."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=600,
        )

        writer2 = Writer(temp_case_path, simple_schema, default_properties)
        writer2.write(batch_generator(400, offset=600))
        cg2 = writer2.close()

        require_fiu.enable(_get_fiu_key(fault_key), one_time=True)
        txn = Transaction(temp_case_path, default_properties)
        txn.append_files(cg2)
        with pytest.raises(Exception):
            txn.commit()
        txn.close()

        assert (
            self._read_all_rows_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            == 600
        )

    def test_close_fail_preserves_committed_data(
        self,
        require_fiu: FaultInjector,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """After Writer.close() failure, previously committed data is intact."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=800,
        )

        require_fiu.enable(FaultInjector.WRITER_CLOSE_FAIL, one_time=True)
        writer2 = Writer(temp_case_path, simple_schema, default_properties)
        writer2.write(batch_generator(300, offset=800))
        with pytest.raises(Exception):
            writer2.close()

        assert (
            self._read_all_rows_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            == 800
        )

    # -----------------------------------------------------------------
    # 2. Read failures are transient — retry succeeds
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("fault_key", _CHUNK_READ_FAIL_KEYS)
    def test_chunk_read_fail_then_retry_succeeds(
        self,
        require_fiu: FaultInjector,
        test_config,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        fault_key: str,
    ):
        """A one-time chunk-level read fault fails once, then retry succeeds."""
        _skip_if_not_s3(test_config, fault_key)

        cg = self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=500,
        )

        reader = Reader(cg, simple_schema, properties=default_properties)
        chunk_reader = reader.get_chunk_reader(0)

        require_fiu.enable(_get_fiu_key(fault_key), one_time=True)

        with pytest.raises(Exception):
            chunk_reader.get_chunk(0)

        chunk = chunk_reader.get_chunk(0)
        assert chunk.num_rows > 0
        chunk_reader.close()

    @pytest.mark.parametrize("fault_key", _SCAN_READ_FAIL_KEYS)
    def test_scan_read_fail_then_retry_succeeds(
        self,
        require_fiu: FaultInjector,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        fault_key: str,
    ):
        """A one-time scan/open-level read fault fails once, then retry succeeds."""
        cg = self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=500,
        )

        require_fiu.enable(_get_fiu_key(fault_key), one_time=True)

        with pytest.raises(Exception):
            reader = Reader(cg, simple_schema, properties=default_properties)
            list(reader.scan())

        reader2 = Reader(cg, simple_schema, properties=default_properties)
        total = sum(b.num_rows for b in reader2.scan())
        assert total == 500

    def test_take_fail_then_retry_succeeds(
        self,
        require_fiu: FaultInjector,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """A one-time take fault fails once, then retry returns correct rows."""
        cg = self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=500,
        )

        reader = Reader(cg, simple_schema, properties=default_properties)
        indices = [0, 10, 50, 100, 499]

        require_fiu.enable(FaultInjector.TAKE_ROWS_FAIL, one_time=True)

        with pytest.raises(Exception):
            reader.take(indices)

        batches = reader.take(indices)
        total = sum(b.num_rows for b in batches)
        assert total == len(indices)

    def test_manifest_read_fail_then_retry_succeeds(
        self,
        require_fiu: FaultInjector,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Manifest read failure is transient; retrying reads manifest successfully."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=600,
        )

        require_fiu.enable(FaultInjector.MANIFEST_READ_FAIL, one_time=True)

        with pytest.raises(Exception):
            txn = Transaction(temp_case_path, default_properties)
            txn.get_manifest()

        assert (
            self._read_all_rows_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            == 600
        )

    @pytest.mark.parametrize("fault_key", _CHUNK_READ_FAIL_KEYS)
    def test_persistent_read_fault_blocks_until_disabled(
        self,
        require_fiu: FaultInjector,
        test_config,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        fault_key: str,
    ):
        """Persistent fault blocks all reads; disabling restores access to data."""
        _skip_if_not_s3(test_config, fault_key)

        cg = self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=800,
        )

        reader = Reader(cg, simple_schema, properties=default_properties)
        chunk_reader = reader.get_chunk_reader(0)

        fiu_key = _get_fiu_key(fault_key)
        require_fiu.enable(fiu_key, one_time=False)

        for _ in range(3):
            with pytest.raises(Exception):
                chunk_reader.get_chunk(0)

        require_fiu.disable(fiu_key)
        chunk = chunk_reader.get_chunk(0)
        assert chunk.num_rows > 0
        chunk_reader.close()

    # -----------------------------------------------------------------
    # 3. Mixed scenarios
    # -----------------------------------------------------------------

    def test_failed_append_then_successful_append(
        self,
        require_fiu: FaultInjector,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """A failed commit does not block a subsequent successful commit."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=500,
        )

        writer2 = Writer(temp_case_path, simple_schema, default_properties)
        writer2.write(batch_generator(300, offset=500))
        cg2 = writer2.close()

        require_fiu.enable(FaultInjector.MANIFEST_COMMIT_FAIL, one_time=True)
        txn1 = Transaction(temp_case_path, default_properties)
        txn1.append_files(cg2)
        with pytest.raises(Exception):
            txn1.commit()
        txn1.close()

        txn2 = Transaction(temp_case_path, default_properties)
        txn2.append_files(cg2)
        txn2.commit()
        txn2.close()

        assert (
            self._read_all_rows_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            == 800
        )

    def test_disable_all_restores_full_functionality(
        self,
        require_fiu: FaultInjector,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """disable_all clears all faults; full write-commit-read cycle works."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=400,
        )

        require_fiu.enable(FaultInjector.WRITER_WRITE_FAIL, one_time=False)
        require_fiu.enable(FaultInjector.COLUMN_GROUP_READ_FAIL, one_time=False)
        require_fiu.enable(FaultInjector.MANIFEST_COMMIT_FAIL, one_time=False)

        require_fiu.disable_all()

        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(600, offset=400))
        cg_new = writer.close()

        txn = Transaction(temp_case_path, default_properties)
        txn.append_files(cg_new)
        txn.commit()
        txn.close()

        assert (
            self._read_all_rows_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            == 1000
        )
