"""
Tests for Transaction class.
Tests basic transaction operations including begin, commit, and manifest operations.
"""

import shutil
import tempfile

import pyarrow as pa
import pytest

from milvus_storage import Transaction, Writer, destroy_column_groups
from milvus_storage.exceptions import InvalidArgumentError, ResourceError
from milvus_storage.transaction import ResolveStrategy


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_schema():
    """Create a sample schema."""
    return pa.schema(
        [
            pa.field("id", pa.int64(), metadata={"PARQUET:field_id": "1"}),
            pa.field("value", pa.float64(), metadata={"PARQUET:field_id": "2"}),
            pa.field("text", pa.string(), metadata={"PARQUET:field_id": "3"}),
        ]
    )


@pytest.fixture
def properties(temp_dir):
    """Create default properties for tests."""
    return {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }


# ============================================================================
# Basic Transaction Tests
# ============================================================================


def test_transaction_begin_and_close(temp_dir, properties):
    """Test basic transaction begin and close."""
    txn = Transaction(temp_dir, properties=properties)

    assert not txn.is_closed
    assert not txn.is_committed

    txn.close()

    assert txn.is_closed


def test_transaction_context_manager(temp_dir, properties):
    """Test transaction with context manager."""
    with Transaction(temp_dir, properties=properties) as txn:
        assert not txn.is_closed
        assert not txn.is_committed

    assert txn.is_closed


def test_transaction_get_read_version(temp_dir, properties):
    """Test getting read version from transaction."""
    with Transaction(temp_dir, properties=properties) as txn:
        version = txn.get_read_version()
        # For a new dataset, version should be -1 or 0
        assert isinstance(version, int)


def test_transaction_get_manifest(temp_dir, properties):
    """Test getting manifest from transaction."""
    with Transaction(temp_dir, properties=properties) as txn:
        manifest = txn.get_manifest()

        # Manifest should have empty lists for a new dataset
        assert hasattr(manifest, "column_groups")
        assert hasattr(manifest, "delta_logs")
        assert hasattr(manifest, "stats")
        assert isinstance(manifest.column_groups, list)
        assert isinstance(manifest.delta_logs, list)
        assert isinstance(manifest.stats, list)


def test_transaction_repr(temp_dir, properties):
    """Test transaction string representation."""
    with Transaction(temp_dir, properties=properties) as txn:
        repr_str = repr(txn)
        assert "Transaction" in repr_str
        assert "open" in repr_str

    # After close
    repr_str = repr(txn)
    assert "closed" in repr_str


# ============================================================================
# Transaction with Writer Integration Tests
# ============================================================================


def test_transaction_append_files(temp_dir, sample_schema, properties):
    """Test appending files to transaction."""
    # First write some data
    data = pa.record_batch(
        [
            [1, 2, 3, 4, 5],
            [1.1, 2.2, 3.3, 4.4, 5.5],
            ["a", "b", "c", "d", "e"],
        ],
        schema=sample_schema,
    )

    with Writer(temp_dir, sample_schema, properties=properties) as writer:
        writer.write(data)
        column_groups = writer.close()

    # Now use transaction to append
    with Transaction(temp_dir, properties=properties) as txn:
        txn.append_files(column_groups)
        version = txn.commit()

        assert txn.is_committed
        assert isinstance(version, int)

    # Clean up
    destroy_column_groups(column_groups)


def test_transaction_commit_returns_version(temp_dir, sample_schema, properties):
    """Test that commit returns a version number."""
    data = pa.record_batch(
        [
            [1, 2, 3],
            [1.1, 2.2, 3.3],
            ["x", "y", "z"],
        ],
        schema=sample_schema,
    )

    with Writer(temp_dir, sample_schema, properties=properties) as writer:
        writer.write(data)
        column_groups = writer.close()

    with Transaction(temp_dir, properties=properties) as txn:
        txn.append_files(column_groups)
        version = txn.commit()

        assert isinstance(version, int)
        assert version >= 0

    destroy_column_groups(column_groups)


# ============================================================================
# Transaction Delta Log and Stat Tests
# ============================================================================


def test_transaction_add_delta_log(temp_dir, sample_schema, properties):
    """Test adding delta log to transaction."""
    # Write initial data
    data = pa.record_batch(
        [[1, 2, 3], [1.1, 2.2, 3.3], ["a", "b", "c"]],
        schema=sample_schema,
    )

    with Writer(temp_dir, sample_schema, properties=properties) as writer:
        writer.write(data)
        column_groups = writer.close()

    with Transaction(temp_dir, properties=properties) as txn:
        txn.append_files(column_groups)
        txn.add_delta_log("delta/log_001.parquet", num_entries=10)
        version = txn.commit()

        assert isinstance(version, int)

    destroy_column_groups(column_groups)


def test_transaction_add_delta_log_invalid_entries(temp_dir, properties):
    """Test adding delta log with invalid num_entries."""
    with Transaction(temp_dir, properties=properties) as txn:
        with pytest.raises(InvalidArgumentError):
            txn.add_delta_log("delta/log.parquet", num_entries=-1)


def test_transaction_update_stat(temp_dir, sample_schema, properties):
    """Test updating stat in transaction."""
    data = pa.record_batch(
        [[1, 2, 3], [1.1, 2.2, 3.3], ["a", "b", "c"]],
        schema=sample_schema,
    )

    with Writer(temp_dir, sample_schema, properties=properties) as writer:
        writer.write(data)
        column_groups = writer.close()

    with Transaction(temp_dir, properties=properties) as txn:
        txn.append_files(column_groups)
        txn.update_stat("bloomfilter", ["stats/bloom_001.bin", "stats/bloom_002.bin"])
        version = txn.commit()

        assert isinstance(version, int)

    destroy_column_groups(column_groups)


def test_transaction_update_stat_empty_key(temp_dir, properties):
    """Test updating stat with empty key raises error."""
    with Transaction(temp_dir, properties=properties) as txn:
        with pytest.raises(InvalidArgumentError):
            txn.update_stat("", ["file.bin"])


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_transaction_operations_after_close(temp_dir, properties):
    """Test that operations after close raise errors."""
    txn = Transaction(temp_dir, properties=properties)
    txn.close()

    with pytest.raises(ResourceError):
        txn.get_manifest()

    with pytest.raises(ResourceError):
        txn.get_read_version()

    with pytest.raises(ResourceError):
        txn.commit()


def test_transaction_operations_after_commit(temp_dir, sample_schema, properties):
    """Test that operations after commit raise errors."""
    data = pa.record_batch(
        [[1, 2], [1.1, 2.2], ["a", "b"]],
        schema=sample_schema,
    )

    with Writer(temp_dir, sample_schema, properties=properties) as writer:
        writer.write(data)
        column_groups = writer.close()

    txn = Transaction(temp_dir, properties=properties)
    txn.append_files(column_groups)
    txn.commit()

    # After commit, cannot commit again
    with pytest.raises(ResourceError):
        txn.commit()

    # After commit, cannot append more files
    with pytest.raises(ResourceError):
        txn.append_files(column_groups)

    txn.close()
    destroy_column_groups(column_groups)


def test_transaction_double_close(temp_dir, properties):
    """Test that double close is safe."""
    txn = Transaction(temp_dir, properties=properties)
    txn.close()
    # Second close should be safe (no-op)
    txn.close()

    assert txn.is_closed


# ============================================================================
# ResolveStrategy Tests
# ============================================================================


def test_resolve_strategy_constants():
    """Test ResolveStrategy constants are defined."""
    assert hasattr(ResolveStrategy, "FAIL")
    assert hasattr(ResolveStrategy, "MERGE")
    assert hasattr(ResolveStrategy, "OVERWRITE")

    # Values should be distinct
    assert ResolveStrategy.FAIL != ResolveStrategy.MERGE
    assert ResolveStrategy.MERGE != ResolveStrategy.OVERWRITE
    assert ResolveStrategy.FAIL != ResolveStrategy.OVERWRITE
