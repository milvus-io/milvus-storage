"""
Long-running stress tests.

Verify stability over extended operation.
All writes go through Transaction, all reads verify via scan, chunk reader, and take.
"""

import random

import pyarrow as pa
import pytest
from milvus_storage import Reader, Transaction, Writer
from milvus_storage.manifest import ColumnGroups


def _commit_cg(path, cg, props):
    """Commit a ColumnGroups via transaction."""
    txn = Transaction(path, props)
    txn.append_files(cg)
    txn.commit()
    txn.close()


def _read_via_manifest(path, schema, props):
    """Open transaction, read manifest, return Reader."""
    txn = Transaction(path, props)
    manifest = txn.get_manifest()
    txn.close()
    cg = ColumnGroups.from_list(manifest.column_groups)
    return Reader(cg, schema, properties=props)


def _verify_read(reader, expected_rows, rng):
    """Verify data through scan, chunk reader, and take."""
    # 1. full scan
    scan_rows = sum(b.num_rows for b in reader.scan())
    assert (
        scan_rows == expected_rows
    ), f"scan: expected {expected_rows}, got {scan_rows}"

    # 2. chunk reader
    chunk_reader = reader.get_chunk_reader(0)
    num_chunks = chunk_reader.get_number_of_chunks()
    assert num_chunks > 0

    single = chunk_reader.get_chunk(0)
    assert single.num_rows > 0

    if num_chunks >= 2:
        sample_size = min(num_chunks, rng.randint(2, min(num_chunks, 8)))
        indices = rng.sample(range(num_chunks), sample_size)
        multi = chunk_reader.get_chunks(indices)
        assert len(multi) == sample_size
        for b in multi:
            assert b.num_rows > 0
    chunk_reader.close()

    # 3. take
    for take_count in [100, 500]:
        actual_take = min(take_count, expected_rows)
        take_indices = sorted(rng.sample(range(expected_rows), actual_take))
        taken = reader.take(take_indices)
        taken_rows = sum(b.num_rows for b in taken)
        assert taken_rows == actual_take
    reader.close()


@pytest.mark.stress
@pytest.mark.slow
class TestLongRunning:
    """Test long-running stability scenarios.

    All hardcoded iteration counts are scaled via stress_params fixture.
    Use --stress-scale=0.01 for quick validation (~1% of default values).
    """

    def test_continuous_write_read_cycles(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        stress_params,
    ):
        """Continuous write/read cycles via transaction for stability."""
        # Default: 1000 cycles, scales down with --stress-scale
        num_cycles = stress_params["long_running_cycles"]
        rows_per_cycle = stress_params["rows_per_cycle"]
        rng = random.Random(42)

        for cycle in range(num_cycles):
            # Write and commit via transaction
            writer = Writer(temp_case_path, simple_schema, default_properties)
            writer.write(batch_generator(rows_per_cycle, offset=cycle * rows_per_cycle))
            cg = writer.close()
            _commit_cg(temp_case_path, cg, default_properties)

            # Read via manifest with full verification
            reader = _read_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            expected_rows = (cycle + 1) * rows_per_cycle
            _verify_read(reader, expected_rows, rng)

    def test_manifest_growth_over_time(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        stress_params,
    ):
        """Manifest grows correctly over many appends."""
        # Initial write
        writer = Writer(temp_case_path, simple_schema, default_properties)
        initial_rows = stress_params["manifest_initial_rows"]
        writer.write(batch_generator(initial_rows, offset=0))
        init_cg = writer.close()
        _commit_cg(temp_case_path, init_cg, default_properties)

        # Default: 500 appends, scales down with --stress-scale
        append_rows = stress_params["manifest_append_rows"]
        num_appends = stress_params["manifest_appends"]
        for i in range(num_appends):
            new_writer = Writer(temp_case_path, simple_schema, default_properties)
            new_writer.write(batch_generator(append_rows, offset=(i + 1) * 1000))
            new_cg = new_writer.close()
            _commit_cg(temp_case_path, new_cg, default_properties)

        # Verify final manifest
        txn = Transaction(temp_case_path, default_properties)
        manifest = txn.get_manifest()
        version = txn.get_read_version()
        txn.close()

        total_files = sum(len(cg.files) for cg in manifest.column_groups)
        assert total_files >= num_appends + 1
        assert version > 0

        # Full read verification
        reader = _read_via_manifest(temp_case_path, simple_schema, default_properties)
        expected_rows = initial_rows + num_appends * append_rows
        _verify_read(reader, expected_rows, random.Random(42))

    def test_repeated_create_destroy(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        stress_params,
    ):
        """Repeatedly create and destroy writers/readers via transaction."""
        # Default: 2000 iterations, scales down with --stress-scale
        num_iterations = stress_params["create_destroy_iterations"]
        rng = random.Random(42)

        for i in range(num_iterations):
            path = f"{temp_case_path}/iter_{i}"
            writer = Writer(path, simple_schema, default_properties)
            rows = stress_params["create_destroy_rows"]
            writer.write(batch_generator(rows, offset=0))
            cg = writer.close()
            _commit_cg(path, cg, default_properties)
            del writer

            reader = _read_via_manifest(path, simple_schema, default_properties)
            _verify_read(reader, rows, rng)

    def test_many_small_transactions(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        stress_params,
    ):
        """Many small read-only transactions don't cause resource leaks."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        init_rows = stress_params["small_txn_init_rows"]
        writer.write(batch_generator(init_rows, offset=0))
        init_cg = writer.close()
        _commit_cg(temp_case_path, init_cg, default_properties)

        # Default: 1000 transactions, scales down with --stress-scale
        num_transactions = stress_params["small_transactions"]
        rng = random.Random(42)
        for i in range(num_transactions):
            # Each iteration: open transaction, read manifest, verify, close
            reader = _read_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            _verify_read(reader, init_rows, rng)

        # Final check - can still open transactions
        txn = Transaction(temp_case_path, default_properties)
        manifest = txn.get_manifest()
        txn.close()
        assert len(manifest.column_groups) > 0
