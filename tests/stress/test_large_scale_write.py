"""
Large-scale write stress tests.

Verify correctness and stability under large data volumes.
All writes go through Transaction with optional periodic close/reopen.
All reads verify via scan, chunk reader, and take.
"""

import random
from typing import Callable, Dict

import pyarrow as pa
import pytest
from milvus_storage import Reader, Transaction, Writer
from milvus_storage.manifest import ColumnGroups


def _verify_read(reader, expected_rows, schema, rng):
    """Verify data through three read paths.

    1. Full scan via RecordBatch reader
    2. Chunk reader: get_chunk (single) + get_chunks (random sample)
    3. Take: 100, 500, 1000 random row indices
    """
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
    for take_count in [100, 500, 1000]:
        actual_take = min(take_count, expected_rows)
        take_indices = sorted(rng.sample(range(expected_rows), actual_take))
        taken = reader.take(take_indices)
        taken_rows = sum(b.num_rows for b in taken)
        assert (
            taken_rows == actual_take
        ), f"take({take_count}): expected {actual_take}, got {taken_rows}"

    reader.close()


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


def _segmented_write(
    path: str,
    schema: pa.Schema,
    props: Dict,
    total_batches: int,
    make_batch: Callable[[int], pa.RecordBatch],
    num_closes: int,
):
    """Write batches with optional periodic close/reopen and commit via transaction.

    Batches are generated lazily via make_batch(batch_index) to avoid
    holding all data in memory.

    Args:
        total_batches: Total number of batches to write.
        make_batch: Callable that takes batch index and returns a RecordBatch.
        num_closes: Number of intermediate close/reopen cycles.
            0 = single writer session, close once at the end (file rolling still applies).
            N>0 = split batches into N+1 segments, close and commit after each segment,
                   then reopen a new writer for the next segment.
    """
    if num_closes == 0:
        writer = Writer(path, schema, props)
        for i in range(total_batches):
            writer.write(make_batch(i))
        cg = writer.close()
        _commit_cg(path, cg, props)
    else:
        num_segments = num_closes + 1
        seg_size = total_batches // num_segments

        for seg in range(num_segments):
            start = seg * seg_size
            end = (seg + 1) * seg_size if seg < num_segments - 1 else total_batches

            writer = Writer(path, schema, props)
            for i in range(start, end):
                writer.write(make_batch(i))
            cg = writer.close()
            _commit_cg(path, cg, props)


# Common parametrize: 0 = no intermediate close, 2/4 = close and reopen 2/4 times
_closes_param = pytest.mark.parametrize("num_closes", [0, 2, 4])


@pytest.mark.stress
class TestLargeScaleWrite:
    """Test large-scale write operations."""

    @_closes_param
    def test_large_row_count(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        num_rows,
        num_closes,
    ):
        """Write and verify large numbers of rows."""
        # Adaptive batch_size: use at most 10k, but not more than num_rows
        batch_size = min(10_000, num_rows)
        total_batches = max(1, num_rows // batch_size)
        actual_rows = total_batches * batch_size

        _segmented_write(
            temp_case_path,
            simple_schema,
            default_properties,
            total_batches=total_batches,
            make_batch=lambda i: batch_generator(batch_size, offset=i * batch_size),
            num_closes=num_closes,
        )

        reader = _read_via_manifest(temp_case_path, simple_schema, default_properties)
        _verify_read(reader, actual_rows, simple_schema, random.Random(42))

    @_closes_param
    def test_wide_table(
        self,
        temp_case_path: str,
        default_properties,
        num_columns,
        num_closes,
        stress_params,
    ):
        """Write wide table with many columns."""
        fields = [pa.field(f"col_{i}", pa.float64()) for i in range(num_columns)]
        schema = pa.schema(fields)

        num_rows = stress_params["wide_table_rows"]
        rows_per_batch = min(stress_params["default_rows_per_batch"], num_rows)

        def make_batch(idx):
            offset = idx * rows_per_batch
            data = {
                f"col_{i}": [float(offset + j) for j in range(rows_per_batch)]
                for i in range(num_columns)
            }
            return pa.RecordBatch.from_pydict(data, schema=schema)

        _segmented_write(
            temp_case_path,
            schema,
            default_properties,
            total_batches=num_rows // rows_per_batch,
            make_batch=make_batch,
            num_closes=num_closes,
        )

        reader = _read_via_manifest(temp_case_path, schema, default_properties)
        _verify_read(reader, num_rows, schema, random.Random(42))

    @_closes_param
    def test_large_string_values(
        self,
        temp_case_path: str,
        default_properties,
        num_closes,
        str_size,
        stress_params,
    ):
        """Write large string values with varying string length."""
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("text", pa.string()),
            ]
        )

        large_text = "A" * str_size
        num_rows = stress_params["large_string_rows"]
        rows_per_batch = min(stress_params["default_rows_per_batch"], num_rows)

        def make_batch(idx):
            offset = idx * rows_per_batch
            return pa.RecordBatch.from_pydict(
                {
                    "id": list(range(offset, offset + rows_per_batch)),
                    "text": [large_text] * rows_per_batch,
                },
                schema=schema,
            )

        _segmented_write(
            temp_case_path,
            schema,
            default_properties,
            total_batches=num_rows // rows_per_batch,
            make_batch=make_batch,
            num_closes=num_closes,
        )

        reader = _read_via_manifest(temp_case_path, schema, default_properties)
        _verify_read(reader, num_rows, schema, random.Random(42))

    @pytest.mark.xfail(
        reason="Parquet loses fixed_size_list semantics",
        raises=Exception,
    )
    def test_large_vector_data(
        self,
        temp_case_path: str,
        default_properties,
        stress_params,
    ):
        """Write large vector data (128-dim float32, ~512 bytes/row)."""
        rng = random.Random(42)

        dim = 128
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )

        num_rows = stress_params["vector_data_rows"]
        rows_per_batch = min(stress_params["default_rows_per_batch"], num_rows)

        def make_batch(idx):
            offset = idx * rows_per_batch
            vectors = [
                [rng.random() for _ in range(dim)] for _ in range(rows_per_batch)
            ]
            return pa.RecordBatch.from_pydict(
                {
                    "id": list(range(offset, offset + rows_per_batch)),
                    "vector": vectors,
                },
                schema=schema,
            )

        _segmented_write(
            temp_case_path,
            schema,
            default_properties,
            total_batches=num_rows // rows_per_batch,
            make_batch=make_batch,
            num_closes=0,
        )

        reader = _read_via_manifest(temp_case_path, schema, default_properties)
        _verify_read(reader, num_rows, schema, rng)
