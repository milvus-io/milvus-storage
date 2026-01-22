"""
Integration tests for Writer and Reader classes.
Tests the complete write/read cycle to verify data round-trips correctly.
"""

import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pytest

from milvus_storage import Reader, Writer
from milvus_storage.exceptions import InvalidArgumentError, ResourceError


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


# ============================================================================
# Write/Read Integration Tests
# ============================================================================


def test_write_read_single_batch(temp_dir, sample_schema):
    """Test writing and reading a single batch."""
    # Write data
    original_batch = pa.record_batch(
        [
            [1, 2, 3, 4, 5],
            [1.1, 2.2, 3.3, 4.4, 5.5],
            ["a", "b", "c", "d", "e"],
        ],
        schema=sample_schema,
    )

    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, sample_schema, properties=property) as writer:
        writer.write(original_batch)
        column_groups = writer.close()

    # Read data back
    with Reader(column_groups, sample_schema, properties=property) as reader:
        batch_reader = reader.scan()

        read_batches = list(batch_reader)
        assert len(read_batches) > 0

        # Combine all batches and verify data
        combined = pa.Table.from_batches(read_batches, schema=sample_schema)
        assert combined.num_rows == 5
        assert combined.column(0).to_pylist() == [1, 2, 3, 4, 5]
        assert combined.column(1).to_pylist() == [1.1, 2.2, 3.3, 4.4, 5.5]
        assert combined.column(2).to_pylist() == ["a", "b", "c", "d", "e"]


def test_write_read_multiple_batches(temp_dir, sample_schema):
    """Test writing and reading multiple batches."""
    # Write multiple batches
    batches_to_write = []
    for i in range(3):
        batch = pa.record_batch(
            [
                list(range(i * 10, (i + 1) * 10)),
                [float(j) * 1.1 for j in range(i * 10, (i + 1) * 10)],
                [f"text_{j}" for j in range(i * 10, (i + 1) * 10)],
            ],
            schema=sample_schema,
        )
        batches_to_write.append(batch)

    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, sample_schema, properties=property) as writer:
        for batch in batches_to_write:
            writer.write(batch)
        column_groups = writer.close()

    # Read data back
    with Reader(column_groups, sample_schema, properties=property) as reader:
        batch_reader = reader.scan()

        total_rows = 0
        all_ids = []
        all_values = []
        all_texts = []

        for batch in batch_reader:
            total_rows += len(batch)
            all_ids.extend(batch.column(0).to_pylist())
            all_values.extend(batch.column(1).to_pylist())
            all_texts.extend(batch.column(2).to_pylist())

        # Verify we got all 30 rows back
        assert total_rows == 30

        # Verify data integrity
        assert all_ids == list(range(30))
        assert all_values == [float(i) * 1.1 for i in range(30)]
        assert all_texts == [f"text_{i}" for i in range(30)]


@pytest.mark.skip(reason="take is not implemented")
def test_write_read_with_take(temp_dir, sample_schema):
    """Test write/read cycle using random access (take)."""
    # Write data
    original_data = pa.record_batch(
        [
            list(range(100)),
            [float(i) * 2.5 for i in range(100)],
            [f"item_{i}" for i in range(100)],
        ],
        schema=sample_schema,
    )

    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, sample_schema, properties=property) as writer:
        writer.write(original_data)
        column_groups = writer.close()

    # Read specific rows using take
    with Reader(column_groups, sample_schema, properties=property) as reader:
        indices = [0, 10, 25, 50, 99]
        batch = reader.take(indices)

        assert len(batch) == len(indices)
        assert batch.column(0).to_pylist() == [0, 10, 25, 50, 99]
        assert batch.column(1).to_pylist() == [0.0, 25.0, 62.5, 125.0, 247.5]
        assert batch.column(2).to_pylist() == ["item_0", "item_10", "item_25", "item_50", "item_99"]


@pytest.mark.skip(reason="take is not implemented")
def test_write_read_with_numpy_indices(temp_dir, sample_schema):
    """Test write/read with numpy array indices."""
    # Write data
    data = pa.record_batch(
        [
            [10, 20, 30, 40, 50],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            ["x", "y", "z", "w", "v"],
        ],
        schema=sample_schema,
    )

    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, sample_schema, properties=property) as writer:
        writer.write(data)
        column_groups = writer.close()

    # Read with numpy array indices
    with Reader(column_groups, sample_schema, properties=property) as reader:
        indices = np.array([1, 3, 4])
        batch = reader.take(indices)

        assert len(batch) == 3
        assert batch.column(0).to_pylist() == [20, 40, 50]
        assert batch.column(2).to_pylist() == ["y", "w", "v"]


def test_write_read_with_column_projection(temp_dir, sample_schema):
    """Test write/read with column projection."""
    # Write data
    data = pa.record_batch(
        [
            [1, 2, 3],
            [1.1, 2.2, 3.3],
            ["a", "b", "c"],
        ],
        schema=sample_schema,
    )

    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, sample_schema, properties=property) as writer:
        writer.write(data)
        column_groups = writer.close()

    # Read only specific columns
    columns = ["id", "text"]
    with Reader(column_groups, sample_schema, columns=columns, properties=property) as reader:
        batch_reader = reader.scan()

        for batch in batch_reader:
            # Verify we can read the projected columns
            assert batch.num_columns <= len(sample_schema)
            ids = batch.column(0).to_pylist()
            # Verify data is correct
            assert 1 in ids or 2 in ids or 3 in ids
            break


def test_write_flush_read_cycle(temp_dir, sample_schema):
    """Test write/flush/read cycle."""
    # Write with explicit flush
    batch1 = pa.record_batch(
        [
            [1, 2, 3],
            [1.1, 2.2, 3.3],
            ["a", "b", "c"],
        ],
        schema=sample_schema,
    )

    batch2 = pa.record_batch(
        [
            [4, 5, 6],
            [4.4, 5.5, 6.6],
            ["d", "e", "f"],
        ],
        schema=sample_schema,
    )

    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, sample_schema, properties=property) as writer:
        writer.write(batch1)
        writer.flush()
        writer.write(batch2)
        column_groups = writer.close()

    # Read back and verify both batches
    with Reader(column_groups, sample_schema, properties=property) as reader:
        batch_reader = reader.scan()

        all_ids = []
        for batch in batch_reader:
            all_ids.extend(batch.column(0).to_pylist())

        assert set(all_ids) == {1, 2, 3, 4, 5, 6}


def test_write_read_with_properties(temp_dir, sample_schema):
    """Test write/read with custom properties."""
    # Write with properties
    write_properties = {
        "storage.memory.limit": str(1024 * 1024 * 100),  # 100MB
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    data = pa.record_batch(
        [
            list(range(20)),
            [float(i) * 0.5 for i in range(20)],
            [f"row_{i}" for i in range(20)],
        ],
        schema=sample_schema,
    )

    with Writer(temp_dir, sample_schema, properties=write_properties) as writer:
        writer.write(data)
        column_groups = writer.close()

    # Read with properties
    read_properties = {
        "storage.batch.size": "1024",
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Reader(column_groups, sample_schema, properties=read_properties) as reader:
        batch_reader = reader.scan()

        total_rows = sum(len(batch) for batch in batch_reader)
        assert total_rows == 20


def test_write_read_large_dataset(temp_dir, sample_schema):
    """Test write/read with larger dataset."""
    # Write 1000 rows across multiple batches
    batch_size = 100
    num_batches = 10

    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, sample_schema, properties=property) as writer:
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch = pa.record_batch(
                [
                    list(range(start, end)),
                    [float(j) * 0.1 for j in range(start, end)],
                    [f"data_{j}" for j in range(start, end)],
                ],
                schema=sample_schema,
            )
            writer.write(batch)
        column_groups = writer.close()

    # Read back and verify count
    with Reader(column_groups, sample_schema, properties=property) as reader:
        batch_reader = reader.scan()
        total_rows = sum(len(batch) for batch in batch_reader)
        assert total_rows == 1000

    # TODO: Enable this test when take is implemented
    # # Also test random access
    # with Reader(column_groups, sample_schema, properties=property) as reader:
    #     sample_indices = [0, 100, 500, 999]
    #     batch = reader.take(sample_indices)
    #     assert len(batch) == len(sample_indices)
    #     assert batch.column(0).to_pylist() == sample_indices


def test_write_read_different_data_types(temp_dir):
    """Test write/read with various Arrow data types."""
    schema = pa.schema(
        [
            pa.field("int32_col", pa.int32(), metadata={"PARQUET:field_id": "1"}),
            pa.field("int64_col", pa.int64(), metadata={"PARQUET:field_id": "2"}),
            pa.field("float32_col", pa.float32(), metadata={"PARQUET:field_id": "3"}),
            pa.field("float64_col", pa.float64(), metadata={"PARQUET:field_id": "4"}),
            pa.field("string_col", pa.string(), metadata={"PARQUET:field_id": "5"}),
            pa.field("bool_col", pa.bool_(), metadata={"PARQUET:field_id": "6"}),
        ]
    )

    # Write data
    data = pa.record_batch(
        [
            [1, 2, 3],
            [10, 20, 30],
            [1.5, 2.5, 3.5],
            [10.5, 20.5, 30.5],
            ["x", "y", "z"],
            [True, False, True],
        ],
        schema=schema,
    )

    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, schema, properties=property) as writer:
        writer.write(data)
        column_groups = writer.close()

    # Read back and verify
    with Reader(column_groups, schema, properties=property) as reader:
        batch_reader = reader.scan()

        for batch in batch_reader:
            assert batch.column(0).to_pylist() == [1, 2, 3]
            assert batch.column(1).to_pylist() == [10, 20, 30]
            assert batch.column(4).to_pylist() == ["x", "y", "z"]
            assert batch.column(5).to_pylist() == [True, False, True]
            break


def test_write_read_context_managers(temp_dir, sample_schema):
    """Test write/read using context managers properly."""
    data = pa.record_batch(
        [
            [100, 200, 300],
            [1.0, 2.0, 3.0],
            ["foo", "bar", "baz"],
        ],
        schema=sample_schema,
    )

    # Write using context manager
    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, sample_schema, properties=property) as writer:
        assert not writer.is_closed
        writer.write(data)
        column_groups = writer.close()

    assert writer.is_closed

    # Read using context manager
    with Reader(column_groups, sample_schema, properties=property) as reader:
        assert not reader.is_closed
        batch_reader = reader.scan()
        result = list(batch_reader)
        assert len(result) > 0

    assert reader.is_closed


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_writer_invalid_schema():
    """Test creating writer with invalid schema."""
    with pytest.raises(InvalidArgumentError):
        Writer("/tmp/test", "not a schema")


def test_reader_invalid_schema(temp_dir, sample_schema):
    """Test creating reader with invalid schema."""
    # Write some data first
    data = pa.record_batch([[1, 2, 3], [1.0, 2.0, 3.0], ["a", "b", "c"]], schema=sample_schema)
    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, sample_schema, properties=property) as writer:
        writer.write(data)
        column_groups = writer.close()

    with pytest.raises(InvalidArgumentError):
        Reader(column_groups, "not a schema")


def test_write_wrong_schema(temp_dir, sample_schema):
    """Test writing batch with wrong schema."""
    wrong_schema = pa.schema([pa.field("x", pa.int32())])
    wrong_batch = pa.record_batch([[1, 2, 3]], schema=wrong_schema)

    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, sample_schema, properties=property) as writer:
        with pytest.raises(InvalidArgumentError):
            writer.write(wrong_batch)


def test_operations_after_close(temp_dir, sample_schema):
    """Test that operations after close raise errors."""
    data = pa.record_batch([[1], [1.0], ["a"]], schema=sample_schema)

    # Test writer operations after close
    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    writer = Writer(temp_dir, sample_schema, properties=property)
    writer.close()

    with pytest.raises(ResourceError):
        writer.write(data)

    with pytest.raises(ResourceError):
        writer.close()

    # Test reader operations after close
    with Writer(temp_dir, sample_schema, properties=property) as w:
        w.write(data)
        column_groups = w.close()

    reader = Reader(column_groups, sample_schema, properties=property)
    reader.close()

    with pytest.raises(ResourceError):
        reader.scan()

    with pytest.raises(ResourceError):
        reader.take([0])


def test_take_empty_indices(temp_dir, sample_schema):
    """Test take with empty indices raises error."""
    data = pa.record_batch([[1, 2], [1.0, 2.0], ["a", "b"]], schema=sample_schema)

    property = {
        "fs.storage_type": "local",
        "fs.root_path": temp_dir,
    }

    with Writer(temp_dir, sample_schema, properties=property) as writer:
        writer.write(data)
        column_groups = writer.close()

    with Reader(column_groups, sample_schema, properties=property) as reader:
        with pytest.raises(InvalidArgumentError):
            reader.take([])
