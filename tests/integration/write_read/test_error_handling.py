"""
Error handling tests.

Verify correct error reporting for invalid operations.
"""

import pyarrow as pa
import pytest
from milvus_storage import Reader, Writer
from milvus_storage.exceptions import InvalidArgumentError, ResourceError


class TestErrorHandling:
    """Test error handling in write/read operations."""

    def test_write_after_close_raises(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Writing after close raises ResourceError."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(100, offset=0))
        writer.close()

        with pytest.raises(ResourceError):
            writer.write(batch_generator(100, offset=100))

    def test_flush_after_close_raises(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Flushing after close raises ResourceError."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(100, offset=0))
        writer.close()

        with pytest.raises(ResourceError):
            writer.flush()

    def test_close_after_close_raises(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
    ):
        """Double close raises ResourceError."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.close()

        with pytest.raises(ResourceError):
            writer.close()

    def test_schema_mismatch_raises(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
    ):
        """Writing batch with wrong schema raises InvalidArgumentError."""
        writer = Writer(temp_case_path, simple_schema, default_properties)

        wrong_schema = pa.schema([pa.field("x", pa.int32())])
        wrong_batch = pa.RecordBatch.from_pydict({"x": [1, 2, 3]}, schema=wrong_schema)

        with pytest.raises(InvalidArgumentError):
            writer.write(wrong_batch)

    def test_invalid_writer_argument(self, temp_case_path: str):
        """Writer with invalid schema type raises InvalidArgumentError."""
        with pytest.raises(InvalidArgumentError):
            Writer(temp_case_path, "not_a_schema", {})

    def test_invalid_batch_argument(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
    ):
        """Writer.write with non-RecordBatch raises InvalidArgumentError."""
        writer = Writer(temp_case_path, simple_schema, default_properties)

        with pytest.raises(InvalidArgumentError):
            writer.write("not_a_batch")

    def test_scan_after_reader_close(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Scanning closed reader raises ResourceError."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(100, offset=0))
        column_groups = writer.close()

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        reader.close()

        with pytest.raises(ResourceError):
            reader.scan()

    def test_take_after_reader_close(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Take on closed reader raises ResourceError."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(100, offset=0))
        column_groups = writer.close()

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        reader.close()

        with pytest.raises(ResourceError):
            reader.take([0, 1])

    def test_take_empty_indices_raises(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Take with empty indices raises InvalidArgumentError."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(100, offset=0))
        column_groups = writer.close()

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        with pytest.raises(InvalidArgumentError):
            reader.take([])

    def test_take_unsorted_indices_raises(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Take with unsorted indices raises InvalidArgumentError."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(100, offset=0))
        column_groups = writer.close()

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        with pytest.raises(InvalidArgumentError):
            reader.take([50, 10, 100])  # Not sorted

    def test_chunk_reader_negative_index_raises(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Chunk reader with negative index raises InvalidArgumentError."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(100, offset=0))
        column_groups = writer.close()

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        chunk_reader = reader.get_chunk_reader(0)

        with pytest.raises(InvalidArgumentError):
            chunk_reader.get_chunk(-1)

        chunk_reader.close()

    def test_chunk_reader_closed_raises(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Operations on closed chunk reader raise ResourceError."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(100, offset=0))
        column_groups = writer.close()

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        chunk_reader = reader.get_chunk_reader(0)
        chunk_reader.close()

        with pytest.raises(ResourceError):
            chunk_reader.get_chunk(0)

        with pytest.raises(ResourceError):
            chunk_reader.get_number_of_chunks()
