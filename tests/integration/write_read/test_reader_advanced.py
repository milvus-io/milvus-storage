"""
Reader advanced functionality tests.

Verify column projection, take by indices, and scan patterns.
"""

import pyarrow as pa
import pytest
from milvus_storage import Reader, Writer


class TestReaderAdvanced:
    """Test advanced reader functionality."""

    def _write_data(
        self, path, schema, batch_generator, props, num_batches=5, rows_per_batch=1000
    ):
        """Helper to write test data."""
        writer = Writer(path, schema, props)
        for i in range(num_batches):
            writer.write(batch_generator(rows_per_batch, offset=i * rows_per_batch))
        return writer.close()

    def test_column_projection(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Read only a subset of columns."""
        column_groups = self._write_data(
            temp_case_path, simple_schema, batch_generator, default_properties
        )

        # Read only 'id' and 'value' columns
        reader = Reader(
            column_groups,
            simple_schema,
            columns=["id", "value"],
            properties=default_properties,
        )
        batches = list(reader.scan())

        assert len(batches) > 0
        for batch in batches:
            assert batch.num_columns == 2
            assert "id" in batch.schema.names
            assert "value" in batch.schema.names
            assert "name" not in batch.schema.names

    def test_take_with_indices(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Take specific rows by indices."""
        column_groups = self._write_data(
            temp_case_path, simple_schema, batch_generator, default_properties
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        indices = [0, 10, 50, 100, 999]
        batches = reader.take(indices)

        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == len(indices)

        all_ids = []
        for batch in batches:
            all_ids.extend(batch.column("id").to_pylist())
        assert all_ids == indices

    def test_take_with_large_index_list(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Take with a large number of indices."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            num_batches=10,
            rows_per_batch=1000,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        # Take every 10th row
        indices = list(range(0, 10000, 10))
        batches = reader.take(indices)

        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == len(indices)

    def test_take_across_multiple_files(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
        default_properties,
    ):
        """Take data spanning multiple rolled files."""
        props = test_config.get_properties(
            file_rolling_size=50 * 1024,  # Small to force rolling
            buffer_size=20 * 1024,
        )

        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            props,
            num_batches=10,
            rows_per_batch=1000,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        # Pick indices that span across file boundaries
        indices = [0, 500, 1000, 2500, 5000, 7500, 9999]
        batches = reader.take(indices)

        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == len(indices)

        all_ids = []
        for batch in batches:
            all_ids.extend(batch.column("id").to_pylist())
        assert all_ids == indices

    def test_full_scan_data_integrity(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Full scan returns all data in order."""
        total_rows = 5000
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        all_ids = []
        for batch in reader.scan():
            all_ids.extend(batch.column("id").to_pylist())

        assert len(all_ids) == total_rows
        assert all_ids == list(range(total_rows))

    def test_multiple_scans_same_reader(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Multiple scans on the same reader produce consistent results."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)

        # First scan
        rows1 = sum(b.num_rows for b in reader.scan())

        # Second scan
        rows2 = sum(b.num_rows for b in reader.scan())

        assert rows1 == rows2 == 5000

    def test_single_column_projection(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Project a single column."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        reader = Reader(
            column_groups, simple_schema, columns=["id"], properties=default_properties
        )
        batches = list(reader.scan())

        assert len(batches) > 0
        for batch in batches:
            assert batch.num_columns == 1
            assert batch.schema.names == ["id"]

    def test_scan_missing_column_fills_null(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Scan with a column not in storage fills it with nulls."""
        # simple_schema has columns: id, name, value
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            num_batches=1,
            rows_per_batch=100,
        )

        # Build a schema that includes "phantom" so Reader accepts it
        extended_schema = pa.schema(
            list(simple_schema) + [pa.field("phantom", pa.int64())]
        )
        # Request "id" (exists in column group) and "phantom" (not in column group)
        reader = Reader(
            column_groups,
            extended_schema,
            columns=["id", "phantom"],
            properties=default_properties,
        )
        batches = list(reader.scan())

        assert len(batches) > 0
        batch = batches[0]
        # "id" has real data, "phantom" is filled with nulls
        assert "id" in batch.schema.names
        assert "phantom" in batch.schema.names
        assert batch.column("id").null_count == 0
        assert batch.column("phantom").null_count == batch.num_rows

        batches = reader.take([0, 1, 2])
        assert len(batches) > 0
        all_names = set()
        total_rows = 0
        for batch in batches:
            all_names.update(batch.schema.names)
            total_rows += batch.num_rows
        assert "id" in all_names
        assert "phantom" in all_names
        assert total_rows == 3

    def test_chunk_reader_no_intersection_raises(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """ChunkReader with no column intersection raises an error."""
        # simple_schema has columns: id, name, value
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            num_batches=1,
            rows_per_batch=100,
        )

        # Request only columns that don't exist in the column group
        reader = Reader(
            column_groups,
            simple_schema,
            columns=["phantom_a", "phantom_b"],
            properties=default_properties,
        )
        with pytest.raises(Exception):
            reader.get_chunk_reader(0)

    def test_no_projection_reads_all_columns(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """No columns specified reads all columns from schema."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            num_batches=1,
            rows_per_batch=100,
        )

        # No columns argument — read all
        reader = Reader(column_groups, simple_schema, properties=default_properties)
        batches = list(reader.scan())

        assert len(batches) > 0
        for batch in batches:
            assert set(batch.schema.names) == set(simple_schema.names)
        assert sum(b.num_rows for b in batches) == 100

    def test_get_chunk_with_projection(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """get_chunk returns only projected columns."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            num_batches=1,
            rows_per_batch=100,
        )

        reader = Reader(
            column_groups,
            simple_schema,
            columns=["id", "value"],
            properties=default_properties,
        )
        chunk_reader = reader.get_chunk_reader(0)
        num_chunks = chunk_reader.get_number_of_chunks()
        assert num_chunks > 0

        batch = chunk_reader.get_chunk(0)
        assert batch.num_columns == 2
        assert "id" in batch.schema.names
        assert "value" in batch.schema.names
        assert "name" not in batch.schema.names
        assert batch.num_rows > 0

    def test_get_chunks_with_projection(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """get_chunks returns only projected columns for multiple chunks."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            num_batches=3,
            rows_per_batch=1000,
        )

        reader = Reader(
            column_groups, simple_schema, columns=["id"], properties=default_properties
        )
        chunk_reader = reader.get_chunk_reader(0)
        num_chunks = chunk_reader.get_number_of_chunks()
        assert num_chunks > 0

        indices = list(range(num_chunks))
        batches = chunk_reader.get_chunks(indices)
        assert len(batches) == num_chunks
        total_rows = 0
        for batch in batches:
            assert batch.num_columns == 1
            assert batch.schema.names == ["id"]
            total_rows += batch.num_rows
        assert total_rows == 3000

    def test_get_chunk_reader_reads_all_without_projection(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """get_chunk_reader without projection returns all columns."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            num_batches=1,
            rows_per_batch=100,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        chunk_reader = reader.get_chunk_reader(0)
        batch = chunk_reader.get_chunk(0)
        assert set(batch.schema.names) == set(simple_schema.names)
        assert batch.num_rows > 0
