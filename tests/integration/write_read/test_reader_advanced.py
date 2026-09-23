"""
Reader advanced functionality tests.

Verify column projection, take by indices, and scan patterns.
"""

import numpy as np
import pyarrow as pa
import pytest
from milvus_storage import Properties, Reader, Writer


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

    @pytest.mark.e2e_smoke
    def test_per_call_projection_restores_reader_default(
        self, temp_case_path, simple_schema, batch_generator, default_properties
    ):
        column_groups = self._write_data(
            temp_case_path, simple_schema, batch_generator, default_properties,
            num_batches=1, rows_per_batch=10,
        )
        with Reader(
            column_groups, simple_schema, columns=["id", "name"],
            properties=default_properties,
        ) as reader:
            projected = pa.Table.from_batches(reader.take([0, 3], columns=["value"]))
            assert projected.schema.names == ["value"]
            assert projected.column("value").to_pylist() == pytest.approx([0.0, 0.3])

            with reader.get_chunk_reader(0, columns=["name"]) as chunk_reader:
                chunk = chunk_reader.get_chunk(0)
                assert chunk.schema.names == ["name"]
                assert chunk.column("name").to_pylist() == [f"name_{i}" for i in range(10)]

            default = pa.Table.from_batches(reader.take([0, 3]))
            assert default.schema.names == ["id", "name"]
            assert default.column("id").to_pylist() == [0, 3]
            assert default.column("name").to_pylist() == ["name_0", "name_3"]

    @pytest.mark.parametrize(
        "indices",
        [
            pytest.param([0, 2, 4, 6, 8], id="list"),
            pytest.param(np.array([0, 2, 4, 6, 8], dtype=np.int64), id="contiguous"),
            pytest.param(
                np.arange(10, dtype=np.int64)[::2],
                id="strided",
            ),
        ],
    )
    @pytest.mark.e2e_smoke
    def test_take_preserves_numpy_index_values(
        self, temp_case_path, simple_schema, batch_generator, default_properties, indices
    ):
        column_groups = self._write_data(
            temp_case_path, simple_schema, batch_generator, default_properties,
            num_batches=1, rows_per_batch=10,
        )
        with Reader(column_groups, simple_schema, properties=default_properties) as reader:
            result = pa.Table.from_batches(reader.take(indices))
        expected = [0, 2, 4, 6, 8]
        assert result.column("id").to_pylist() == expected
        assert result.column("name").to_pylist() == [f"name_{i}" for i in expected]
        assert result.column("value").to_pylist() == pytest.approx([i * 0.1 for i in expected])

    @pytest.mark.e2e_smoke
    def test_scan_respects_batch_row_limit(
        self, temp_case_path, simple_schema, batch_generator, default_properties
    ):
        groups = self._write_data(
            temp_case_path, simple_schema, batch_generator, default_properties,
            num_batches=1, rows_per_batch=31,
        )
        properties = dict(default_properties)
        properties["reader.record_batch_max_rows"] = "7"
        with groups, Reader(groups, simple_schema, properties=properties) as reader:
            batches = list(reader.scan())
        assert len(batches) >= 5
        assert all(0 < batch.num_rows <= 7 for batch in batches)
        ids = [row for batch in batches for row in batch.column("id").to_pylist()]
        assert ids == list(range(31))

    @pytest.mark.parametrize("crt_enabled", ["true", "false"])
    def test_remote_reads_with_crt_setting(
        self, temp_case_path, simple_schema, batch_generator,
        default_properties, test_config, crt_enabled
    ):
        if not test_config.is_s3_compatible:
            pytest.skip("CRT read settings require an S3-compatible backend")
        groups = self._write_data(
            temp_case_path, simple_schema, batch_generator, default_properties,
            num_batches=2, rows_per_batch=10,
        )
        read_properties = dict(default_properties)
        read_properties["fs.s3.crt_async_read"] = crt_enabled
        with groups, Reader(groups, simple_schema, properties=read_properties) as reader:
            scanned = pa.Table.from_batches(list(reader.scan())).to_pydict()
            assert scanned == {
                "id": list(range(20)),
                "name": [f"name_{i}" for i in range(20)],
                "value": [i * 0.1 for i in range(20)],
            }

            selected = [0, 9, 10, 19]
            taken = pa.Table.from_batches(reader.take(selected)).to_pydict()
            assert taken["id"] == selected
            assert taken["name"] == [f"name_{i}" for i in selected]
            with reader.get_chunk_reader(0) as chunk_reader:
                chunk_ids = [
                    value for index in range(chunk_reader.get_number_of_chunks())
                    for value in chunk_reader.get_chunk(index).column("id").to_pylist()
                ]
            assert chunk_ids == list(range(20))

    @pytest.mark.parametrize("parallelism", [1, 2, 4])
    def test_take_parallelism_preserves_selected_rows(
        self, temp_case_path, simple_schema, batch_generator,
        default_properties, parallelism
    ):
        groups = self._write_data(
            temp_case_path, simple_schema, batch_generator, default_properties,
            num_batches=2, rows_per_batch=15,
        )
        with groups, Reader(groups, simple_schema, properties=default_properties) as reader:
            selected = [0, 14, 15, 29]
            taken = pa.Table.from_batches(reader.take(selected, parallelism=parallelism))
        assert taken.column("id").to_pylist() == selected
        assert taken.column("name").to_pylist() == [f"name_{i}" for i in selected]
        assert taken.column("value").to_pylist() == [i * 0.1 for i in selected]

    @pytest.mark.parametrize(
        "cache_enabled,hole_size,range_size",
        [("true", "0", "0"), ("false", "64", "256")],
    )
    def test_parquet_cache_and_prebuffer_options_preserve_data(
        self, temp_case_path, simple_schema, batch_generator,
        default_properties, cache_enabled, hole_size, range_size
    ):
        groups = self._write_data(
            temp_case_path, simple_schema, batch_generator, default_properties,
            num_batches=1, rows_per_batch=100,
        )
        properties = dict(default_properties)
        properties.update(
            {
                "reader.metadata_cache.enable": cache_enabled,
                "reader.parquet.prebuffer.hole_size_limit": hole_size,
                "reader.parquet.prebuffer.range_size_limit": range_size,
            }
        )
        reader_properties = Properties(properties)
        assert reader_properties.get("reader.metadata_cache.enable") == cache_enabled
        assert reader_properties.get("no.such.option", default="fallback") == "fallback"

        with groups, Reader(groups, simple_schema, properties=properties) as reader:
            scanned = pa.Table.from_batches(list(reader.scan())).to_pydict()
            assert scanned == batch_generator(100).to_pydict()
            selected = [0, 50, 99]
            taken = pa.Table.from_batches(reader.take(selected)).to_pydict()
            assert taken["id"] == selected
            assert taken["name"] == [f"name_{i}" for i in selected]

    def test_small_byte_budget_preserves_scan_values(
        self, temp_case_path, default_properties
    ):
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("payload", pa.string())])
        payloads = [f"{i:03d}-" + "x" * 64 for i in range(100)]
        batch = pa.RecordBatch.from_pydict(
            {"id": list(range(100)), "payload": payloads}, schema=schema
        )
        with Writer(temp_case_path, schema, default_properties) as writer:
            writer.write(batch)
            groups = writer.close()
        properties = dict(default_properties)
        properties["reader.record_batch_max_size"] = "100"
        properties["reader.record_batch_max_rows"] = "100"
        with groups, Reader(groups, schema, properties=properties) as reader:
            batches = list(reader.scan())
        scanned = pa.Table.from_batches(batches).to_pydict()
        assert scanned == {"id": list(range(100)), "payload": payloads}

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
