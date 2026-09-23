"""
Boundary condition tests.

Verify edge cases and boundary behavior for write/read operations.
"""

import pyarrow as pa
import pytest
from milvus_storage import Reader, Transaction, Writer
from milvus_storage.manifest import ColumnGroups


class TestBoundaryConditions:
    """Test boundary conditions in write/read operations."""

    def test_zero_row_batches_do_not_change_committed_data(
        self, temp_case_path, default_properties
    ):
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("name", pa.string())])
        batch = pa.RecordBatch.from_pydict(
            {"id": [0, 1, 2, 3, 4], "name": [None, "", "name_2", None, "name_4"]},
            schema=schema,
        )
        with Writer(temp_case_path, schema, default_properties) as writer:
            writer.write(batch.slice(0, 0))
            writer.write(batch.slice(0, 3))
            writer.write(batch.slice(3, 0))
            writer.flush()
            writer.write(batch.slice(3, 2))
            writer.write(batch.slice(5, 0))
            groups = writer.close()
        with Transaction(temp_case_path, default_properties) as txn:
            txn.append_files(groups)
            txn.commit()
        groups.destroy()
        with Transaction(temp_case_path, default_properties) as txn:
            manifest = txn.get_manifest()
        with ColumnGroups.from_list(manifest.column_groups) as groups:
            with Reader(groups, schema, properties=default_properties) as reader:
                actual = pa.Table.from_batches(list(reader.scan())).to_pydict()
        assert actual == batch.to_pydict()

    def test_single_row_write(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
    ):
        """Write and read a single row."""
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [42],
                "name": ["only_one"],
                "value": [3.14],
            },
            schema=simple_schema,
        )

        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch)
        column_groups = writer.close()

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        result = list(reader.scan())
        total_rows = sum(b.num_rows for b in result)
        assert total_rows == 1

        ids = result[0].column("id").to_pylist()
        assert ids == [42]

    def test_empty_string_values(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """Write and read empty string values."""
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("text", pa.string()),
            ]
        )
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [0, 1, 2],
                "text": ["", "hello", ""],
            },
            schema=schema,
        )

        writer = Writer(temp_case_path, schema, default_properties)
        writer.write(batch)
        column_groups = writer.close()

        reader = Reader(column_groups, schema, properties=default_properties)
        result = list(reader.scan())
        texts = []
        for b in result:
            texts.extend(b.column("text").to_pylist())
        assert texts == ["", "hello", ""]

    def test_null_values(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """Write and read null values in nullable columns."""
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("optional_val", pa.float64()),
            ]
        )
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [0, 1, 2, 3],
                "optional_val": [1.0, None, 3.0, None],
            },
            schema=schema,
        )

        writer = Writer(temp_case_path, schema, default_properties)
        writer.write(batch)
        column_groups = writer.close()

        reader = Reader(column_groups, schema, properties=default_properties)
        result = list(reader.scan())
        vals = []
        for b in result:
            vals.extend(b.column("optional_val").to_pylist())
        assert vals == [1.0, None, 3.0, None]

    def test_large_string_values(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """Write and read large string values (1KB each)."""
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("text", pa.string()),
            ]
        )
        large_text = "x" * 1024  # 1KB string
        batch = pa.RecordBatch.from_pydict(
            {
                "id": list(range(100)),
                "text": [large_text] * 100,
            },
            schema=schema,
        )

        writer = Writer(temp_case_path, schema, default_properties)
        writer.write(batch)
        column_groups = writer.close()

        reader = Reader(column_groups, schema, properties=default_properties)
        result = list(reader.scan())
        texts = []
        for b in result:
            texts.extend(b.column("text").to_pylist())
        assert all(t == large_text for t in texts)
        assert len(texts) == 100

    def test_multiple_data_types(
        self,
        temp_case_path: str,
        all_types_schema: pa.Schema,
        default_properties,
    ):
        """Write and read all supported data types."""
        batch = pa.RecordBatch.from_pydict(
            {
                "col_int8": pa.array([1, -1, 127], type=pa.int8()),
                "col_int16": pa.array([256, -256, 32767], type=pa.int16()),
                "col_int32": pa.array([100000, -100000, 0], type=pa.int32()),
                "col_int64": pa.array([1, 2, 3], type=pa.int64()),
                "col_uint8": pa.array([0, 128, 255], type=pa.uint8()),
                "col_uint16": pa.array([0, 1000, 65535], type=pa.uint16()),
                "col_uint32": pa.array([0, 100000, 4294967295], type=pa.uint32()),
                "col_uint64": pa.array([0, 1, 2], type=pa.uint64()),
                "col_float32": pa.array([1.5, -2.5, 0.0], type=pa.float32()),
                "col_float64": pa.array([1.5, -2.5, 0.0], type=pa.float64()),
                "col_bool": [True, False, True],
                "col_string": ["hello", "world", ""],
                "col_binary": [b"\x00\x01", b"\xff", b""],
                "col_list": [[1, 2], [3], [4, 5, 6]],
            },
            schema=all_types_schema,
        )

        writer = Writer(temp_case_path, all_types_schema, default_properties)
        writer.write(batch)
        column_groups = writer.close()

        reader = Reader(column_groups, all_types_schema, properties=default_properties)
        result = list(reader.scan())
        total_rows = sum(b.num_rows for b in result)
        assert total_rows == 3

    def test_many_batches_write(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Write many small batches."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        total_rows = 0
        for i in range(100):
            writer.write(batch_generator(10, offset=i * 10))
            total_rows += 10
        column_groups = writer.close()

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        read_rows = sum(b.num_rows for b in reader.scan())
        assert read_rows == total_rows

    def test_writer_flush_then_continue(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Flush mid-write then continue writing."""
        writer = Writer(temp_case_path, simple_schema, default_properties)

        writer.write(batch_generator(500, offset=0))
        writer.flush()
        writer.write(batch_generator(500, offset=500))
        column_groups = writer.close()

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        all_ids = []
        for batch in reader.scan():
            all_ids.extend(batch.column("id").to_pylist())
        assert all_ids == list(range(1000))

    def test_writer_close_without_write(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
    ):
        """Close writer without writing any data."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        column_groups = writer.close()

        # Should get empty column_groups
        cg_list = column_groups.to_list()
        total_files = sum(len(cg.files) for cg in cg_list)
        assert total_files == 0

    @pytest.mark.xfail(
        strict=True,
        reason="D14: Parquet fixed_size_list scan exports an invalid Arrow buffer layout",
        raises=pa.ArrowInvalid,
    )
    def test_vector_data_roundtrip(
        self,
        temp_case_path: str,
        vector_schema: pa.Schema,
        vector_batch: pa.RecordBatch,
        default_properties,
    ):
        """Write and read vector (fixed-size list) data."""
        properties = {**default_properties, "writer.format": "parquet"}
        writer = Writer(temp_case_path, vector_schema, properties)
        writer.write(vector_batch)
        column_groups = writer.close()

        reader = Reader(column_groups, vector_schema, properties=properties)
        result = list(reader.scan())
        total_rows = sum(b.num_rows for b in result)
        assert total_rows == vector_batch.num_rows

    def test_writer_context_manager(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Writer works as context manager."""
        with Writer(temp_case_path, simple_schema, default_properties) as writer:
            writer.write(batch_generator(100, offset=0))
            column_groups = writer.close()

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        read_rows = sum(b.num_rows for b in reader.scan())
        assert read_rows == 100

    def test_reader_context_manager(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Reader works as context manager."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(100, offset=0))
        column_groups = writer.close()

        with Reader(
            column_groups, simple_schema, properties=default_properties
        ) as reader:
            read_rows = sum(b.num_rows for b in reader.scan())
            assert read_rows == 100
