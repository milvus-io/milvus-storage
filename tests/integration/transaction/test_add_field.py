"""
Transaction add field (schema evolution) tests.

Verify adding new column groups with different columns via add_column_group.
"""

import pyarrow as pa
from milvus_storage import Transaction, Writer


class TestTransactionAddField:
    """Test adding fields via transactions."""

    def _commit_data(self, path, schema, batch_gen, props, rows=1000):
        """Write data and commit via transaction. Return ColumnGroups."""
        writer = Writer(path, schema, props)
        writer.write(batch_gen(rows, offset=0))
        cg = writer.close()

        txn = Transaction(path, props)
        txn.append_files(cg)
        txn.commit()
        txn.close()
        return cg

    def test_add_column_group_with_new_columns(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Add a new column group with extra columns."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        # Write a new column group with different columns
        new_schema = pa.schema([pa.field("score", pa.float32())])
        new_writer = Writer(temp_case_path, new_schema, default_properties)
        batch = pa.RecordBatch.from_pydict(
            {"score": pa.array([float(i) for i in range(1000)], type=pa.float32())},
            schema=new_schema,
        )
        new_writer.write(batch)
        new_cg = new_writer.close()

        txn = Transaction(temp_case_path, default_properties)
        for cg in new_cg.to_list():
            txn.add_column_group(cg)
        version = txn.commit()
        txn.close()

        assert version >= 0

    def test_add_column_group_metadata(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Adding a column group is reflected in manifest metadata."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        # Get initial manifest
        txn1 = Transaction(temp_case_path, default_properties)
        initial_manifest = txn1.get_manifest()
        initial_files = sum(len(cg.files) for cg in initial_manifest.column_groups)
        txn1.close()

        # Add new column group (must have same row count as existing: 1000)
        new_schema = pa.schema([pa.field("extra", pa.int64())])
        new_writer = Writer(temp_case_path, new_schema, default_properties)
        batch = pa.RecordBatch.from_pydict(
            {"extra": list(range(1000))},
            schema=new_schema,
        )
        new_writer.write(batch)
        new_cg = new_writer.close()

        txn2 = Transaction(temp_case_path, default_properties)
        for cg in new_cg.to_list():
            txn2.add_column_group(cg)
        txn2.commit()
        txn2.close()

        # Verify manifest updated
        txn3 = Transaction(temp_case_path, default_properties)
        new_manifest = txn3.get_manifest()
        txn3.close()

        new_files = sum(len(cg.files) for cg in new_manifest.column_groups)
        assert new_files > initial_files

    def test_multiple_column_types(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Add columns of different types through separate transactions."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=100,
        )

        type_schemas = [
            pa.schema([pa.field("col_int32", pa.int32())]),
            pa.schema([pa.field("col_float64", pa.float64())]),
            pa.schema([pa.field("col_string", pa.string())]),
            pa.schema([pa.field("col_bool", pa.bool_())]),
        ]

        type_data = [
            {"col_int32": pa.array(list(range(100)), type=pa.int32())},
            {"col_float64": [float(i) for i in range(100)]},
            {"col_string": [f"str_{i}" for i in range(100)]},
            {"col_bool": [i % 2 == 0 for i in range(100)]},
        ]

        for schema, data in zip(type_schemas, type_data):
            new_writer = Writer(temp_case_path, schema, default_properties)
            batch = pa.RecordBatch.from_pydict(data, schema=schema)
            new_writer.write(batch)
            new_cg = new_writer.close()

            txn = Transaction(temp_case_path, default_properties)
            for cg in new_cg.to_list():
                txn.add_column_group(cg)
            txn.commit()
            txn.close()

        # Verify all columns in manifest
        txn_check = Transaction(temp_case_path, default_properties)
        manifest = txn_check.get_manifest()
        txn_check.close()

        all_columns = set()
        for cg in manifest.column_groups:
            all_columns.update(cg.columns)

        for expected in ["col_int32", "col_float64", "col_string", "col_bool"]:
            assert expected in all_columns
