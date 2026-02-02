"""
Schema evolution tests.

Verify reading data after schema changes (new columns added).

Schema evolution uses Transaction.add_column_group() to register new columns,
while Transaction.append_files() is for appending rows with the same schema.
"""

import pyarrow as pa
from milvus_storage import Reader, Transaction, Writer
from milvus_storage.manifest import ColumnGroups


class TestSchemaEvolution:
    """Test schema evolution (adding columns via separate column groups)."""

    def test_read_with_original_schema(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Data written with schema v1 can be read with schema v1."""
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(1000, offset=0))
        column_groups = writer.close()

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        total_rows = sum(b.num_rows for b in reader.scan())
        assert total_rows == 1000

    def test_add_column_group_via_transaction(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Add new column group through transaction and verify manifest."""
        # Initial write + commit
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(500, offset=0))
        initial_cg = writer.close()

        txn_init = Transaction(temp_case_path, default_properties)
        txn_init.append_files(initial_cg)
        txn_init.commit()
        txn_init.close()

        # Add new column group with different columns via add_column_group
        new_schema = pa.schema([pa.field("score", pa.float32())])
        new_writer = Writer(temp_case_path, new_schema, default_properties)
        batch = pa.RecordBatch.from_pydict(
            {
                "score": pa.array([float(i) for i in range(500)], type=pa.float32()),
            },
            schema=new_schema,
        )
        new_writer.write(batch)
        new_cg = new_writer.close()

        # Convert ColumnGroups (C ptr) to Python ColumnGroup list
        cg_list = new_cg.to_list()

        txn = Transaction(temp_case_path, default_properties)
        for cg in cg_list:
            txn.add_column_group(cg)
        txn.commit()
        txn.close()

        # Verify manifest now has column groups from both schemas
        txn2 = Transaction(temp_case_path, default_properties)
        manifest = txn2.get_manifest()
        txn2.close()

        assert len(manifest.column_groups) >= 2
        all_columns = set()
        for cg in manifest.column_groups:
            all_columns.update(cg.columns)
        assert "score" in all_columns

        # Read with old schema — should still work, only original columns
        old_cg = ColumnGroups.from_list(manifest.column_groups)
        old_reader = Reader(old_cg, simple_schema, properties=default_properties)
        old_batches = list(old_reader.scan())
        assert sum(b.num_rows for b in old_batches) == 500
        for b in old_batches:
            assert set(b.schema.names) == set(simple_schema.names)

        # Read with merged schema (original + score) — all columns present
        merged_schema = pa.schema(
            list(simple_schema) + [pa.field("score", pa.float32())]
        )
        merged_cg = ColumnGroups.from_list(manifest.column_groups)
        merged_reader = Reader(merged_cg, merged_schema, properties=default_properties)
        merged_batches = list(merged_reader.scan())
        assert sum(b.num_rows for b in merged_batches) == 500
        for b in merged_batches:
            assert set(b.schema.names) == set(merged_schema.names)

    def test_incremental_schema_evolution(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """Incrementally add columns through multiple transactions."""
        # v1: id column — initial commit
        schema_v1 = pa.schema([pa.field("id", pa.int64())])
        writer1 = Writer(temp_case_path, schema_v1, default_properties)
        batch1 = pa.RecordBatch.from_pydict({"id": list(range(100))}, schema=schema_v1)
        writer1.write(batch1)
        cg1 = writer1.close()

        txn1 = Transaction(temp_case_path, default_properties)
        txn1.append_files(cg1)
        txn1.commit()
        txn1.close()

        # v2: add name column via add_column_group
        schema_v2 = pa.schema([pa.field("name", pa.string())])
        writer2 = Writer(temp_case_path, schema_v2, default_properties)
        batch2 = pa.RecordBatch.from_pydict(
            {"name": [f"n_{i}" for i in range(100)]}, schema=schema_v2
        )
        writer2.write(batch2)
        cg2 = writer2.close()

        txn2 = Transaction(temp_case_path, default_properties)
        for cg in cg2.to_list():
            txn2.add_column_group(cg)
        txn2.commit()
        txn2.close()

        # v3: add score column via add_column_group
        schema_v3 = pa.schema([pa.field("score", pa.float64())])
        writer3 = Writer(temp_case_path, schema_v3, default_properties)
        batch3 = pa.RecordBatch.from_pydict(
            {"score": [float(i) for i in range(100)]}, schema=schema_v3
        )
        writer3.write(batch3)
        cg3 = writer3.close()

        txn3 = Transaction(temp_case_path, default_properties)
        for cg in cg3.to_list():
            txn3.add_column_group(cg)
        txn3.commit()
        txn3.close()

        # Verify manifest has all three column groups
        txn_check = Transaction(temp_case_path, default_properties)
        manifest = txn_check.get_manifest()
        txn_check.close()

        all_columns = set()
        for cg in manifest.column_groups:
            all_columns.update(cg.columns)
        assert all_columns == {"id", "name", "score"}

        # Read with old schema (v1 only) — should still work
        old_cg = ColumnGroups.from_list(manifest.column_groups)
        old_reader = Reader(old_cg, schema_v1, properties=default_properties)
        old_batches = list(old_reader.scan())
        assert sum(b.num_rows for b in old_batches) == 100
        for b in old_batches:
            assert set(b.schema.names) == {"id"}

        # Read with full merged schema — all 3 columns present
        full_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("score", pa.float64()),
            ]
        )
        full_cg = ColumnGroups.from_list(manifest.column_groups)
        full_reader = Reader(full_cg, full_schema, properties=default_properties)
        full_batches = list(full_reader.scan())
        assert sum(b.num_rows for b in full_batches) == 100
        for b in full_batches:
            assert set(b.schema.names) == {"id", "name", "score"}

    def test_projection_subset_across_column_groups(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """Projection reads subset of columns spanning multiple column groups."""
        # Setup: 3 column groups — id, name, score
        schema_id = pa.schema([pa.field("id", pa.int64())])
        writer1 = Writer(temp_case_path, schema_id, default_properties)
        writer1.write(
            pa.RecordBatch.from_pydict({"id": list(range(100))}, schema=schema_id)
        )
        cg1 = writer1.close()

        txn1 = Transaction(temp_case_path, default_properties)
        txn1.append_files(cg1)
        txn1.commit()
        txn1.close()

        schema_name = pa.schema([pa.field("name", pa.string())])
        writer2 = Writer(temp_case_path, schema_name, default_properties)
        writer2.write(
            pa.RecordBatch.from_pydict(
                {"name": [f"n_{i}" for i in range(100)]}, schema=schema_name
            )
        )
        cg2 = writer2.close()

        txn2 = Transaction(temp_case_path, default_properties)
        for cg in cg2.to_list():
            txn2.add_column_group(cg)
        txn2.commit()
        txn2.close()

        schema_score = pa.schema([pa.field("score", pa.float64())])
        writer3 = Writer(temp_case_path, schema_score, default_properties)
        writer3.write(
            pa.RecordBatch.from_pydict(
                {"score": [float(i) for i in range(100)]}, schema=schema_score
            )
        )
        cg3 = writer3.close()

        txn3 = Transaction(temp_case_path, default_properties)
        for cg in cg3.to_list():
            txn3.add_column_group(cg)
        txn3.commit()
        txn3.close()

        txn_check = Transaction(temp_case_path, default_properties)
        manifest = txn_check.get_manifest()
        txn_check.close()

        # Projection: only id + score, skip name column group
        proj_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("score", pa.float64()),
            ]
        )
        proj_cg = ColumnGroups.from_list(manifest.column_groups)
        proj_reader = Reader(
            proj_cg, proj_schema, columns=["id", "score"], properties=default_properties
        )
        proj_batches = list(proj_reader.scan())
        assert sum(b.num_rows for b in proj_batches) == 100
        for b in proj_batches:
            assert set(b.schema.names) == {"id", "score"}

    def test_projection_nonexistent_column_filled_with_null(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """Projection with a column not in storage gets NULL-filled."""
        schema = pa.schema([pa.field("id", pa.int64())])
        writer = Writer(temp_case_path, schema, default_properties)
        writer.write(pa.RecordBatch.from_pydict({"id": list(range(50))}, schema=schema))
        cg = writer.close()

        txn = Transaction(temp_case_path, default_properties)
        txn.append_files(cg)
        txn.commit()
        txn.close()

        txn2 = Transaction(temp_case_path, default_properties)
        manifest = txn2.get_manifest()
        txn2.close()

        # Schema includes "phantom" column not in any column group
        read_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("phantom", pa.string()),
            ]
        )
        read_cg = ColumnGroups.from_list(manifest.column_groups)
        reader = Reader(
            read_cg,
            read_schema,
            columns=["id", "phantom"],
            properties=default_properties,
        )
        batches = list(reader.scan())
        assert sum(b.num_rows for b in batches) == 50
        for b in batches:
            assert set(b.schema.names) == {"id", "phantom"}
            # phantom column should be all null
            assert b.column("phantom").null_count == b.num_rows
