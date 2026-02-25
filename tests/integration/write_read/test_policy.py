"""
Writer policy tests.

Verify writer column group policies: single, schema_based, size_based.
"""

import pyarrow as pa
import pytest
from milvus_storage import PropertyKeys, Reader, Transaction, Writer
from milvus_storage.manifest import ColumnGroups


class TestWriterPolicy:
    """Test writer column group policies."""

    # ===== Single Policy Tests =====

    def test_single_policy(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
    ):
        """Single policy puts all columns in one group (default behavior)."""
        # Test both default and explicit single policy
        for policy in [None, "single"]:
            path = f"{temp_case_path}/{'default' if policy is None else 'explicit'}"
            props = test_config.get_properties()
            if policy:
                props[PropertyKeys.WRITER_POLICY] = policy

            writer = Writer(path, simple_schema, props)
            writer.write(batch_generator(1000))
            column_groups = writer.close()

            # Should have exactly 1 column group
            cg_list = column_groups.to_list()
            assert len(cg_list) == 1

            # The single group should contain all columns
            assert set(cg_list[0].columns) == set(simple_schema.names)

            # Verify data integrity
            reader = Reader(column_groups, simple_schema, properties=props)
            batches = list(reader.scan())
            total_rows = sum(b.num_rows for b in batches)
            assert total_rows == 1000

    # ===== Schema Based Policy Tests =====

    def test_schema_based_simple_pattern(
        self,
        temp_case_path: str,
        test_config,
    ):
        """Split columns by simple regex pattern."""
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("score", pa.float32()),
            ]
        )

        props = test_config.get_properties()
        props[PropertyKeys.WRITER_POLICY] = "schema_based"
        # Pattern: group1=[id, name], group2=[value, score]
        props[PropertyKeys.WRITER_SCHEMA_BASE_PATTERNS] = "id|name,value|score"

        writer = Writer(temp_case_path, schema, props)
        batch = pa.RecordBatch.from_pydict(
            {
                "id": list(range(500)),
                "name": [f"name_{i}" for i in range(500)],
                "value": [float(i) for i in range(500)],
                "score": pa.array(
                    [float(i) * 0.1 for i in range(500)], type=pa.float32()
                ),
            },
            schema=schema,
        )
        writer.write(batch)
        column_groups = writer.close()

        # Should have exactly 2 column groups
        cg_list = column_groups.to_list()
        assert len(cg_list) == 2

        # Verify column distribution
        all_columns = []
        for cg in cg_list:
            all_columns.extend(cg.columns)
        assert set(all_columns) == {"id", "name", "value", "score"}

        # One group should have [id, name], another [value, score]
        columns_by_group = [set(cg.columns) for cg in cg_list]
        assert {"id", "name"} in columns_by_group
        assert {"value", "score"} in columns_by_group

        # Verify data integrity
        reader = Reader(column_groups, schema, properties=props)
        batches = list(reader.scan())
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 500

    def test_schema_based_partial_match(
        self,
        temp_case_path: str,
        test_config,
    ):
        """Unmatched columns go to default group."""
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("extra1", pa.int32()),
                pa.field("extra2", pa.int32()),
            ]
        )

        props = test_config.get_properties()
        props[PropertyKeys.WRITER_POLICY] = "schema_based"
        # Only match id and name, others go to default group
        props[PropertyKeys.WRITER_SCHEMA_BASE_PATTERNS] = "id|name"

        writer = Writer(temp_case_path, schema, props)
        batch = pa.RecordBatch.from_pydict(
            {
                "id": list(range(500)),
                "name": [f"name_{i}" for i in range(500)],
                "value": [float(i) for i in range(500)],
                "extra1": list(range(500)),
                "extra2": list(range(500, 1000)),
            },
            schema=schema,
        )
        writer.write(batch)
        column_groups = writer.close()

        # Should have 2 groups: matched + default
        cg_list = column_groups.to_list()
        assert len(cg_list) == 2

        # Verify column distribution
        columns_by_group = [set(cg.columns) for cg in cg_list]
        assert {"id", "name"} in columns_by_group
        assert {"value", "extra1", "extra2"} in columns_by_group

        # Verify data integrity
        reader = Reader(column_groups, schema, properties=props)
        batches = list(reader.scan())
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 500

    def test_schema_based_no_match_all_default(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
    ):
        """No patterns match, all columns go to default group."""
        props = test_config.get_properties()
        props[PropertyKeys.WRITER_POLICY] = "schema_based"
        # Pattern that matches nothing
        props[PropertyKeys.WRITER_SCHEMA_BASE_PATTERNS] = "nonexistent_column"

        writer = Writer(temp_case_path, simple_schema, props)
        writer.write(batch_generator(500))
        column_groups = writer.close()

        # All columns in default group
        cg_list = column_groups.to_list()
        assert len(cg_list) == 1
        assert set(cg_list[0].columns) == set(simple_schema.names)

        # Verify data integrity
        reader = Reader(column_groups, simple_schema, properties=props)
        batches = list(reader.scan())
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 500

    def test_schema_based_different_patterns_append(
        self,
        temp_case_path: str,
        test_config,
    ):
        """Write with pattern A, append with pattern B, then read all."""
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("score", pa.float32()),
            ]
        )

        def make_batch(offset, count):
            return pa.RecordBatch.from_pydict(
                {
                    "id": list(range(offset, offset + count)),
                    "name": [f"name_{i}" for i in range(offset, offset + count)],
                    "value": [float(i) for i in range(offset, offset + count)],
                    "score": pa.array(
                        [float(i) * 0.1 for i in range(count)], type=pa.float32()
                    ),
                },
                schema=schema,
            )

        # Step 1: Write with pattern A: [id, name] + [value, score]
        props_a = test_config.get_properties()
        props_a[PropertyKeys.WRITER_POLICY] = "schema_based"
        props_a[PropertyKeys.WRITER_SCHEMA_BASE_PATTERNS] = "id|name,value|score"

        writer_a = Writer(temp_case_path, schema, props_a)
        writer_a.write(make_batch(0, 500))
        cg_a = writer_a.close()

        # Commit first write
        txn1 = Transaction(temp_case_path, props_a)
        txn1.append_files(cg_a)
        txn1.commit()
        txn1.close()

        # Step 2: Write with pattern B: [id, value] + [name, score]
        props_b = test_config.get_properties()
        props_b[PropertyKeys.WRITER_POLICY] = "schema_based"
        props_b[PropertyKeys.WRITER_SCHEMA_BASE_PATTERNS] = "id|value,name|score"

        writer_b = Writer(temp_case_path, schema, props_b)
        writer_b.write(make_batch(500, 500))
        cg_b = writer_b.close()

        # Verify pattern B produced different grouping
        cg_b_list = cg_b.to_list()
        assert len(cg_b_list) == 2
        columns_b = [set(cg.columns) for cg in cg_b_list]
        assert {"id", "value"} in columns_b
        assert {"name", "score"} in columns_b

        # Step 3: Append second write (different column group structure)
        # This should fail because column group count/structure doesn't match
        txn2 = Transaction(temp_case_path, props_b)
        txn2.append_files(cg_b)
        with pytest.raises(Exception):
            txn2.commit()
        txn2.close()

        # Verify original data is intact
        txn3 = Transaction(temp_case_path, props_a)
        manifest = txn3.get_manifest()
        txn3.close()

        cg_read = ColumnGroups.from_list(manifest.column_groups)
        reader = Reader(cg_read, schema, properties=props_a)
        batches = list(reader.scan())
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 500  # Only first write succeeded

    def test_schema_based_same_pattern_append(
        self,
        temp_case_path: str,
        test_config,
    ):
        """Write with same pattern twice, then read all."""
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("score", pa.float32()),
            ]
        )

        def make_batch(offset, count):
            return pa.RecordBatch.from_pydict(
                {
                    "id": list(range(offset, offset + count)),
                    "name": [f"name_{i}" for i in range(offset, offset + count)],
                    "value": [float(i) for i in range(offset, offset + count)],
                    "score": pa.array(
                        [float(i) * 0.1 for i in range(count)], type=pa.float32()
                    ),
                },
                schema=schema,
            )

        props = test_config.get_properties()
        props[PropertyKeys.WRITER_POLICY] = "schema_based"
        props[PropertyKeys.WRITER_SCHEMA_BASE_PATTERNS] = "id|name,value|score"

        # Step 1: First write
        writer1 = Writer(temp_case_path, schema, props)
        writer1.write(make_batch(0, 500))
        cg1 = writer1.close()

        txn1 = Transaction(temp_case_path, props)
        txn1.append_files(cg1)
        txn1.commit()
        txn1.close()

        # Step 2: Second write with same pattern
        writer2 = Writer(temp_case_path, schema, props)
        writer2.write(make_batch(500, 500))
        cg2 = writer2.close()

        txn2 = Transaction(temp_case_path, props)
        txn2.append_files(cg2)
        txn2.commit()
        txn2.close()

        # Step 3: Read all data
        txn3 = Transaction(temp_case_path, props)
        manifest = txn3.get_manifest()
        txn3.close()

        cg_read = ColumnGroups.from_list(manifest.column_groups)
        reader = Reader(cg_read, schema, properties=props)
        batches = list(reader.scan())
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 1000

        # Verify data content
        all_ids = []
        for b in batches:
            all_ids.extend(b.column("id").to_pylist())
        assert sorted(all_ids) == list(range(1000))

    # ===== Size Based Policy Tests =====

    def test_size_based_large_column_separate(
        self,
        temp_case_path: str,
        test_config,
    ):
        """Large columns are put in separate groups."""
        # Create schema with one large column (binary vector) and small columns
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field(
                    "vector", pa.list_(pa.float32())
                ),  # variable-length list, ~512 bytes/row with 128 elements
            ]
        )

        props = test_config.get_properties()
        props[PropertyKeys.WRITER_POLICY] = "size_based"
        # Set threshold low enough that vector column exceeds it
        props[PropertyKeys.WRITER_SIZE_BASE_MACS] = "100"  # 100 bytes avg
        props[PropertyKeys.WRITER_SIZE_BASE_MCIG] = "10"  # max 10 columns per group

        num_rows = 500
        batch = pa.RecordBatch.from_pydict(
            {
                "id": list(range(num_rows)),
                "name": [f"name_{i}" for i in range(num_rows)],
                "vector": [[float(j) for j in range(128)] for _ in range(num_rows)],
            },
            schema=schema,
        )

        writer = Writer(temp_case_path, schema, props)
        writer.write(batch)
        column_groups = writer.close()

        # Should have multiple groups (vector likely separate)
        cg_list = column_groups.to_list()
        assert len(cg_list) >= 2

        # Verify all columns are present
        all_columns = []
        for cg in cg_list:
            all_columns.extend(cg.columns)
        assert set(all_columns) == {"id", "name", "vector"}

        # Verify data integrity
        reader = Reader(column_groups, schema, properties=props)
        batches = list(reader.scan())
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == num_rows

    def test_size_based_mixed_sizes(
        self,
        temp_case_path: str,
        test_config,
    ):
        """Mixed column sizes with both thresholds."""
        # Create schema with varying column sizes
        schema = pa.schema(
            [
                pa.field("tiny1", pa.int8()),
                pa.field("tiny2", pa.int8()),
                pa.field("small", pa.int32()),
                pa.field("medium", pa.string()),
                pa.field(
                    "large", pa.list_(pa.float64())
                ),  # variable-length list, ~512 bytes/row with 64 elements
            ]
        )

        props = test_config.get_properties()
        props[PropertyKeys.WRITER_POLICY] = "size_based"
        props[PropertyKeys.WRITER_SIZE_BASE_MACS] = "50"  # 50 bytes threshold
        props[PropertyKeys.WRITER_SIZE_BASE_MCIG] = "3"  # max 3 columns per group

        num_rows = 300
        batch = pa.RecordBatch.from_pydict(
            {
                "tiny1": [i % 128 for i in range(num_rows)],
                "tiny2": [i % 128 for i in range(num_rows)],
                "small": list(range(num_rows)),
                "medium": [f"medium_value_{i}" for i in range(num_rows)],
                "large": [[float(j) for j in range(64)] for _ in range(num_rows)],
            },
            schema=schema,
        )

        writer = Writer(temp_case_path, schema, props)
        writer.write(batch)
        column_groups = writer.close()

        cg_list = column_groups.to_list()

        # With these settings, expect multiple groups
        # Large column should be separate due to size threshold
        assert len(cg_list) >= 2

        # Verify all columns present
        all_columns = []
        for cg in cg_list:
            all_columns.extend(cg.columns)
        assert set(all_columns) == {"tiny1", "tiny2", "small", "medium", "large"}

        # Verify data integrity
        reader = Reader(column_groups, schema, properties=props)
        batches = list(reader.scan())
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == num_rows

    def test_size_based_different_config_append(
        self,
        temp_case_path: str,
        test_config,
    ):
        """Write with config A, append with config B (different grouping), commit should fail."""
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("value", pa.float64()),
                pa.field(
                    "vector", pa.list_(pa.float32())
                ),  # ~512 bytes/row with 128 elements
            ]
        )

        def make_batch(offset, count):
            return pa.RecordBatch.from_pydict(
                {
                    "id": list(range(offset, offset + count)),
                    "name": [f"name_{i}" for i in range(offset, offset + count)],
                    "value": [float(i) for i in range(offset, offset + count)],
                    "vector": [[float(j) for j in range(128)] for _ in range(count)],
                },
                schema=schema,
            )

        # Config A: low threshold -> vector separate, others together
        props_a = test_config.get_properties()
        props_a[PropertyKeys.WRITER_POLICY] = "size_based"
        props_a[PropertyKeys.WRITER_SIZE_BASE_MACS] = "100"  # 100 bytes threshold
        props_a[PropertyKeys.WRITER_SIZE_BASE_MCIG] = "10"

        writer_a = Writer(temp_case_path, schema, props_a)
        writer_a.write(make_batch(0, 500))
        cg_a = writer_a.close()

        cg_a_list = cg_a.to_list()
        # Should have multiple groups (vector likely separate)
        assert len(cg_a_list) >= 2

        # Commit first write
        txn1 = Transaction(temp_case_path, props_a)
        txn1.append_files(cg_a)
        txn1.commit()
        txn1.close()

        # Config B: high threshold -> all columns in one group
        props_b = test_config.get_properties()
        props_b[PropertyKeys.WRITER_POLICY] = "size_based"
        props_b[PropertyKeys.WRITER_SIZE_BASE_MACS] = "999999999"  # very high threshold
        props_b[PropertyKeys.WRITER_SIZE_BASE_MCIG] = "100"

        writer_b = Writer(temp_case_path, schema, props_b)
        writer_b.write(make_batch(500, 500))
        cg_b = writer_b.close()

        cg_b_list = cg_b.to_list()
        # Should have only 1 group (all columns together)
        assert len(cg_b_list) == 1

        # Append with different column group structure should fail
        txn2 = Transaction(temp_case_path, props_b)
        txn2.append_files(cg_b)
        with pytest.raises(Exception):
            txn2.commit()
        txn2.close()

        # Verify original data intact
        txn3 = Transaction(temp_case_path, props_a)
        manifest = txn3.get_manifest()
        txn3.close()

        cg_read = ColumnGroups.from_list(manifest.column_groups)
        reader = Reader(cg_read, schema, properties=props_a)
        batches = list(reader.scan())
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 500  # Only first write succeeded
