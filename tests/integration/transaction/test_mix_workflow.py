"""
Transaction mixed workflow tests.

Verify complex workflows combining writes, appends, delta logs, and stats.
"""

import pyarrow as pa
import pytest
from milvus_storage import Filesystem, PropertyKeys, Reader, Transaction, Writer
from milvus_storage.manifest import ColumnGroups


class TestMixWorkflow:
    """Test mixed transaction workflows."""

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

    def _read_all_via_manifest(self, path, schema, props):
        """Open a fresh Transaction, read manifest, return all batches."""
        txn = Transaction(path, props)
        manifest = txn.get_manifest()
        txn.close()

        cg = ColumnGroups.from_list(manifest.column_groups)
        reader = Reader(cg, schema, properties=props)
        return list(reader.scan())

    def _read_all_rows_via_manifest(self, path, schema, props):
        """Open a fresh Transaction, read manifest, return row count."""
        batches = self._read_all_via_manifest(path, schema, props)
        return sum(b.num_rows for b in batches)

    def _collect_ids(self, batches):
        """Extract sorted id column from batches."""
        ids = []
        for b in batches:
            ids.extend(b.column("id").to_pylist())
        return sorted(ids)

    def test_append_with_delta_log(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Append data and add delta logs across multiple transactions."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        fs = Filesystem.get(properties=default_properties)

        for i in range(3):
            delta_path = f"{temp_case_path}/delta_{i}.log"
            fs.write_file(delta_path, f"delta_{i}".encode())

            txn = Transaction(temp_case_path, default_properties)

            new_writer = Writer(temp_case_path, simple_schema, default_properties)
            new_writer.write(batch_generator(100, offset=(i + 1) * 1000))
            new_cg = new_writer.close()

            txn.append_files(new_cg)
            txn.add_delta_log(f"delta_{i}.log", i * 10 + 5)
            txn.commit()
            txn.close()

        # Verify manifest has all 3 delta logs
        txn2 = Transaction(temp_case_path, default_properties)
        manifest = txn2.get_manifest()
        txn2.close()

        assert len(manifest.delta_logs) == 3

        # Verify data content: 1000 initial + 3*100 appended
        batches = self._read_all_via_manifest(
            temp_case_path, simple_schema, default_properties
        )
        assert sum(b.num_rows for b in batches) == 1300
        ids = self._collect_ids(batches)
        expected = (
            list(range(1000))
            + list(range(1000, 1100))
            + list(range(2000, 2100))
            + list(range(3000, 3100))
        )
        assert ids == sorted(expected)

    def test_append_with_stat_update(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Append data and update stats in the same transaction."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        # Create stat file via Filesystem.write_file
        fs = Filesystem.get(properties=default_properties)
        stat_path = f"{temp_case_path}/stats/pk_delete.stat"
        fs.create_dir(f"{temp_case_path}/stats")
        fs.write_file(stat_path, b"stat_content")

        txn = Transaction(temp_case_path, default_properties)
        txn.update_stat("pk.delete", ["stats/pk_delete.stat"])

        new_writer = Writer(temp_case_path, simple_schema, default_properties)
        new_writer.write(batch_generator(100, offset=1000))
        new_cg = new_writer.close()

        txn.append_files(new_cg)
        txn.commit()
        txn.close()

        # Verify stats in manifest
        txn2 = Transaction(temp_case_path, default_properties)
        manifest = txn2.get_manifest()
        txn2.close()

        assert len(manifest.stats) >= 1
        stat_keys = [s.key for s in manifest.stats]
        assert "pk.delete" in stat_keys

        # Verify data content: 1000 initial + 100 appended
        batches = self._read_all_via_manifest(
            temp_case_path, simple_schema, default_properties
        )
        assert sum(b.num_rows for b in batches) == 1100
        ids = self._collect_ids(batches)
        expected = list(range(1000)) + list(range(1000, 1100))
        assert ids == sorted(expected)

    def test_read_only_transaction(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Open a transaction just to read the manifest, without committing."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        txn = Transaction(temp_case_path, default_properties)
        manifest = txn.get_manifest()
        version = txn.get_read_version()
        txn.close()  # Close without commit

        assert len(manifest.column_groups) > 0
        assert version >= 0

        # Verify data content
        batches = self._read_all_via_manifest(
            temp_case_path, simple_schema, default_properties
        )
        assert sum(b.num_rows for b in batches) == 1000
        assert self._collect_ids(batches) == list(range(1000))

    def test_chained_write_append_read(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Full chain: initial write -> append -> read all data."""
        # Step 1: initial write + commit
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=1000,
        )

        # Step 2: append
        txn = Transaction(temp_case_path, default_properties)
        new_writer = Writer(temp_case_path, simple_schema, default_properties)
        new_writer.write(batch_generator(500, offset=1000))
        new_cg = new_writer.close()
        txn.append_files(new_cg)
        txn.commit()
        txn.close()

        # Step 3: read all data via manifest and verify content
        batches = self._read_all_via_manifest(
            temp_case_path, simple_schema, default_properties
        )
        assert sum(b.num_rows for b in batches) == 1500
        ids = self._collect_ids(batches)
        assert ids == list(range(1000)) + list(range(1000, 1500))

    def test_mix_append_add_field(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Append N times, add a new column group, then append with the merged schema."""
        # Step 1: initial commit
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=500,
        )

        # Step 2: append 3 more times with original schema
        for i in range(3):
            txn = Transaction(temp_case_path, default_properties)
            w = Writer(temp_case_path, simple_schema, default_properties)
            w.write(batch_generator(100, offset=(i + 1) * 1000))
            cg = w.close()
            txn.append_files(cg)
            txn.commit()
            txn.close()

        # Step 3: add_column_group with a new column "score"
        # Must match existing row count: 500 initial + 3*100 appends = 800
        score_schema = pa.schema([pa.field("score", pa.float32())])
        score_writer = Writer(temp_case_path, score_schema, default_properties)
        score_batch = pa.RecordBatch.from_pydict(
            {"score": pa.array([float(i) for i in range(800)], type=pa.float32())},
            schema=score_schema,
        )
        score_writer.write(score_batch)
        score_cg = score_writer.close()

        txn = Transaction(temp_case_path, default_properties)
        for cg in score_cg.to_list():
            txn.add_column_group(cg)
        txn.commit()
        txn.close()

        # Step 4: append with merged schema using schema_based policy
        # so the Writer splits columns into 2 groups matching the existing structure
        merged_schema = pa.schema(
            list(simple_schema) + [pa.field("score", pa.float32())]
        )
        schema_based_props = dict(default_properties)
        schema_based_props[PropertyKeys.WRITER_POLICY] = "schema_based"
        schema_based_props[PropertyKeys.WRITER_SCHEMA_BASE_PATTERNS] = (
            "id|name|value,score"
        )

        merged_writer = Writer(temp_case_path, merged_schema, schema_based_props)
        merged_batch = pa.RecordBatch.from_pydict(
            {
                "id": list(range(2000, 2200)),
                "name": [f"name_{i}" for i in range(2000, 2200)],
                "value": [float(i) for i in range(2000, 2200)],
                "score": pa.array(
                    [float(i) * 0.1 for i in range(200)], type=pa.float32()
                ),
            },
            schema=merged_schema,
        )
        merged_writer.write(merged_batch)
        merged_cg = merged_writer.close()

        txn = Transaction(temp_case_path, default_properties)
        txn.append_files(merged_cg)
        txn.commit()
        txn.close()

        # Verify: read with old schema — row count and id values
        old_batches = self._read_all_via_manifest(
            temp_case_path, simple_schema, default_properties
        )
        assert sum(b.num_rows for b in old_batches) == 1000
        old_ids = self._collect_ids(old_batches)
        expected_ids = (
            list(range(500))
            + list(range(1000, 1100))
            + list(range(2000, 2100))
            + list(range(3000, 3100))
            + list(range(2000, 2200))
        )
        assert old_ids == sorted(expected_ids)

        # Verify: read with merged schema — all columns present and correct count
        merged_batches = self._read_all_via_manifest(
            temp_case_path, merged_schema, default_properties
        )
        assert sum(b.num_rows for b in merged_batches) == 1000
        for b in merged_batches:
            assert set(b.schema.names) == set(merged_schema.names)
        merged_ids = self._collect_ids(merged_batches)
        assert merged_ids == sorted(expected_ids)

    @pytest.mark.xfail(
        reason="Row count mismatch validation not enforced in add_column_group",
    )
    def test_add_field_after_append_fail(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Add field fails when: duplicate column name, or row count mismatch."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=500,
        )

        # Case 1: duplicate column name — add a column group with "id" which already exists
        dup_schema = pa.schema([pa.field("id", pa.int64())])
        dup_writer = Writer(temp_case_path, dup_schema, default_properties)
        dup_writer.write(
            pa.RecordBatch.from_pydict({"id": list(range(500))}, schema=dup_schema)
        )
        dup_cg = dup_writer.close()

        txn = Transaction(temp_case_path, default_properties)
        for cg in dup_cg.to_list():
            txn.add_column_group(cg)
        with pytest.raises(Exception):
            txn.commit()
        txn.close()

        # Case 2: row count mismatch — existing has 500 rows, add 300
        new_schema = pa.schema([pa.field("score", pa.float32())])
        new_writer = Writer(temp_case_path, new_schema, default_properties)
        new_writer.write(
            pa.RecordBatch.from_pydict(
                {"score": pa.array([float(i) for i in range(300)], type=pa.float32())},
                schema=new_schema,
            )
        )
        new_cg = new_writer.close()

        txn2 = Transaction(temp_case_path, default_properties)
        for cg in new_cg.to_list():
            txn2.add_column_group(cg)
        with pytest.raises(Exception):
            txn2.commit()
        txn2.close()

        # Original data intact after both failures — verify content
        batches = self._read_all_via_manifest(
            temp_case_path, simple_schema, default_properties
        )
        assert sum(b.num_rows for b in batches) == 500
        assert self._collect_ids(batches) == list(range(500))

    def test_append_after_add_field_fail(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Append after add_field fails when: wrong policy, or stale schema."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=500,
        )

        # Add a new column group (score)
        score_schema = pa.schema([pa.field("score", pa.float32())])
        score_writer = Writer(temp_case_path, score_schema, default_properties)
        score_writer.write(
            pa.RecordBatch.from_pydict(
                {"score": pa.array([float(i) for i in range(500)], type=pa.float32())},
                schema=score_schema,
            )
        )
        score_cg = score_writer.close()

        txn = Transaction(temp_case_path, default_properties)
        for cg in score_cg.to_list():
            txn.add_column_group(cg)
        txn.commit()
        txn.close()

        # Case 1: wrong policy — default single policy produces 1 group, manifest has 2
        merged_schema = pa.schema(
            list(simple_schema) + [pa.field("score", pa.float32())]
        )
        bad_writer = Writer(temp_case_path, merged_schema, default_properties)
        bad_writer.write(
            pa.RecordBatch.from_pydict(
                {
                    "id": list(range(100)),
                    "name": [f"n_{i}" for i in range(100)],
                    "value": [float(i) for i in range(100)],
                    "score": pa.array(
                        [float(i) for i in range(100)], type=pa.float32()
                    ),
                },
                schema=merged_schema,
            )
        )
        bad_cg = bad_writer.close()

        txn2 = Transaction(temp_case_path, default_properties)
        txn2.append_files(bad_cg)
        with pytest.raises(Exception):
            txn2.commit()
        txn2.close()

        # Case 2: correct policy but stale schema (missing score column) — still 1 group
        schema_based_props = dict(default_properties)
        schema_based_props[PropertyKeys.WRITER_POLICY] = "schema_based"
        schema_based_props[PropertyKeys.WRITER_SCHEMA_BASE_PATTERNS] = (
            "id|name|value,score"
        )

        stale_writer = Writer(temp_case_path, simple_schema, schema_based_props)
        stale_writer.write(batch_generator(100, offset=500))
        stale_cg = stale_writer.close()

        txn3 = Transaction(temp_case_path, default_properties)
        txn3.append_files(stale_cg)
        with pytest.raises(Exception):
            txn3.commit()
        txn3.close()

        # Original data intact after both failures — verify content
        batches = self._read_all_via_manifest(
            temp_case_path, simple_schema, default_properties
        )
        assert sum(b.num_rows for b in batches) == 500
        assert self._collect_ids(batches) == list(range(500))
