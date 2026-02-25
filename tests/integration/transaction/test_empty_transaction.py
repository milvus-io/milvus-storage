"""
Empty transaction tests.

Verify that committing a transaction without any modifications is rejected,
and that rejection does not corrupt existing data or block subsequent operations.
"""

import pyarrow as pa
import pytest
from milvus_storage import Reader, Transaction, Writer
from milvus_storage.manifest import ColumnGroups


class TestEmptyTransaction:
    """Test empty transaction behavior."""

    def _commit_data(self, path, schema, batch_gen, props, rows=500):
        """Write data and commit via transaction."""
        writer = Writer(path, schema, props)
        writer.write(batch_gen(rows, offset=0))
        cg = writer.close()

        txn = Transaction(path, props)
        txn.append_files(cg)
        txn.commit()
        txn.close()

    def _read_and_verify(self, path, schema, props, expected_ids):
        """Read via manifest, verify row count and id values."""
        txn = Transaction(path, props)
        manifest = txn.get_manifest()
        txn.close()

        cg = ColumnGroups.from_list(manifest.column_groups)
        reader = Reader(cg, schema, properties=props)
        ids = []
        for b in reader.scan():
            ids.extend(b.column("id").to_pylist())
        assert sorted(ids) == sorted(expected_ids)

    def test_empty_commit_rejected(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Empty commit is rejected; existing data remains intact."""
        # Empty transaction should be rejected
        txn = Transaction(temp_case_path, default_properties)
        with pytest.raises(Exception):
            txn.commit()
        txn.close()

        # No manifest exists — verify empty state
        txn2 = Transaction(temp_case_path, default_properties)
        manifest = txn2.get_manifest()
        txn2.close()
        assert manifest.column_groups == []

    def test_multiple_empty_commits_rejected(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Multiple consecutive empty commits are all rejected; data stays intact."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=300,
        )

        for _ in range(3):
            txn = Transaction(temp_case_path, default_properties)
            with pytest.raises(Exception):
                txn.commit()
            txn.close()

        # Data still intact
        self._read_and_verify(
            temp_case_path, simple_schema, default_properties, list(range(300))
        )

    def test_empty_commit_then_real_append(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Rejected empty commit does not block subsequent real appends."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=500,
        )

        # Empty commit rejected
        txn = Transaction(temp_case_path, default_properties)
        with pytest.raises(Exception):
            txn.commit()
        txn.close()

        # Real append still works
        txn2 = Transaction(temp_case_path, default_properties)
        w = Writer(temp_case_path, simple_schema, default_properties)
        w.write(batch_generator(200, offset=1000))
        cg = w.close()
        txn2.append_files(cg)
        txn2.commit()
        txn2.close()

        # Verify all data
        self._read_and_verify(
            temp_case_path,
            simple_schema,
            default_properties,
            list(range(500)) + list(range(1000, 1200)),
        )
