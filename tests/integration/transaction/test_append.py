"""
Transaction append tests.

Verify data append through transactions.
"""

import pyarrow as pa
from milvus_storage import Reader, Transaction, Writer
from milvus_storage.manifest import ColumnGroups


class TestTransactionAppend:
    """Test appending data through transactions."""

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

    def _read_all_rows_via_manifest(self, path, schema, props):
        """Open a fresh Transaction, read manifest, return row count."""
        txn = Transaction(path, props)
        manifest = txn.get_manifest()
        txn.close()

        cg = ColumnGroups.from_list(manifest.column_groups)
        reader = Reader(cg, schema, properties=props)
        return sum(b.num_rows for b in reader.scan())

    def test_single_append(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Single append adds data correctly."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        # Append new data
        txn = Transaction(temp_case_path, default_properties)
        new_writer = Writer(temp_case_path, simple_schema, default_properties)
        new_writer.write(batch_generator(500, offset=1000))
        new_cg = new_writer.close()

        txn.append_files(new_cg)
        version = txn.commit()
        txn.close()

        assert version >= 0
        assert (
            self._read_all_rows_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            == 1500
        )

    def test_manifest_version_tracking(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Committed version increases with each append."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        versions = []
        for i in range(5):
            txn = Transaction(temp_case_path, default_properties)
            read_ver = txn.get_read_version()

            new_writer = Writer(temp_case_path, simple_schema, default_properties)
            new_writer.write(batch_generator(100, offset=(i + 1) * 1000))
            new_cg = new_writer.close()

            txn.append_files(new_cg)
            committed_ver = txn.commit()
            txn.close()

            assert committed_ver > read_ver
            versions.append(committed_ver)

        for i in range(1, len(versions)):
            assert versions[i] > versions[i - 1]

    def test_append_different_batch_sizes(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Appending batches of varying sizes works correctly."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=100,
        )

        total_rows = 100
        for batch_size in [10, 100, 1000, 50]:
            txn = Transaction(temp_case_path, default_properties)
            new_writer = Writer(temp_case_path, simple_schema, default_properties)
            new_writer.write(batch_generator(batch_size, offset=total_rows))
            new_cg = new_writer.close()

            txn.append_files(new_cg)
            txn.commit()
            txn.close()
            total_rows += batch_size

        assert (
            self._read_all_rows_via_manifest(
                temp_case_path, simple_schema, default_properties
            )
            == total_rows
        )

    def test_get_manifest_after_append(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Manifest reflects appended data."""
        self._commit_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        # Append
        txn = Transaction(temp_case_path, default_properties)
        new_writer = Writer(temp_case_path, simple_schema, default_properties)
        new_writer.write(batch_generator(500, offset=1000))
        new_cg = new_writer.close()
        txn.append_files(new_cg)
        txn.commit()
        txn.close()

        # Read latest manifest
        txn2 = Transaction(temp_case_path, default_properties)
        manifest = txn2.get_manifest()
        txn2.close()

        assert len(manifest.column_groups) > 0
        total_files = sum(len(cg.files) for cg in manifest.column_groups)
        assert total_files >= 2
