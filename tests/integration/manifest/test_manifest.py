"""
Manifest tests.

Verify manifest structure, versioning, and content after operations.
"""

import pyarrow as pa
from milvus_storage import Transaction, Writer


class TestManifest:
    """Test manifest structure and content."""

    def _write_initial_data(self, path, schema, batch_generator, props, rows=1000):
        """Write initial data and commit via transaction."""
        writer = Writer(path, schema, props)
        writer.write(batch_generator(rows, offset=0))
        cg = writer.close()

        txn = Transaction(path, props)
        txn.append_files(cg)
        txn.commit()
        txn.close()
        return cg

    def test_initial_manifest_structure(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Initial manifest contains correct column groups."""
        self._write_initial_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        txn = Transaction(temp_case_path, default_properties)
        manifest = txn.get_manifest()
        txn.close()

        assert len(manifest.column_groups) > 0
        cg = manifest.column_groups[0]
        assert len(cg.columns) > 0
        assert cg.format != ""
        assert len(cg.files) > 0

    def test_manifest_column_names(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Manifest column names match schema."""
        self._write_initial_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        txn = Transaction(temp_case_path, default_properties)
        manifest = txn.get_manifest()
        txn.close()

        all_columns = set()
        for cg in manifest.column_groups:
            all_columns.update(cg.columns)

        schema_columns = set(simple_schema.names)
        assert all_columns == schema_columns

    def test_manifest_file_indices(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """File start/end indices are consistent."""
        self._write_initial_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=2000,
        )

        txn = Transaction(temp_case_path, default_properties)
        manifest = txn.get_manifest()
        txn.close()

        for cg in manifest.column_groups:
            for f in cg.files:
                # start_index < end_index
                assert (
                    f.start_index < f.end_index
                ), f"Invalid range: [{f.start_index}, {f.end_index}) in {f.path}"
                assert f.path != ""

    def test_manifest_grows_after_appends(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Manifest file count grows after appends."""
        self._write_initial_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        txn_check = Transaction(temp_case_path, default_properties)
        m1 = txn_check.get_manifest()
        txn_check.close()
        initial_files = sum(len(cg.files) for cg in m1.column_groups)

        # Append
        txn = Transaction(temp_case_path, default_properties)
        new_writer = Writer(temp_case_path, simple_schema, default_properties)
        new_writer.write(batch_generator(500, offset=1000))
        new_cg = new_writer.close()
        txn.append_files(new_cg)
        txn.commit()
        txn.close()

        txn_check2 = Transaction(temp_case_path, default_properties)
        m2 = txn_check2.get_manifest()
        txn_check2.close()
        new_files = sum(len(cg.files) for cg in m2.column_groups)

        assert new_files > initial_files

    def test_manifest_delta_logs_initially_empty(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Initial manifest has no delta logs."""
        self._write_initial_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        txn = Transaction(temp_case_path, default_properties)
        manifest = txn.get_manifest()
        txn.close()

        assert len(manifest.delta_logs) == 0

    def test_manifest_stats_initially_empty(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Initial manifest has no stats."""
        self._write_initial_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        txn = Transaction(temp_case_path, default_properties)
        manifest = txn.get_manifest()
        txn.close()

        assert len(manifest.stats) == 0

    def test_manifest_format(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        data_format,
    ):
        """Manifest format matches configured format."""
        self._write_initial_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        txn = Transaction(temp_case_path, default_properties)
        manifest = txn.get_manifest()
        txn.close()

        for cg in manifest.column_groups:
            assert (
                cg.format == data_format
            ), f"Expected format {data_format}, got {cg.format}"
