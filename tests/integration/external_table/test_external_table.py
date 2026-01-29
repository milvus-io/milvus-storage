"""
External table tests.

Verify external file exploration and manifest reading for parquet and vortex files.

NOTE: These tests use pyarrow/vortex to write files directly to local filesystem,
so they only work with local backend. S3/cloud backends are marked as xfail.
"""

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import vortex as vx
from milvus_storage import ExternalTable, Reader, Transaction
from milvus_storage.manifest import ColumnGroup, ColumnGroupFile, ColumnGroups

from ...config import get_config


def _generate_batch(
    schema: pa.Schema, num_rows: int, offset: int = 0
) -> pa.RecordBatch:
    """Generate a batch with the given schema."""
    data = {}
    for field in schema:
        name = field.name
        if pa.types.is_int64(field.type):
            data[name] = list(range(offset, offset + num_rows))
        elif pa.types.is_string(field.type):
            data[name] = [f"v_{i}" for i in range(offset, offset + num_rows)]
        elif pa.types.is_float64(field.type):
            data[name] = [float(i) for i in range(offset, offset + num_rows)]
        else:
            data[name] = [None] * num_rows
    return pa.RecordBatch.from_pydict(data, schema=schema)


def _write_parquet_files(
    directory: str, schema: pa.Schema, num_files: int, rows_per_file: int
) -> list:
    """Write parquet files using pyarrow directly."""
    os.makedirs(directory, exist_ok=True)
    file_paths = []
    for i in range(num_files):
        batch = _generate_batch(schema, rows_per_file, offset=i * rows_per_file)
        table = pa.Table.from_batches([batch])
        file_path = f"{directory}/data_{i}.parquet"
        pq.write_table(table, file_path)
        file_paths.append(file_path)
    return file_paths


def _write_vortex_files(
    directory: str, schema: pa.Schema, num_files: int, rows_per_file: int
) -> list:
    """Write vortex files using vortex-data library."""
    os.makedirs(directory, exist_ok=True)
    file_paths = []
    for i in range(num_files):
        batch = _generate_batch(schema, rows_per_file, offset=i * rows_per_file)
        table = pa.Table.from_batches([batch])
        vtx = vx.array(table)
        file_path = f"{directory}/data_{i}.vortex"
        vx.io.write(vtx, file_path)
        file_paths.append(file_path)
    return file_paths


@pytest.mark.xfail(
    reason="External table tests use pyarrow/vortex to write files directly to local filesystem",
)
class TestExternalTable:
    """Test external table operations."""

    def test_explore_parquet_directory(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
        test_config,
    ):
        """Explore a directory containing parquet files."""
        # Get full path for writing files (need to use OS path for pyarrow)
        root_path = test_config.root_path
        full_base_path = f"{root_path}/{temp_case_path}"
        data_path = f"{full_base_path}/data"

        # Write parquet files using pyarrow to full OS path
        num_files = 5
        rows_per_file = 200
        _write_parquet_files(data_path, simple_schema, num_files, rows_per_file)

        # For ExternalTable.explore, use relative paths (relative to SubtreeFilesystem root)
        relative_base = temp_case_path
        relative_data = f"{temp_case_path}/data"

        # Explore should find the parquet files
        found_files, manifest_path = ExternalTable.explore(
            columns=simple_schema.names,
            format="parquet",
            base_dir=relative_base,
            explore_dir=relative_data,
            properties=default_properties,
        )

        assert found_files == num_files
        assert manifest_path != ""

        # Read and verify manifest
        manifest = ExternalTable.read_manifest(
            manifest_path, properties=default_properties
        )
        assert len(manifest.column_groups) > 0

        total_files = sum(len(cg.files) for cg in manifest.column_groups)
        assert total_files == num_files

    def test_explore_vortex_directory(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
        test_config,
    ):
        """Explore a directory containing vortex files."""
        root_path = test_config.root_path
        full_base_path = f"{root_path}/{temp_case_path}"
        data_path = f"{full_base_path}/data"

        # Write vortex files
        num_files = 5
        rows_per_file = 200
        _write_vortex_files(data_path, simple_schema, num_files, rows_per_file)

        relative_base = temp_case_path
        relative_data = f"{temp_case_path}/data"

        # Explore should find the vortex files
        found_files, manifest_path = ExternalTable.explore(
            columns=simple_schema.names,
            format="vortex",
            base_dir=relative_base,
            explore_dir=relative_data,
            properties=default_properties,
        )

        assert found_files == num_files
        assert manifest_path != ""

        # Read and verify manifest
        manifest = ExternalTable.read_manifest(
            manifest_path, properties=default_properties
        )
        assert len(manifest.column_groups) > 0

        total_files = sum(len(cg.files) for cg in manifest.column_groups)
        assert total_files == num_files

    def test_get_file_info_parquet(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
        test_config,
    ):
        """Get file info for a parquet file."""
        root_path = test_config.root_path
        full_path = f"{root_path}/{temp_case_path}"
        os.makedirs(full_path, exist_ok=True)

        num_rows = 500
        batch = _generate_batch(simple_schema, num_rows)
        table = pa.Table.from_batches([batch])
        file_path = f"{full_path}/test.parquet"
        pq.write_table(table, file_path)

        # Get file info using relative path
        relative_file = f"{temp_case_path}/test.parquet"
        row_count = ExternalTable.get_file_info(
            format="parquet",
            file_path=relative_file,
            properties=default_properties,
        )
        assert row_count == num_rows

    def test_get_file_info_vortex(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
        test_config,
    ):
        """Get file info for a vortex file."""
        root_path = test_config.root_path
        full_path = f"{root_path}/{temp_case_path}"
        os.makedirs(full_path, exist_ok=True)

        num_rows = 500
        batch = _generate_batch(simple_schema, num_rows)
        table = pa.Table.from_batches([batch])
        vtx = vx.array(table)
        file_path = f"{full_path}/test.vortex"
        vx.io.write(vtx, file_path)

        # Get file info using relative path
        relative_file = f"{temp_case_path}/test.vortex"
        row_count = ExternalTable.get_file_info(
            format="vortex",
            file_path=relative_file,
            properties=default_properties,
        )
        assert row_count == num_rows

    def test_explore_empty_directory(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
        test_config,
    ):
        """Explore an empty directory returns 0 files."""
        root_path = test_config.root_path
        empty_full_path = f"{root_path}/{temp_case_path}/empty"
        os.makedirs(empty_full_path, exist_ok=True)

        relative_base = temp_case_path
        relative_empty = f"{temp_case_path}/empty"

        num_files, manifest_path = ExternalTable.explore(
            columns=simple_schema.names,
            format="parquet",
            base_dir=relative_base,
            explore_dir=relative_empty,
            properties=default_properties,
        )

        assert num_files == 0

    def test_external_table_import_parquet(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
        test_config,
    ):
        """Import external parquet files into storage via transaction."""
        from milvus_storage import PropertyKeys

        root_path = test_config.root_path

        # Setup paths: explore_path for external files, base_path for imported data
        explore_path = f"{root_path}/{temp_case_path}/external"
        base_path = f"{root_path}/{temp_case_path}/imported"

        full_explore_path = explore_path

        # Use "/" as root_path so file paths are absolute
        props_with_root = {**default_properties, PropertyKeys.FS_ROOT_PATH: "/"}

        # Write parquet files to explore_path
        num_files = 3
        rows_per_file = 100
        _write_parquet_files(full_explore_path, simple_schema, num_files, rows_per_file)

        # 1. Explore external files (use absolute paths with root="/")
        found_files, manifest_path = ExternalTable.explore(
            columns=simple_schema.names,
            format="parquet",
            base_dir=explore_path,
            explore_dir=explore_path,
            properties=props_with_root,
        )
        assert found_files == num_files

        # 2. Read manifest
        manifest = ExternalTable.read_manifest(
            manifest_path, properties=props_with_root
        )

        # 3. Update file indices using get_file_info
        updated_column_groups = []
        for cg in manifest.column_groups:
            updated_files = []
            for f in cg.files:
                row_count = ExternalTable.get_file_info(
                    format="parquet",
                    file_path=f.path,
                    properties=props_with_root,
                )
                updated_files.append(
                    ColumnGroupFile(
                        path=f.path,
                        start_index=0,
                        end_index=row_count,
                        metadata=f.metadata,
                    )
                )
            updated_column_groups.append(
                ColumnGroup(
                    columns=cg.columns,
                    format=cg.format,
                    files=updated_files,
                )
            )

        # 4. Write to base_path via transaction
        cg_to_write = ColumnGroups.from_list(updated_column_groups)
        txn = Transaction(base_path, props_with_root)
        txn.append_files(cg_to_write)
        txn.commit()
        txn.close()

        # 5. Read via transaction and verify
        txn2 = Transaction(base_path, props_with_root)
        read_manifest = txn2.get_manifest()
        txn2.close()

        cg_for_reader = ColumnGroups.from_list(read_manifest.column_groups)
        reader = Reader(cg_for_reader, simple_schema, properties=props_with_root)

        total_rows = sum(b.num_rows for b in reader.scan())
        expected_rows = num_files * rows_per_file
        assert total_rows == expected_rows

        reader.close()

    def test_external_table_import_vortex(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
        test_config,
    ):
        """Import external vortex files into storage via transaction."""
        from milvus_storage import PropertyKeys

        root_path = test_config.root_path

        # Setup paths: explore_path for external files, base_path for imported data
        explore_path = f"{root_path}/{temp_case_path}/external"
        base_path = f"{root_path}/{temp_case_path}/imported"

        full_explore_path = explore_path

        # Use "/" as root_path so file paths are absolute
        props_with_root = {**default_properties, PropertyKeys.FS_ROOT_PATH: "/"}

        # Write vortex files to explore_path
        num_files = 3
        rows_per_file = 100
        _write_vortex_files(full_explore_path, simple_schema, num_files, rows_per_file)

        # 1. Explore external files (use absolute paths with root="/")
        found_files, manifest_path = ExternalTable.explore(
            columns=simple_schema.names,
            format="vortex",
            base_dir=explore_path,
            explore_dir=explore_path,
            properties=props_with_root,
        )
        assert found_files == num_files

        # 2. Read manifest
        manifest = ExternalTable.read_manifest(
            manifest_path, properties=props_with_root
        )

        # 3. Update file indices using get_file_info
        updated_column_groups = []
        for cg in manifest.column_groups:
            updated_files = []
            for f in cg.files:
                row_count = ExternalTable.get_file_info(
                    format="vortex",
                    file_path=f.path,
                    properties=props_with_root,
                )
                updated_files.append(
                    ColumnGroupFile(
                        path=f.path,
                        start_index=0,
                        end_index=row_count,
                        metadata=f.metadata,
                    )
                )
            updated_column_groups.append(
                ColumnGroup(
                    columns=cg.columns,
                    format=cg.format,
                    files=updated_files,
                )
            )

        # 4. Write to base_path via transaction
        cg_to_write = ColumnGroups.from_list(updated_column_groups)
        txn = Transaction(base_path, props_with_root)
        txn.append_files(cg_to_write)
        txn.commit()
        txn.close()

        # 5. Read via transaction and verify
        txn2 = Transaction(base_path, props_with_root)
        read_manifest = txn2.get_manifest()
        txn2.close()

        cg_for_reader = ColumnGroups.from_list(read_manifest.column_groups)
        reader = Reader(cg_for_reader, simple_schema, properties=props_with_root)

        total_rows = sum(b.num_rows for b in reader.scan())
        expected_rows = num_files * rows_per_file
        assert total_rows == expected_rows

        reader.close()
