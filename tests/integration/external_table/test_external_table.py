"""
External table tests.

Verify external file exploration and manifest reading for parquet, vortex, and lance files.
Works with both local and S3/cloud backends.
"""

import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import vortex as vx
from milvus_storage import ExternalTable, Reader, Transaction
from milvus_storage.manifest import ColumnGroup, ColumnGroupFile, ColumnGroups


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


def _get_pyarrow_filesystem(test_config):
    """Get a pyarrow filesystem for writing test data.

    Returns None for local backend (use os operations directly).
    Returns pyarrow.fs.S3FileSystem for S3-compatible backends.
    """
    if test_config.is_local:
        return None

    if test_config.is_s3_compatible:
        import pyarrow.fs as pafs

        config = test_config.backend_config
        address = config.get("address", "")
        # Extract host:port from address (remove http:// or https://)
        endpoint = address.replace("http://", "").replace("https://", "")
        scheme = "http" if address.startswith("http://") else "https"

        return pafs.S3FileSystem(
            access_key=config.get("access_key", ""),
            secret_key=config.get("secret_key", ""),
            endpoint_override=endpoint,
            scheme=scheme,
            region=config.get("region", "") or "us-east-1",
        )

    pytest.skip(
        f"Unsupported backend for external table test: {test_config.storage_backend}"
    )


def _get_write_path(test_config, relative_path):
    """Get the full write path for the given relative path.

    For local: returns OS absolute path (root_path/relative_path)
    For S3: returns bucket/relative_path (pyarrow S3 format)
    """
    if test_config.is_local:
        return f"{test_config.root_path}/{relative_path}"
    else:
        return f"{test_config.bucket_name}/{relative_path}"


def _get_extfs_properties(default_properties, test_config, alias="ext1"):
    """Add extfs properties mirroring the default fs config.

    Duplicates fs.* properties as extfs.<alias>.* properties,
    so the external filesystem resolution can find matching config.

    For local: sets bucket_name to "local" (matching URI convention).
    For S3: keeps full address (e.g., http://localhost:9000) so consumers
            like Lance can determine the correct scheme. resolve_config
            normalizes the address during URI matching.
    """
    props = default_properties.copy()
    config = test_config.backend_config

    if test_config.is_local:
        props[f"extfs.{alias}.storage_type"] = "local"
        props[f"extfs.{alias}.root_path"] = test_config.root_path
        props[f"extfs.{alias}.bucket_name"] = "local"
    else:
        props[f"extfs.{alias}.storage_type"] = "remote"
        if "cloud_provider" in config:
            props[f"extfs.{alias}.cloud_provider"] = config["cloud_provider"]
        if "address" in config:
            props[f"extfs.{alias}.address"] = config["address"]
        if test_config.bucket_name:
            props[f"extfs.{alias}.bucket_name"] = test_config.bucket_name
        if "access_key" in config:
            props[f"extfs.{alias}.access_key_id"] = config["access_key"]
        if "secret_key" in config:
            props[f"extfs.{alias}.access_key_value"] = config["secret_key"]
        if "region" in config:
            props[f"extfs.{alias}.region"] = config.get("region", "")

    return props


def _get_explore_uri(test_config, relative_path):
    """Build a URI for explore_dir that triggers external fs resolution.

    StorageUri::Parse extracts scheme/address/bucket_name from the URI.
    FilesystemCache::get matches extfs config by address + bucket_name.

    For local: local:///local/<relative_path>
    For S3:    aws://<endpoint>/<bucket>/<relative_path>
    """
    if test_config.is_local:
        return f"local:///local/{relative_path}"
    else:
        config = test_config.backend_config
        address = config.get("address", "")
        endpoint = address.replace("http://", "").replace("https://", "")
        return f"aws://{endpoint}/{test_config.bucket_name}/{relative_path}"


def _get_lance_write_uri(test_config, relative_path):
    """Get the Lance URI for writing a dataset.

    For local: absolute path (root_path/relative_path)
    For S3: s3://bucket/relative_path
    """
    if test_config.is_local:
        return f"{test_config.root_path}/{relative_path}"
    else:
        return f"s3://{test_config.bucket_name}/{relative_path}"


def _get_lance_storage_options(test_config):
    """Get Lance storage options for cloud backends.

    Returns None for local storage.
    Returns dict with AWS-style keys for S3-compatible backends.
    """
    if test_config.is_local:
        return None

    config = test_config.backend_config
    options = {}

    if test_config.is_s3_compatible:
        if "access_key" in config:
            options["aws_access_key_id"] = config["access_key"]
        if "secret_key" in config:
            options["aws_secret_access_key"] = config["secret_key"]
        if "region" in config:
            options["aws_region"] = config.get("region", "") or "us-east-1"
        if "address" in config:
            address = config["address"]
            options["aws_endpoint"] = address
            if address.startswith("http://"):
                options["allow_http"] = "true"

    return options if options else None


def _write_parquet_files(
    directory: str,
    schema: pa.Schema,
    num_files: int,
    rows_per_file: int,
    filesystem=None,
) -> list:
    """Write parquet files using pyarrow.

    Args:
        filesystem: Optional pyarrow filesystem. If provided, files are written
            locally first then uploaded (for S3/remote backends).
    """
    if filesystem is None:
        os.makedirs(directory, exist_ok=True)
    file_paths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(num_files):
            batch = _generate_batch(schema, rows_per_file, offset=i * rows_per_file)
            table = pa.Table.from_batches([batch])
            target_path = f"{directory}/data_{i}.parquet"
            if filesystem is not None:
                local_path = f"{tmp_dir}/data_{i}.parquet"
                pq.write_table(table, local_path)
                with open(local_path, "rb") as f:
                    with filesystem.open_output_stream(target_path) as out:
                        out.write(f.read())
            else:
                pq.write_table(table, target_path)
            file_paths.append(target_path)
    return file_paths


def _write_vortex_files(
    directory: str,
    schema: pa.Schema,
    num_files: int,
    rows_per_file: int,
    filesystem=None,
) -> list:
    """Write vortex files using vortex-data library.

    Args:
        filesystem: Optional pyarrow filesystem. If provided, files are written
            locally first then uploaded (for S3/remote backends).
    """
    if filesystem is None:
        os.makedirs(directory, exist_ok=True)
    file_paths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(num_files):
            batch = _generate_batch(schema, rows_per_file, offset=i * rows_per_file)
            table = pa.Table.from_batches([batch])
            vtx = vx.array(table)
            target_path = f"{directory}/data_{i}.vortex"
            if filesystem is not None:
                local_path = f"{tmp_dir}/data_{i}.vortex"
                vx.io.write(vtx, local_path)
                with open(local_path, "rb") as f:
                    with filesystem.open_output_stream(target_path) as out:
                        out.write(f.read())
            else:
                vx.io.write(vtx, target_path)
            file_paths.append(target_path)
    return file_paths


def _write_single_file(test_config, write_path, schema, num_rows, fmt="parquet"):
    """Write a single test file (parquet or vortex).

    Returns the pyarrow filesystem used (None for local).
    """
    write_fs = _get_pyarrow_filesystem(test_config)

    if write_fs is None:
        os.makedirs(write_path, exist_ok=True)

    batch = _generate_batch(schema, num_rows)
    table = pa.Table.from_batches([batch])

    ext = "parquet" if fmt == "parquet" else "vortex"
    target_path = f"{write_path}/test.{ext}"

    if write_fs is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = f"{tmp_dir}/test.{ext}"
            if fmt == "parquet":
                pq.write_table(table, local_path)
            else:
                vtx = vx.array(table)
                vx.io.write(vtx, local_path)
            with open(local_path, "rb") as f:
                with write_fs.open_output_stream(target_path) as out:
                    out.write(f.read())
    else:
        if fmt == "parquet":
            pq.write_table(table, target_path)
        else:
            vtx = vx.array(table)
            vx.io.write(vtx, target_path)

    return write_fs


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
        write_fs = _get_pyarrow_filesystem(test_config)
        write_path = _get_write_path(test_config, f"{temp_case_path}/data")

        num_files = 5
        rows_per_file = 200
        _write_parquet_files(
            write_path, simple_schema, num_files, rows_per_file, filesystem=write_fs
        )

        # Use relative paths (relative to SubtreeFilesystem root)
        relative_base = temp_case_path
        relative_data = f"{temp_case_path}/data"

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
        write_fs = _get_pyarrow_filesystem(test_config)
        write_path = _get_write_path(test_config, f"{temp_case_path}/data")

        num_files = 5
        rows_per_file = 200
        _write_vortex_files(
            write_path, simple_schema, num_files, rows_per_file, filesystem=write_fs
        )

        relative_base = temp_case_path
        relative_data = f"{temp_case_path}/data"

        found_files, manifest_path = ExternalTable.explore(
            columns=simple_schema.names,
            format="vortex",
            base_dir=relative_base,
            explore_dir=relative_data,
            properties=default_properties,
        )

        assert found_files == num_files
        assert manifest_path != ""

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
        write_path = _get_write_path(test_config, temp_case_path)
        num_rows = 500
        _write_single_file(
            test_config, write_path, simple_schema, num_rows, fmt="parquet"
        )

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
        write_path = _get_write_path(test_config, temp_case_path)
        num_rows = 500
        _write_single_file(
            test_config, write_path, simple_schema, num_rows, fmt="vortex"
        )

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
        if not test_config.is_local:
            pytest.skip("Empty directory test only works on local filesystem")

        empty_full_path = f"{test_config.root_path}/{temp_case_path}/empty"
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

    @staticmethod
    def _verify_import(base_rel, schema, expected_rows, properties):
        """Verify imported data by reading with 3 methods: scan, get_chunk, take.

        Reads the committed manifest from base_rel, prints it, then verifies
        data integrity using all Reader access patterns.
        """
        # Read manifest from committed transaction
        txn = Transaction(base_rel, properties)
        manifest = txn.get_manifest()
        txn.close()

        cg_for_reader = ColumnGroups.from_list(manifest.column_groups)

        # Print manifest via debug_string
        print(f"\n=== Base Manifest (path: {base_rel}) ===")
        print(cg_for_reader.debug_string())

        # --- Method 1: scan (RecordBatchReader) ---
        print("\n--- Read method 1: scan ---")
        reader = Reader(cg_for_reader, schema, properties=properties)
        scan_rows = sum(b.num_rows for b in reader.scan())
        print(f"  scan total rows: {scan_rows}")
        assert scan_rows == expected_rows
        reader.close()

        # --- Method 2: get_chunk (column group 0) ---
        print("\n--- Read method 2: get_chunk (column_group=0) ---")
        reader = Reader(cg_for_reader, schema, properties=properties)
        chunk_reader = reader.get_chunk_reader(0)
        num_chunks = chunk_reader.get_number_of_chunks()
        print(f"  number of chunks: {num_chunks}")
        chunk_rows = 0
        for i in range(num_chunks):
            chunk = chunk_reader.get_chunk(i)
            print(f"  chunk[{i}]: {chunk.num_rows} rows")
            chunk_rows += chunk.num_rows
        print(f"  get_chunk total rows: {chunk_rows}")
        assert chunk_rows == expected_rows
        chunk_reader.close()
        reader.close()

        # --- Method 3: take (random access) ---
        print("\n--- Read method 3: take ---")
        reader = Reader(cg_for_reader, schema, properties=properties)
        # Build sorted indices: first, middle, last
        take_indices = sorted(
            set([0, expected_rows // 4, expected_rows // 2, expected_rows - 1])
        )
        print(f"  take indices: {take_indices}")
        batches = reader.take(take_indices)
        take_rows = sum(b.num_rows for b in batches)
        print(f"  take total rows: {take_rows}")
        assert take_rows == len(take_indices)
        reader.close()

    def test_external_table_import_parquet(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
        test_config,
    ):
        """Import external parquet files into storage via transaction."""
        write_fs = _get_pyarrow_filesystem(test_config)

        explore_rel = f"{temp_case_path}/external"
        explore_meta_rel = f"{temp_case_path}/explore_meta"
        base_rel = f"{temp_case_path}/imported"

        write_path = _get_write_path(test_config, explore_rel)

        # Write parquet files
        num_files = 3
        rows_per_file = 100
        _write_parquet_files(
            write_path, simple_schema, num_files, rows_per_file, filesystem=write_fs
        )

        # Add extfs properties (same config as default fs)
        props = _get_extfs_properties(default_properties, test_config)

        # 1. Explore external files (URI triggers external fs resolution)
        explore_uri = _get_explore_uri(test_config, explore_rel)
        print(f"  explore_dir URI: {explore_uri}")
        found_files, manifest_path = ExternalTable.explore(
            columns=simple_schema.names,
            format="parquet",
            base_dir=explore_meta_rel,
            explore_dir=explore_uri,
            properties=props,
        )
        assert found_files == num_files

        # 2. Read manifest and update file indices via get_file_info
        manifest = ExternalTable.read_manifest(manifest_path, properties=props)
        updated_column_groups = []
        for cg in manifest.column_groups:
            updated_files = []
            for f in cg.files:
                row_count = ExternalTable.get_file_info(
                    format="parquet",
                    file_path=f.path,
                    properties=props,
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
                ColumnGroup(columns=cg.columns, format=cg.format, files=updated_files)
            )

        # 3. Commit to base_path via transaction
        cg_to_write = ColumnGroups.from_list(updated_column_groups)
        txn = Transaction(base_rel, props)
        txn.append_files(cg_to_write)
        txn.commit()
        txn.close()

        # 4. Verify with all 3 read methods
        self._verify_import(base_rel, simple_schema, num_files * rows_per_file, props)

    def test_external_table_import_vortex(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
        test_config,
    ):
        """Import external vortex files into storage via transaction."""
        write_fs = _get_pyarrow_filesystem(test_config)

        explore_rel = f"{temp_case_path}/external"
        explore_meta_rel = f"{temp_case_path}/explore_meta"
        base_rel = f"{temp_case_path}/imported"

        write_path = _get_write_path(test_config, explore_rel)

        # Write vortex files
        num_files = 3
        rows_per_file = 100
        _write_vortex_files(
            write_path, simple_schema, num_files, rows_per_file, filesystem=write_fs
        )

        # Add extfs properties (same config as default fs)
        props = _get_extfs_properties(default_properties, test_config)

        # 1. Explore external files (URI triggers external fs resolution)
        explore_uri = _get_explore_uri(test_config, explore_rel)
        print(f"  explore_dir URI: {explore_uri}")
        found_files, manifest_path = ExternalTable.explore(
            columns=simple_schema.names,
            format="vortex",
            base_dir=explore_meta_rel,
            explore_dir=explore_uri,
            properties=props,
        )
        assert found_files == num_files

        # 2. Read manifest and update file indices via get_file_info
        manifest = ExternalTable.read_manifest(manifest_path, properties=props)
        updated_column_groups = []
        for cg in manifest.column_groups:
            updated_files = []
            for f in cg.files:
                row_count = ExternalTable.get_file_info(
                    format="vortex",
                    file_path=f.path,
                    properties=props,
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
                ColumnGroup(columns=cg.columns, format=cg.format, files=updated_files)
            )

        # 3. Commit to base_path via transaction
        cg_to_write = ColumnGroups.from_list(updated_column_groups)
        txn = Transaction(base_rel, props)
        txn.append_files(cg_to_write)
        txn.commit()
        txn.close()

        # 4. Verify with all 3 read methods
        self._verify_import(base_rel, simple_schema, num_files * rows_per_file, props)

    def test_explore_lance_dataset(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
        test_config,
    ):
        """Explore a Lance dataset — produces final manifest with correct indices."""
        lance = pytest.importorskip("lance")

        explore_rel = f"{temp_case_path}/lance_data"
        base_rel = f"{temp_case_path}/lance_meta"

        num_batches = 3
        rows_per_batch = 100
        total_rows = num_batches * rows_per_batch

        # Write Lance dataset with multiple fragments (one per append)
        lance_uri = _get_lance_write_uri(test_config, explore_rel)
        storage_options = _get_lance_storage_options(test_config)

        for i in range(num_batches):
            batch = _generate_batch(
                simple_schema, rows_per_batch, offset=i * rows_per_batch
            )
            table = pa.Table.from_batches([batch])
            mode = "create" if i == 0 else "append"
            lance.write_dataset(
                table, lance_uri, mode=mode, storage_options=storage_options
            )

        # Explore — Lance produces final manifest with correct indices
        props = _get_extfs_properties(default_properties, test_config)
        explore_uri = _get_explore_uri(test_config, explore_rel)
        print(f"  explore_dir URI: {explore_uri}")
        found_files, manifest_path = ExternalTable.explore(
            columns=simple_schema.names,
            format="lance-table",
            base_dir=base_rel,
            explore_dir=explore_uri,
            properties=props,
        )

        assert found_files == num_batches
        assert manifest_path != ""

        # Read and verify manifest
        manifest = ExternalTable.read_manifest(manifest_path, properties=props)
        assert len(manifest.column_groups) > 0

        # Verify indices are set correctly (not -1 or INT64_MAX)
        total_manifest_rows = 0
        for cg in manifest.column_groups:
            assert cg.format == "lance-table"
            for f in cg.files:
                assert f.start_index == 0
                assert f.end_index > 0
                total_manifest_rows += f.end_index - f.start_index

        assert total_manifest_rows == total_rows

    def test_external_table_import_lance(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        default_properties,
        test_config,
    ):
        """Import external Lance dataset — no get_file_info step needed."""
        lance = pytest.importorskip("lance")

        explore_rel = f"{temp_case_path}/lance_data"
        explore_meta_rel = f"{temp_case_path}/lance_explore_meta"
        base_rel = f"{temp_case_path}/lance_imported"

        num_batches = 3
        rows_per_batch = 100

        # Write Lance dataset
        lance_uri = _get_lance_write_uri(test_config, explore_rel)
        storage_options = _get_lance_storage_options(test_config)

        for i in range(num_batches):
            batch = _generate_batch(
                simple_schema, rows_per_batch, offset=i * rows_per_batch
            )
            table = pa.Table.from_batches([batch])
            mode = "create" if i == 0 else "append"
            lance.write_dataset(
                table, lance_uri, mode=mode, storage_options=storage_options
            )

        # Add extfs properties
        props = _get_extfs_properties(default_properties, test_config)

        # 1. Explore — produces final manifest (indices already correct)
        explore_uri = _get_explore_uri(test_config, explore_rel)
        print(f"  explore_dir URI: {explore_uri}")
        found_files, manifest_path = ExternalTable.explore(
            columns=simple_schema.names,
            format="lance-table",
            base_dir=explore_meta_rel,
            explore_dir=explore_uri,
            properties=props,
        )
        assert found_files == num_batches

        # 2. Read manifest — no get_file_info needed for Lance
        manifest = ExternalTable.read_manifest(manifest_path, properties=props)

        # 3. Commit to base_path via transaction
        cg_to_write = ColumnGroups.from_list(manifest.column_groups)
        txn = Transaction(base_rel, props)
        txn.append_files(cg_to_write)
        txn.commit()
        txn.close()

        # 4. Verify with all 3 read methods
        self._verify_import(
            base_rel, simple_schema, num_batches * rows_per_batch, props
        )
