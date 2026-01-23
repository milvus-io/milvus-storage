"""
Global pytest fixtures for milvus-storage integration tests.

This module provides common fixtures used across all test categories.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional

import pyarrow as pa
import pytest

from milvus_storage import ChunkReader, Filesystem, Properties, Reader, Writer

from .config import TestConfig, get_config, reload_config


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Session-scoped test configuration."""
    return get_config()


@pytest.fixture(scope="session")
def storage_backend(test_config: TestConfig) -> str:
    """Return the active storage backend name."""
    return test_config.storage_backend


@pytest.fixture(scope="session")
def data_format(test_config: TestConfig) -> str:
    """Return the data format (parquet or vortex)."""
    return test_config.format


@pytest.fixture(scope="session")
def is_local_backend(test_config: TestConfig) -> bool:
    """Check if using local filesystem backend."""
    return test_config.is_local


@pytest.fixture(scope="session")
def is_cloud_backend(test_config: TestConfig) -> bool:
    """Check if using cloud storage backend."""
    return test_config.is_cloud


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data.

    The directory is automatically cleaned up after the test.
    """
    tmpdir = tempfile.mkdtemp(prefix="milvus_storage_test_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_case_path(test_config: TestConfig, request) -> Generator[str, None, None]:
    """Create a temporary storage path based on backend configuration.

    The path includes the test method name for easier debugging.
    Path is relative to SubtreeFilesystem root (root_path for local, bucket_name for cloud).

    Steps:
    1. Generate path: base_path/test_name
    2. Delete the path if exists (clean slate)
    3. Create the directory
    """
    test_name = request.node.name
    base = test_config.base_path
    path = f"{base}/{test_name}" if base else test_name

    # Get filesystem with appropriate properties
    fs = Filesystem.get(properties=test_config.to_fs_properties())

    # Delete existing files if any (clean slate for each test)
    try:
        files = fs.list_dir(path, recursive=True)
        for f in files:
            if not f.is_dir:
                fs.delete_file(f.path)
    except Exception:
        pass  # Directory may not exist, that's fine

    # Print path info for debugging
    if test_config.is_local:
        root = test_config.root_path
        print(f"\n[temp_case_path] root_path: {root}")
    else:
        root = test_config.bucket_name
        print(f"\n[temp_case_path] bucket_name: {root}")
    print(f"[temp_case_path] path: {path}")
    full_path = f"{root}/{path}"
    print(f"[temp_case_path] full_path: {full_path}")

    yield path

    # Cleanup after test
    try:
        files = fs.list_dir(path, recursive=True)
        for f in files:
            if not f.is_dir:
                fs.delete_file(f.path)
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def unique_path(temp_case_path: str) -> Callable[[], str]:
    """Factory fixture to generate unique paths within temp_case_path."""
    counter = [0]

    def _unique_path() -> str:
        counter[0] += 1
        return f"{temp_case_path}/data_{counter[0]}_{uuid.uuid4().hex[:6]}"

    return _unique_path


# =============================================================================
# Schema Fixtures
# =============================================================================


@pytest.fixture
def simple_schema() -> pa.Schema:
    """Simple test schema with basic types."""
    return pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("value", pa.float64()),
    ])


@pytest.fixture
def wide_schema() -> pa.Schema:
    """Wide table schema with 100 columns."""
    fields = [pa.field(f"col_{i}", pa.float64()) for i in range(100)]
    return pa.schema(fields)


@pytest.fixture
def vector_schema() -> pa.Schema:
    """Schema with vector column (128-dim float32)."""
    return pa.schema([
        pa.field("id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 128)),
        pa.field("metadata", pa.string()),
    ])


@pytest.fixture
def complex_schema() -> pa.Schema:
    """Schema with complex nested types."""
    return pa.schema([
        pa.field("id", pa.int64()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("attributes", pa.struct([
            pa.field("key", pa.string()),
            pa.field("value", pa.string()),
        ])),
        pa.field("scores", pa.list_(pa.float64())),
    ])


@pytest.fixture
def all_types_schema() -> pa.Schema:
    """Schema with all supported data types."""
    return pa.schema([
        pa.field("col_int8", pa.int8()),
        pa.field("col_int16", pa.int16()),
        pa.field("col_int32", pa.int32()),
        pa.field("col_int64", pa.int64()),
        pa.field("col_uint8", pa.uint8()),
        pa.field("col_uint16", pa.uint16()),
        pa.field("col_uint32", pa.uint32()),
        pa.field("col_uint64", pa.uint64()),
        pa.field("col_float32", pa.float32()),
        pa.field("col_float64", pa.float64()),
        pa.field("col_bool", pa.bool_()),
        pa.field("col_string", pa.string()),
        pa.field("col_binary", pa.binary()),
        pa.field("col_list", pa.list_(pa.int32())),
    ])


# =============================================================================
# Data Generation Fixtures
# =============================================================================


@pytest.fixture
def simple_batch(simple_schema: pa.Schema) -> pa.RecordBatch:
    """Sample data batch with 1000 rows."""
    return pa.RecordBatch.from_pydict({
        "id": list(range(1000)),
        "name": [f"name_{i}" for i in range(1000)],
        "value": [float(i) * 0.1 for i in range(1000)],
    }, schema=simple_schema)


@pytest.fixture
def small_batch(simple_schema: pa.Schema) -> pa.RecordBatch:
    """Small data batch with 10 rows."""
    return pa.RecordBatch.from_pydict({
        "id": list(range(10)),
        "name": [f"name_{i}" for i in range(10)],
        "value": [float(i) * 0.1 for i in range(10)],
    }, schema=simple_schema)


@pytest.fixture
def batch_generator(simple_schema: pa.Schema) -> Callable[[int, int], pa.RecordBatch]:
    """Factory fixture to generate batches with specified size and offset."""
    def _generate(num_rows: int, offset: int = 0) -> pa.RecordBatch:
        return pa.RecordBatch.from_pydict({
            "id": list(range(offset, offset + num_rows)),
            "name": [f"name_{i}" for i in range(offset, offset + num_rows)],
            "value": [float(i) * 0.1 for i in range(offset, offset + num_rows)],
        }, schema=simple_schema)
    return _generate


@pytest.fixture
def vector_batch(vector_schema: pa.Schema) -> pa.RecordBatch:
    """Sample batch with 128-dim vectors."""
    import random
    random.seed(42)

    num_rows = 100
    vectors = [[random.random() for _ in range(128)] for _ in range(num_rows)]

    return pa.RecordBatch.from_pydict({
        "id": list(range(num_rows)),
        "vector": vectors,
        "metadata": [f"meta_{i}" for i in range(num_rows)],
    }, schema=vector_schema)


# =============================================================================
# Properties Fixtures
# =============================================================================


@pytest.fixture
def fs_properties(test_config: TestConfig) -> Dict[str, str]:
    """Filesystem properties based on test configuration."""
    return test_config.to_fs_properties()


@pytest.fixture
def default_writer_properties(test_config: TestConfig) -> Dict[str, str]:
    """Default writer properties with 1MB file rolling."""
    return test_config.get_writer_properties(
        file_rolling_size=1024 * 1024,  # 1MB
        buffer_size=4 * 1024 * 1024,    # 4MB
    )


@pytest.fixture
def small_rolling_properties(test_config: TestConfig) -> Dict[str, str]:
    """Writer properties with small file rolling (100KB)."""
    return test_config.get_writer_properties(
        file_rolling_size=100 * 1024,   # 100KB
        buffer_size=1024 * 1024,        # 1MB
    )


@pytest.fixture
def large_rolling_properties(test_config: TestConfig) -> Dict[str, str]:
    """Writer properties with large file rolling (100MB)."""
    return test_config.get_writer_properties(
        file_rolling_size=100 * 1024 * 1024,  # 100MB
        buffer_size=16 * 1024 * 1024,         # 16MB
    )


# =============================================================================
# Writer/Reader Factory Fixtures
# =============================================================================


@pytest.fixture
def create_writer(
    test_config: TestConfig,
    default_writer_properties: Dict[str, str],
) -> Callable[..., Writer]:
    """Factory fixture to create a Writer instance."""
    def _create(
        path: str,
        schema: pa.Schema,
        properties: Optional[Dict[str, str]] = None,
    ) -> Writer:
        props = properties if properties is not None else default_writer_properties
        return Writer(path, schema, Properties(props))

    return _create


@pytest.fixture
def create_reader(test_config: TestConfig) -> Callable[..., Reader]:
    """Factory fixture to create a Reader instance."""
    def _create(
        column_groups: str,
        schema: pa.Schema,
        columns: Optional[List[str]] = None,
        properties: Optional[Dict[str, str]] = None,
    ) -> Reader:
        props = Properties(properties) if properties else None
        return Reader(column_groups, schema, columns, props)

    return _create


# =============================================================================
# Convenience Fixtures
# =============================================================================


@pytest.fixture
def written_simple_data(
    temp_case_path: str,
    simple_schema: pa.Schema,
    simple_batch: pa.RecordBatch,
    create_writer: Callable[..., Writer],
):
    """Fixture that writes simple data and returns (path, column_groups, schema)."""
    path = f"{temp_case_path}/simple_data"
    writer = create_writer(path, simple_schema)
    writer.write(simple_batch)
    column_groups = writer.close()
    return path, column_groups, simple_schema


@pytest.fixture
def written_multi_batch_data(
    temp_case_path: str,
    simple_schema: pa.Schema,
    batch_generator: Callable[[int, int], pa.RecordBatch],
    create_writer: Callable[..., Writer],
):
    """Fixture that writes multiple batches and returns (path, column_groups, schema)."""
    path = f"{temp_case_path}/multi_batch_data"
    writer = create_writer(path, simple_schema)

    # Write 5 batches of 1000 rows each
    for i in range(5):
        batch = batch_generator(1000, i * 1000)
        writer.write(batch)

    column_groups = writer.close()
    return path, column_groups, simple_schema, 5000  # total rows


# =============================================================================
# Skip Markers
# =============================================================================


@pytest.fixture
def skip_if_local(is_local_backend: bool):
    """Skip test if running on local filesystem."""
    if is_local_backend:
        pytest.skip("Test requires cloud storage backend")


@pytest.fixture
def skip_if_cloud(is_cloud_backend: bool):
    """Skip test if running on cloud storage."""
    if is_cloud_backend:
        pytest.skip("Test requires local filesystem backend")


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "stress: marks tests as stress tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take minutes)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "cloud: marks tests that require cloud storage"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their location."""
    for item in items:
        # Mark all tests under integration/ as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark all tests under stress/ as stress tests
        if "stress" in str(item.fspath):
            item.add_marker(pytest.mark.stress)
