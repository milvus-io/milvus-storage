"""
Stress test specific fixtures.

These fixtures are available to all tests under tests/stress/.
"""

import gc
import os
import random
import sys
import tracemalloc
from typing import Callable, Dict, Generator, List

import pyarrow as pa
import pytest


# =============================================================================
# Memory Monitoring Fixtures
# =============================================================================


@pytest.fixture
def memory_tracker() -> Generator[Dict, None, None]:
    """Track memory usage during test execution.

    Usage:
        def test_memory(memory_tracker):
            # do work
            assert memory_tracker["peak_mb"] < 1000
    """
    tracemalloc.start()
    gc.collect()

    tracker = {
        "start_mb": 0,
        "peak_mb": 0,
        "current_mb": 0,
    }

    current, peak = tracemalloc.get_traced_memory()
    tracker["start_mb"] = current / (1024 * 1024)

    yield tracker

    current, peak = tracemalloc.get_traced_memory()
    tracker["current_mb"] = current / (1024 * 1024)
    tracker["peak_mb"] = peak / (1024 * 1024)
    tracemalloc.stop()


@pytest.fixture
def assert_no_memory_leak(memory_tracker):
    """Assert that memory doesn't grow significantly during test."""
    def _assert(max_growth_mb: float = 100):
        """Assert memory growth is within limits.

        Args:
            max_growth_mb: Maximum allowed memory growth in MB
        """
        growth = memory_tracker["current_mb"] - memory_tracker["start_mb"]
        assert growth < max_growth_mb, (
            f"Memory leak detected: grew by {growth:.2f}MB "
            f"(start: {memory_tracker['start_mb']:.2f}MB, "
            f"current: {memory_tracker['current_mb']:.2f}MB)"
        )

    return _assert


# =============================================================================
# Large Data Generation Fixtures
# =============================================================================


@pytest.fixture
def large_batch_generator():
    """Generate large batches for stress testing."""
    def _generate(
        schema: pa.Schema,
        num_rows: int,
        seed: int = 42,
    ) -> pa.RecordBatch:
        """Generate a large batch with specified number of rows.

        Args:
            schema: Arrow schema
            num_rows: Number of rows to generate
            seed: Random seed for reproducibility

        Returns:
            RecordBatch with random data
        """
        random.seed(seed)
        data = {}

        for field in schema:
            name = field.name
            dtype = field.type

            if pa.types.is_int8(dtype):
                data[name] = [random.randint(-128, 127) for _ in range(num_rows)]
            elif pa.types.is_int16(dtype):
                data[name] = [random.randint(-32768, 32767) for _ in range(num_rows)]
            elif pa.types.is_int32(dtype):
                data[name] = [random.randint(-2**31, 2**31-1) for _ in range(num_rows)]
            elif pa.types.is_int64(dtype):
                data[name] = list(range(num_rows))  # Sequential for easier verification
            elif pa.types.is_uint8(dtype):
                data[name] = [random.randint(0, 255) for _ in range(num_rows)]
            elif pa.types.is_uint16(dtype):
                data[name] = [random.randint(0, 65535) for _ in range(num_rows)]
            elif pa.types.is_uint32(dtype):
                data[name] = [random.randint(0, 2**32-1) for _ in range(num_rows)]
            elif pa.types.is_uint64(dtype):
                data[name] = [random.randint(0, 2**63-1) for _ in range(num_rows)]
            elif pa.types.is_float32(dtype):
                data[name] = [random.random() for _ in range(num_rows)]
            elif pa.types.is_float64(dtype):
                data[name] = [random.random() * 1000 for _ in range(num_rows)]
            elif pa.types.is_boolean(dtype):
                data[name] = [random.choice([True, False]) for _ in range(num_rows)]
            elif pa.types.is_string(dtype):
                data[name] = [f"str_{i}_{random.randint(0, 1000)}" for i in range(num_rows)]
            elif pa.types.is_binary(dtype):
                data[name] = [os.urandom(64) for _ in range(num_rows)]
            elif pa.types.is_list(dtype):
                # Assume list of float32 for vectors
                list_size = 128  # default vector dimension
                data[name] = [[random.random() for _ in range(list_size)] for _ in range(num_rows)]
            else:
                # Default to None for unsupported types
                data[name] = [None] * num_rows

        return pa.RecordBatch.from_pydict(data, schema=schema)

    return _generate


@pytest.fixture
def streaming_batch_generator():
    """Generate batches in a streaming fashion for memory efficiency."""
    def _generate(
        schema: pa.Schema,
        total_rows: int,
        batch_size: int = 10000,
        seed: int = 42,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Generate batches in a streaming fashion.

        Args:
            schema: Arrow schema
            total_rows: Total number of rows to generate
            batch_size: Number of rows per batch
            seed: Random seed

        Yields:
            RecordBatch objects
        """
        from tests.stress.conftest import _generate_batch_data

        random.seed(seed)
        rows_generated = 0

        while rows_generated < total_rows:
            current_batch_size = min(batch_size, total_rows - rows_generated)
            data = {}

            for field in schema:
                name = field.name
                dtype = field.type

                if pa.types.is_int64(dtype):
                    data[name] = list(range(rows_generated, rows_generated + current_batch_size))
                elif pa.types.is_float64(dtype):
                    data[name] = [random.random() * 1000 for _ in range(current_batch_size)]
                elif pa.types.is_string(dtype):
                    data[name] = [f"str_{i}" for i in range(rows_generated, rows_generated + current_batch_size)]
                else:
                    data[name] = [None] * current_batch_size

            yield pa.RecordBatch.from_pydict(data, schema=schema)
            rows_generated += current_batch_size

    return _generate


# =============================================================================
# Resource Monitoring Fixtures
# =============================================================================


@pytest.fixture
def file_handle_counter():
    """Count open file handles."""
    try:
        import psutil
        process = psutil.Process()

        def _count() -> int:
            return len(process.open_files())

        return _count
    except ImportError:
        pytest.skip("psutil required for file handle counting")


@pytest.fixture
def assert_no_file_handle_leak(file_handle_counter):
    """Assert that file handles are properly closed."""
    initial_count = file_handle_counter()

    def _assert(max_growth: int = 5):
        """Assert file handle count hasn't grown significantly.

        Args:
            max_growth: Maximum allowed growth in file handles
        """
        gc.collect()  # Ensure finalizers run
        current_count = file_handle_counter()
        growth = current_count - initial_count
        assert growth <= max_growth, (
            f"File handle leak detected: grew by {growth} "
            f"(initial: {initial_count}, current: {current_count})"
        )

    return _assert


# =============================================================================
# Stress Test Configuration
# =============================================================================


@pytest.fixture(scope="session")
def stress_config() -> Dict:
    """Configuration for stress tests."""
    return {
        "max_rows": 100_000_000,      # 100 million rows
        "max_data_size_gb": 10,       # 10GB
        "max_threads": 100,           # 100 concurrent threads
        "default_batch_size": 10000,  # 10K rows per batch
        "long_running_seconds": 3600, # 1 hour for long-running tests
    }


@pytest.fixture
def scale_factor(stress_config) -> float:
    """Scale factor for stress tests (0.1 = 10%, 1.0 = 100%).

    Can be set via STRESS_SCALE_FACTOR environment variable.
    Useful for quick validation runs.
    """
    return float(os.environ.get("STRESS_SCALE_FACTOR", "1.0"))


@pytest.fixture
def scaled_row_count(stress_config, scale_factor) -> int:
    """Scaled row count based on scale_factor."""
    return int(stress_config["max_rows"] * scale_factor)


@pytest.fixture
def scaled_thread_count(stress_config, scale_factor) -> int:
    """Scaled thread count based on scale_factor."""
    return max(1, int(stress_config["max_threads"] * scale_factor))
