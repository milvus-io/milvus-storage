"""
Stress test specific fixtures.

These fixtures are available to all tests under tests/stress/.

To run with smaller values for quick validation:
    pytest stress/ --stress-scale=0.01  # 1% of default values
    pytest stress/ --stress-scale=0.1   # 10% of default values

Or via environment variable:
    STRESS_SCALE_FACTOR=0.01 pytest stress/
"""

import gc
import os
import random
import tracemalloc
from typing import Callable, Dict, Generator

import pyarrow as pa
import pytest


def pytest_addoption(parser):
    """Add stress test command line options."""
    parser.addoption(
        "--stress-scale",
        action="store",
        default=None,
        type=float,
        help="Scale factor for stress tests (0.01 = 1%%, 0.1 = 10%%, 1.0 = 100%%)",
    )


def _get_scale_factor(config):
    """Get scale factor from config or environment."""
    cli_scale = config.getoption("--stress-scale", default=None)
    if cli_scale is not None:
        return cli_scale
    return float(os.environ.get("STRESS_SCALE_FACTOR", "1.0"))


def _scale_value(val, scale, min_val=1):
    """Scale a value with minimum."""
    return max(min_val, int(val * scale))


# Module-level base parameters shared between pytest_generate_tests and stress_params.
# Each entry is (base_value, min_value) for scalars,
# or a list of (base_value, min_value) tuples for parametrize lists.
_BASE_PARAMS = {
    # Parametrize lists
    "row_counts": [
        (100_000, 1000),
        (1_000_000, 10000),
        (10_000_000, 100000),
        (100_000_000, 1000000),
    ],
    "column_counts": [(100, 10), (200, 20), (500, 50), (1000, 100)],
    "string_sizes": [(10_000, 100), (100_000, 1000), (1_000_000, 10000)],
    # Common parameters
    "iterations": (10, 1),
    "rows_per_iteration": (100_000, 1000),
    "batch_size": (10_000, 100),
    "num_rounds": (3, 1),
    # Long-running test parameters
    "long_running_cycles": (1000, 10),
    "rows_per_cycle": (500, 50),
    "manifest_appends": (500, 5),
    "manifest_initial_rows": (100, 10),
    "manifest_append_rows": (50, 10),
    "create_destroy_iterations": (2000, 20),
    "create_destroy_rows": (50, 10),
    "small_transactions": (1000, 10),
    "small_txn_init_rows": (100, 10),
    # Concurrency test parameters
    "concurrent_initial_rows": (1000, 100),
    "concurrent_append_rows": (100, 10),
    # Large-scale write parameters
    "wide_table_rows": (10_000, 1000),
    "large_string_rows": (10_000, 1000),
    "vector_data_rows": (10_000, 1000),
    "default_rows_per_batch": (1_000, 100),
}


def pytest_generate_tests(metafunc):
    """Generate scaled parameters for stress tests."""
    scale = _get_scale_factor(metafunc.config)

    for param_name, base_key in [
        ("num_rows", "row_counts"),
        ("num_columns", "column_counts"),
        ("str_size", "string_sizes"),
    ]:
        if param_name in metafunc.fixturenames:
            scaled = [_scale_value(v, scale, m) for v, m in _BASE_PARAMS[base_key]]
            # Remove duplicates while preserving order
            scaled = list(dict.fromkeys(scaled))
            metafunc.parametrize(param_name, scaled)


# =============================================================================
# Properties Fixtures (override global fixture for stress tests)
# =============================================================================


# Note: We rely on the global conftest.py default_properties fixture.
# Stress tests may use larger buffer sizes via test_config.get_properties().


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
                data[name] = [
                    random.randint(-(2**31), 2**31 - 1) for _ in range(num_rows)
                ]
            elif pa.types.is_int64(dtype):
                data[name] = list(range(num_rows))  # Sequential for easier verification
            elif pa.types.is_uint8(dtype):
                data[name] = [random.randint(0, 255) for _ in range(num_rows)]
            elif pa.types.is_uint16(dtype):
                data[name] = [random.randint(0, 65535) for _ in range(num_rows)]
            elif pa.types.is_uint32(dtype):
                data[name] = [random.randint(0, 2**32 - 1) for _ in range(num_rows)]
            elif pa.types.is_uint64(dtype):
                data[name] = [random.randint(0, 2**63 - 1) for _ in range(num_rows)]
            elif pa.types.is_float32(dtype):
                data[name] = [random.random() for _ in range(num_rows)]
            elif pa.types.is_float64(dtype):
                data[name] = [random.random() * 1000 for _ in range(num_rows)]
            elif pa.types.is_boolean(dtype):
                data[name] = [random.choice([True, False]) for _ in range(num_rows)]
            elif pa.types.is_string(dtype):
                data[name] = [
                    f"str_{i}_{random.randint(0, 1000)}" for i in range(num_rows)
                ]
            elif pa.types.is_binary(dtype):
                data[name] = [os.urandom(64) for _ in range(num_rows)]
            elif pa.types.is_list(dtype):
                # Assume list of float32 for vectors
                list_size = 128  # default vector dimension
                data[name] = [
                    [random.random() for _ in range(list_size)] for _ in range(num_rows)
                ]
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
        random.seed(seed)
        rows_generated = 0

        while rows_generated < total_rows:
            current_batch_size = min(batch_size, total_rows - rows_generated)
            data = {}

            for field in schema:
                name = field.name
                dtype = field.type

                if pa.types.is_int64(dtype):
                    data[name] = list(
                        range(rows_generated, rows_generated + current_batch_size)
                    )
                elif pa.types.is_float64(dtype):
                    data[name] = [
                        random.random() * 1000 for _ in range(current_batch_size)
                    ]
                elif pa.types.is_string(dtype):
                    data[name] = [
                        f"str_{i}"
                        for i in range(
                            rows_generated, rows_generated + current_batch_size
                        )
                    ]
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
# Performance Measurement Fixtures
# =============================================================================


@pytest.fixture
def measure_time():
    """Helper fixture to measure execution time."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time

    def _measure():
        return Timer()

    return _measure


@pytest.fixture
def measure_throughput(measure_time):
    """Helper fixture to measure throughput (rows/s, MB/s)."""

    def _measure(func: Callable, data_size_bytes: int, num_rows: int) -> Dict:
        timer = measure_time()
        with timer:
            func()
        return {
            "elapsed_s": timer.elapsed,
            "rows_per_s": num_rows / timer.elapsed if timer.elapsed > 0 else 0,
            "mb_per_s": (
                (data_size_bytes / (1024 * 1024)) / timer.elapsed
                if timer.elapsed > 0
                else 0
            ),
        }

    return _measure


@pytest.fixture(scope="session")
def stress_params(request) -> Dict:
    """Scaled stress test parameters.

    All base values are defined in _BASE_PARAMS and scaled by --stress-scale.
    Use these instead of hardcoded values in tests.

    Example usage in test:
        def test_foo(stress_params):
            num_rows = stress_params["row_counts"][0]  # smallest row count
            num_iterations = stress_params["iterations"]

    To run with smaller values:
        pytest stress/ --stress-scale=0.01  # 1% of default values
        pytest stress/ --stress-scale=0.1   # 10% of default values
    """
    scale = _get_scale_factor(request.config)
    result = {}
    for key, val in _BASE_PARAMS.items():
        if isinstance(val, list):
            scaled = [_scale_value(v, scale, m) for v, m in val]
            scaled = list(dict.fromkeys(scaled))
            result[key] = scaled
        else:
            base, min_val = val
            result[key] = _scale_value(base, scale, min_val)
    return result
