"""
Integration test specific fixtures.

These fixtures are available to all tests under tests/integration/.
"""

import fnmatch
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import pyarrow as pa
import pytest
from milvus_storage import Filesystem, Reader

# =============================================================================
# Concurrency Fixtures
# =============================================================================


@pytest.fixture
def thread_pool() -> ThreadPoolExecutor:
    """Thread pool for concurrent tests."""
    pool = ThreadPoolExecutor(max_workers=10)
    yield pool
    pool.shutdown(wait=True)


@pytest.fixture
def run_concurrent():
    """Helper fixture to run functions concurrently and collect results."""

    def _run_concurrent(
        func: Callable,
        args_list: List[Tuple],
        num_threads: int = 10,
    ) -> List:
        """Run func with different args in parallel threads.

        Args:
            func: Function to run
            args_list: List of argument tuples for each call
            num_threads: Maximum number of concurrent threads

        Returns:
            List of results (or exceptions) from each call
        """
        results = [None] * len(args_list)
        errors = [None] * len(args_list)

        def worker(index: int, args: Tuple):
            try:
                results[index] = func(*args)
            except Exception as e:
                errors[index] = e

        threads = []
        for i, args in enumerate(args_list):
            t = threading.Thread(target=worker, args=(i, args))
            threads.append(t)

        # Start threads in batches
        batch_size = num_threads
        for i in range(0, len(threads), batch_size):
            batch = threads[i : i + batch_size]
            for t in batch:
                t.start()
            for t in batch:
                t.join()

        # Raise first error if any
        for e in errors:
            if e is not None:
                raise e

        return results

    return _run_concurrent


# =============================================================================
# Data Verification Fixtures
# =============================================================================


@pytest.fixture
def verify_data_integrity():
    """Helper fixture to verify data integrity after operations."""

    def _verify(
        reader: Reader,
        expected_schema: pa.Schema,
        expected_row_count: int,
        expected_columns: Optional[List[str]] = None,
    ):
        """Verify reader returns expected data.

        Args:
            reader: Reader instance
            expected_schema: Expected schema
            expected_row_count: Expected total row count
            expected_columns: Expected column names (optional)
        """
        total_rows = 0
        batches = list(reader.scan())

        for batch in batches:
            total_rows += batch.num_rows

            # Verify schema compatibility
            if expected_columns:
                actual_columns = set(batch.schema.names)
                for col in expected_columns:
                    assert col in actual_columns, f"Missing column: {col}"

        assert (
            total_rows == expected_row_count
        ), f"Row count mismatch: expected {expected_row_count}, got {total_rows}"

    return _verify


@pytest.fixture
def compare_batches():
    """Helper fixture to compare two record batches."""

    def _compare(batch1: pa.RecordBatch, batch2: pa.RecordBatch) -> bool:
        """Compare two record batches for equality.

        Args:
            batch1: First batch
            batch2: Second batch

        Returns:
            True if batches are equal
        """
        if batch1.num_rows != batch2.num_rows:
            return False

        if batch1.num_columns != batch2.num_columns:
            return False

        if batch1.schema != batch2.schema:
            return False

        for i in range(batch1.num_columns):
            if not batch1.column(i).equals(batch2.column(i)):
                return False

        return True

    return _compare


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
        """Measure throughput of a function.

        Args:
            func: Function to measure
            data_size_bytes: Size of data in bytes
            num_rows: Number of rows processed

        Returns:
            Dict with elapsed_s, rows_per_s, mb_per_s
        """
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


# =============================================================================
# File System Helpers
# =============================================================================


@pytest.fixture
def count_files(test_config):
    """Helper fixture to count files matching a pattern using Filesystem API."""

    def _count_files(path: str, pattern: str) -> int:
        """Count files matching pattern in a directory.

        Args:
            path: Directory path (relative to SubtreeFilesystem root)
            pattern: Glob pattern (e.g., "*.parquet" or "**/*.parquet")

        Returns:
            Number of matching files
        """
        props = test_config.get_properties()
        fs = Filesystem.get(properties=props)

        try:
            # List all files recursively
            files = fs.list_dir(path, recursive=True)
        except Exception:
            return 0

        # Extract the filename pattern (handle **/ prefix)
        if pattern.startswith("**/"):
            file_pattern = pattern[3:]
        elif pattern.startswith("**"):
            file_pattern = pattern[2:]
        else:
            file_pattern = pattern

        # Count files matching the pattern
        count = 0
        for f in files:
            if not f.is_dir:
                # Match against filename only
                filename = f.path.split("/")[-1]
                if fnmatch.fnmatch(filename, file_pattern):
                    count += 1
        return count

    return _count_files


# =============================================================================
# Error Injection Fixtures (Placeholder for Phase 2)
# =============================================================================


@pytest.fixture
def fault_injection_available() -> bool:
    """Check if fault injection (libfiu) is available."""
    # TODO: Implement in Phase 2
    return False


@pytest.fixture
def skip_without_fault_injection(fault_injection_available: bool):
    """Skip test if fault injection is not available."""
    if not fault_injection_available:
        pytest.skip("Test requires fault injection (libfiu)")
