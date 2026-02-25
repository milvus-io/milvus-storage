"""
High concurrency stress tests.

Verify stability and correctness under high thread counts.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pytest
from milvus_storage import Reader, Transaction, Writer


@pytest.mark.stress
class TestHighConcurrency:
    """Test high concurrency scenarios."""

    def _commit_initial_data(self, path, schema, batch_generator, props, rows=1000):
        """Write initial data and commit via transaction."""
        writer = Writer(path, schema, props)
        writer.write(batch_generator(rows, offset=0))
        cg = writer.close()

        txn = Transaction(path, props)
        txn.append_files(cg)
        txn.commit()
        txn.close()

    @pytest.mark.parametrize(
        "num_writers,num_readers",
        [(3, 5), (5, 10)],
    )
    def test_concurrent_write_read_transactions(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        stress_params,
        num_writers,
        num_readers,
    ):
        """N write transactions compete (only 1 succeeds), M read transactions all succeed.

        Each read transaction performs:
        1. full scan
        2. chunk reader with get_chunk / get_chunks on random chunks
        3. take with random indices
        """
        import random
        import threading

        from milvus_storage.manifest import ColumnGroups

        initial_rows = stress_params["concurrent_initial_rows"]
        append_rows = stress_params["concurrent_append_rows"]
        self._commit_initial_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
            rows=initial_rows,
        )

        num_rounds = stress_params["num_rounds"]
        expected_rows = initial_rows

        for round_idx in range(num_rounds):
            barrier = threading.Barrier(num_writers + num_readers)
            write_results = []  # (thread_id, success: bool)
            read_errors = []

            def write_task(thread_id):
                # Write new data to the same base path
                w = Writer(temp_case_path, simple_schema, default_properties)
                w.write(
                    batch_generator(
                        append_rows,
                        offset=10000 + round_idx * 1000 + thread_id * append_rows,
                    )
                )
                new_cg = w.close()

                # Create transaction before barrier to lock the same read_version
                txn = Transaction(temp_case_path, default_properties)
                txn.append_files(new_cg)

                barrier.wait()
                try:
                    txn.commit()
                    txn.close()
                    return (thread_id, True)
                except Exception:
                    txn.close()
                    return (thread_id, False)

            def read_task(thread_id):
                barrier.wait()
                txn = Transaction(temp_case_path, default_properties)
                manifest = txn.get_manifest()
                txn.close()

                cg = ColumnGroups.from_list(manifest.column_groups)
                reader = Reader(cg, simple_schema, properties=default_properties)

                # 1. full scan
                scan_rows = sum(b.num_rows for b in reader.scan())
                assert scan_rows >= expected_rows, (
                    f"Round {round_idx} reader {thread_id}: "
                    f"scan got {scan_rows}, expected >= {expected_rows}"
                )

                # 2. chunk reader: get_chunk + get_chunks
                chunk_reader = reader.get_chunk_reader(0)
                num_chunks = chunk_reader.get_number_of_chunks()
                assert num_chunks > 0
                rng = random.Random(thread_id + round_idx * 100)

                random_idx = rng.randint(0, num_chunks - 1)
                single_batch = chunk_reader.get_chunk(random_idx)
                assert single_batch.num_rows > 0

                if num_chunks >= 2:
                    sample_size = min(num_chunks, rng.randint(2, 4))
                    random_indices = rng.sample(range(num_chunks), sample_size)
                    multi_batches = chunk_reader.get_chunks(random_indices)
                    assert len(multi_batches) == sample_size
                    for b in multi_batches:
                        assert b.num_rows > 0

                chunk_reader.close()

                # 3. take with random indices
                take_indices = sorted(rng.sample(range(scan_rows), min(10, scan_rows)))
                taken = reader.take(take_indices)
                taken_rows = sum(b.num_rows for b in taken)
                assert taken_rows == len(take_indices), (
                    f"Round {round_idx} reader {thread_id}: "
                    f"take got {taken_rows}, expected {len(take_indices)}"
                )

                reader.close()
                return True

            with ThreadPoolExecutor(max_workers=num_writers + num_readers) as pool:
                w_futures = [pool.submit(write_task, i) for i in range(num_writers)]
                r_futures = [pool.submit(read_task, i) for i in range(num_readers)]

                for f in as_completed(w_futures):
                    write_results.append(f.result())

                for f in as_completed(r_futures):
                    try:
                        f.result()
                    except Exception as e:
                        read_errors.append(e)

            # Exactly one writer should succeed per round (fail strategy)
            success_count = sum(1 for _, ok in write_results if ok)
            assert success_count == 1, (
                f"Round {round_idx}: expected exactly 1 writer to succeed, "
                f"got {success_count}, results={write_results}"
            )

            # All readers should succeed
            assert (
                len(read_errors) == 0
            ), f"Round {round_idx}: reader errors: {read_errors}"

            # Update expected rows for next round (one writer succeeded)
            expected_rows += append_rows

    def test_resource_cleanup_under_load(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
        assert_no_file_handle_leak,
        assert_no_memory_leak,
        stress_params,
    ):
        """Write via transactions and read with full coverage to test cleanup."""
        import gc
        import random

        from milvus_storage.manifest import ColumnGroups

        num_iterations = stress_params["iterations"]
        rows_per_iter = stress_params["rows_per_iteration"]
        batch_size = stress_params["batch_size"]
        rng = random.Random(42)

        for i in range(num_iterations):
            # Write large data via transaction
            writer = Writer(temp_case_path, simple_schema, default_properties)
            for b in range(rows_per_iter // batch_size):
                writer.write(
                    batch_generator(
                        batch_size, offset=i * rows_per_iter + b * batch_size
                    )
                )
            cg = writer.close()

            txn = Transaction(temp_case_path, default_properties)
            if i == 0:
                txn.append_files(cg)
            else:
                txn.append_files(cg)
            txn.commit()
            txn.close()

            # Read via manifest
            txn = Transaction(temp_case_path, default_properties)
            manifest = txn.get_manifest()
            txn.close()

            read_cg = ColumnGroups.from_list(manifest.column_groups)
            reader = Reader(read_cg, simple_schema, properties=default_properties)

            # 1. full scan
            scan_rows = sum(b.num_rows for b in reader.scan())
            assert scan_rows >= rows_per_iter

            # 2. chunk reader: get_chunk + get_chunks
            chunk_reader = reader.get_chunk_reader(0)
            num_chunks = chunk_reader.get_number_of_chunks()
            assert num_chunks > 0

            single = chunk_reader.get_chunk(0)
            assert single.num_rows > 0

            if num_chunks >= 2:
                sample = min(num_chunks, rng.randint(2, 4))
                indices = rng.sample(range(num_chunks), sample)
                multi = chunk_reader.get_chunks(indices)
                assert len(multi) == sample
                for b in multi:
                    assert b.num_rows > 0
            chunk_reader.close()

            # 3. take with random indices
            take_indices = sorted(rng.sample(range(scan_rows), min(10, scan_rows)))
            taken = reader.take(take_indices)
            assert sum(b.num_rows for b in taken) == len(take_indices)

            reader.close()

        gc.collect()
        assert_no_file_handle_leak(max_growth=10)
        assert_no_memory_leak(max_growth_mb=10)
