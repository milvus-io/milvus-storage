"""
Concurrent transaction tests.

Verify concurrent transaction behavior with multiple threads.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
from milvus_storage import Transaction, Writer


class TestConcurrentMixWorkflow:
    """Test concurrent transaction scenarios."""

    def _write_initial_data(self, path, schema, batch_generator, props, rows=1000):
        """Write initial data."""
        writer = Writer(path, schema, props)
        writer.write(batch_generator(rows, offset=0))
        return writer.close()

    def test_concurrent_appends(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Concurrent commits with no retry: exactly one succeeds per round."""
        # Initial commit
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(1000, offset=0))
        init_cg = writer.close()

        txn_init = Transaction(temp_case_path, default_properties)
        txn_init.append_files(init_cg)
        txn_init.commit()
        txn_init.close()

        num_threads = 4

        for round_idx in range(3):
            barrier = threading.Barrier(num_threads)
            successes = []
            errors = []

            def append_data(thread_id):
                # Each thread prepares its own data and transaction
                w = Writer(temp_case_path, simple_schema, default_properties)
                w.write(
                    batch_generator(
                        100, offset=(round_idx * 10 + thread_id + 1) * 10000
                    )
                )
                cg = w.close()

                txn = Transaction(temp_case_path, default_properties)
                txn.append_files(cg)

                # All threads wait here, then commit simultaneously
                barrier.wait()

                version = txn.commit()
                txn.close()
                return version

            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                futures = {pool.submit(append_data, i): i for i in range(num_threads)}
                for future in as_completed(futures):
                    try:
                        version = future.result()
                        successes.append(version)
                    except Exception:
                        errors.append(futures[future])

            assert len(successes) == 1, (
                f"Round {round_idx}: expected exactly 1 success, "
                f"got {len(successes)} successes, {len(errors)} errors"
            )

    def test_concurrent_read_write(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Concurrent reads during writes see consistent snapshots."""
        from milvus_storage import Reader
        from milvus_storage.manifest import ColumnGroups

        # Initial commit
        writer = Writer(temp_case_path, simple_schema, default_properties)
        writer.write(batch_generator(500, offset=0))
        init_cg = writer.close()

        txn_init = Transaction(temp_case_path, default_properties)
        txn_init.append_files(init_cg)
        txn_init.commit()
        txn_init.close()

        # Writer thread: sequential appends (no contention)
        write_done = threading.Event()

        def writer_loop():
            for i in range(5):
                w = Writer(temp_case_path, simple_schema, default_properties)
                w.write(batch_generator(100, offset=(i + 1) * 10000))
                cg = w.close()

                txn = Transaction(temp_case_path, default_properties)
                txn.append_files(cg)
                txn.commit()
                txn.close()
            write_done.set()

        # Reader threads: read manifest and verify data consistency
        num_readers = 4
        read_errors = []

        def reader_loop(thread_id):
            while not write_done.is_set():
                try:
                    txn = Transaction(temp_case_path, default_properties)
                    manifest = txn.get_manifest()
                    version = txn.get_read_version()
                    txn.close()

                    assert version >= 0
                    assert len(manifest.column_groups) > 0

                    # Read data and verify it's self-consistent
                    cg = ColumnGroups.from_list(manifest.column_groups)
                    reader = Reader(cg, simple_schema, properties=default_properties)
                    ids = []
                    for batch in reader.scan():
                        ids.extend(batch.column("id").to_pylist())

                    # Should have at least the initial 500 rows
                    assert len(ids) >= 500
                    # ids within each batch should be valid integers
                    assert all(isinstance(i, int) for i in ids)
                except Exception as e:
                    read_errors.append(e)
                    return

        writer_thread = threading.Thread(target=writer_loop)
        reader_threads = [
            threading.Thread(target=reader_loop, args=(i,)) for i in range(num_readers)
        ]

        writer_thread.start()
        for t in reader_threads:
            t.start()

        writer_thread.join()
        for t in reader_threads:
            t.join()

        assert len(read_errors) == 0, f"Reader errors: {read_errors}"

    def test_many_concurrent_readers(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Many concurrent readers don't interfere with each other."""
        self._write_initial_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        num_readers = 10
        errors = []

        def read_manifest(thread_id):
            txn = Transaction(temp_case_path, default_properties)
            manifest = txn.get_manifest()
            version = txn.get_read_version()
            txn.close()
            return len(manifest.column_groups), version

        with ThreadPoolExecutor(max_workers=num_readers) as pool:
            futures = [pool.submit(read_manifest, i) for i in range(num_readers)]
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    errors.append(e)

        assert len(errors) == 0, f"Reader errors: {errors}"
        assert len(results) == num_readers

        # All readers should see the same version (no writes happening)
        versions = [r[1] for r in results]
        assert len(set(versions)) == 1
