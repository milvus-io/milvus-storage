"""
File rolling tests.

Verify file rolling behavior under different configurations.
"""

import pytest
import pyarrow as pa

from milvus_storage import Reader, Writer


class TestFileRolling:
    """Test file rolling functionality."""

    def test_file_rolling_by_size(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
        count_files,
    ):
        """Verify files are rolled when reaching size threshold.

        Steps:
        1. Set file_rolling_size = 100KB
        2. Write ~500KB data (multiple batches)
        3. Verify multiple files are generated
        4. Verify all data is readable and intact
        """
        # Configure small file rolling size (100KB)
        file_rolling_size = 100 * 1024  # 100KB
        props = test_config.get_writer_properties(
            file_rolling_size=file_rolling_size,
            buffer_size=50 * 1024,  # 50KB buffer
        )

        # temp_case_path already includes test method name
        path = temp_case_path

        # Write data - each batch is ~24KB (1000 rows * 24 bytes avg per row)
        # Write 25 batches = ~600KB total, should create multiple files
        writer = Writer(path, simple_schema, props)

        total_rows = 0
        num_batches = 25
        rows_per_batch = 1000

        for i in range(num_batches):
            batch = batch_generator(rows_per_batch, offset=i * rows_per_batch)
            writer.write(batch)
            total_rows += rows_per_batch

        column_groups = writer.close()

        # Get file count from column_groups
        cg_list = column_groups.to_list()
        cg_file_count = sum(len(cg.files) for cg in cg_list)

        # Verify multiple files were created due to rolling
        assert cg_file_count >= 5, (
            f"Expected at least 5 parquet files due to rolling, got {cg_file_count}"
        )

        # Verify count_files matches column_groups file count
        parquet_files = count_files(path, "**/*.parquet")
        assert parquet_files == cg_file_count, (
            f"File count mismatch: count_files={parquet_files}, column_groups={cg_file_count}"
        )

        # Verify all data is readable
        reader = Reader(column_groups, simple_schema)
        read_rows = 0
        for batch in reader.scan():
            read_rows += batch.num_rows

        assert read_rows == total_rows, (
            f"Row count mismatch: wrote {total_rows}, read {read_rows}"
        )

        # Verify data integrity - check first and last batch values
        reader = Reader(column_groups, simple_schema)
        all_batches = list(reader.scan())

        # Combine all batches and verify id column is sequential
        all_ids = []
        for batch in all_batches:
            all_ids.extend(batch.column("id").to_pylist())

        expected_ids = list(range(total_rows))
        assert all_ids == expected_ids, "Data integrity check failed: IDs are not sequential"
