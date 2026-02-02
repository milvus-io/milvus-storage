"""
File rolling tests.

Verify file rolling behavior under different configurations.
"""

import pyarrow as pa
from milvus_storage import PropertyKeys, Reader, Writer


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
        props = test_config.get_properties(
            **{
                PropertyKeys.WRITER_FILE_ROLLING_SIZE: file_rolling_size,
                PropertyKeys.WRITER_BUFFER_SIZE: 50 * 1024,  # 50KB buffer
            }
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
        assert (
            cg_file_count >= 5
        ), f"Expected at least 5 parquet files due to rolling, got {cg_file_count}"

        # Verify count_files matches column_groups file count
        parquet_files = count_files(path, "**/*.parquet")
        assert (
            parquet_files == cg_file_count
        ), f"File count mismatch: count_files={parquet_files}, column_groups={cg_file_count}"

        # Verify all data is readable
        reader_props = test_config.get_properties()
        reader = Reader(column_groups, simple_schema, properties=reader_props)
        read_rows = 0
        for batch in reader.scan():
            read_rows += batch.num_rows

        assert (
            read_rows == total_rows
        ), f"Row count mismatch: wrote {total_rows}, read {read_rows}"

        # Verify data integrity - check first and last batch values
        reader = Reader(column_groups, simple_schema, properties=reader_props)
        all_batches = list(reader.scan())

        # Combine all batches and verify id column is sequential
        all_ids = []
        for batch in all_batches:
            all_ids.extend(batch.column("id").to_pylist())

        expected_ids = list(range(total_rows))
        assert (
            all_ids == expected_ids
        ), "Data integrity check failed: IDs are not sequential"

    def test_file_rolling_small_threshold(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
    ):
        """Small threshold rolling produces many files."""
        props = test_config.get_properties(
            **{
                PropertyKeys.WRITER_FILE_ROLLING_SIZE: 10 * 1024,  # 10KB - very small
                PropertyKeys.WRITER_BUFFER_SIZE: 5 * 1024,
            }
        )

        writer = Writer(temp_case_path, simple_schema, props)
        for i in range(10):
            writer.write(batch_generator(500, offset=i * 500))
        column_groups = writer.close()

        cg_list = column_groups.to_list()
        file_count = sum(len(cg.files) for cg in cg_list)
        # Very small threshold should produce many files
        assert (
            file_count >= 10
        ), f"Expected many files with 10KB threshold, got {file_count}"

    def test_file_rolling_large_threshold(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
    ):
        """Large threshold rolling keeps data in fewer files."""
        props = test_config.get_properties(
            **{
                PropertyKeys.WRITER_FILE_ROLLING_SIZE: 100
                * 1024
                * 1024,  # 100MB - very large
                PropertyKeys.WRITER_BUFFER_SIZE: 16 * 1024 * 1024,
            }
        )

        writer = Writer(temp_case_path, simple_schema, props)
        for i in range(10):
            writer.write(batch_generator(1000, offset=i * 1000))
        column_groups = writer.close()

        cg_list = column_groups.to_list()
        file_count = sum(len(cg.files) for cg in cg_list)
        # Large threshold: all data fits in one file
        assert (
            file_count == 1
        ), f"Expected 1 file with 100MB threshold, got {file_count}"

    def test_file_rolling_preserves_data(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
    ):
        """Data integrity is preserved across rolled files."""
        props = test_config.get_properties(
            **{
                PropertyKeys.WRITER_FILE_ROLLING_SIZE: 50 * 1024,  # 50KB
                PropertyKeys.WRITER_BUFFER_SIZE: 20 * 1024,
            }
        )

        total_rows = 5000
        writer = Writer(temp_case_path, simple_schema, props)
        for i in range(5):
            writer.write(batch_generator(1000, offset=i * 1000))
        column_groups = writer.close()

        # Read all data back
        reader_props = test_config.get_properties()
        reader = Reader(column_groups, simple_schema, properties=reader_props)
        all_batches = list(reader.scan())

        all_ids = []
        all_names = []
        all_values = []
        for batch in all_batches:
            all_ids.extend(batch.column("id").to_pylist())
            all_names.extend(batch.column("name").to_pylist())
            all_values.extend(batch.column("value").to_pylist())

        assert len(all_ids) == total_rows
        assert all_ids == list(range(total_rows))
        assert all_names == [f"name_{i}" for i in range(total_rows)]
        for i in range(total_rows):
            assert abs(all_values[i] - i * 0.1) < 1e-6

    def test_file_rolling_count_verification(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
        count_files,
    ):
        """Rolling file count should be consistent between filesystem and column groups."""
        props = test_config.get_properties(
            **{
                PropertyKeys.WRITER_FILE_ROLLING_SIZE: 80 * 1024,
                PropertyKeys.WRITER_BUFFER_SIZE: 30 * 1024,
            }
        )

        writer = Writer(temp_case_path, simple_schema, props)
        for i in range(20):
            writer.write(batch_generator(500, offset=i * 500))
        column_groups = writer.close()

        # Count from column_groups metadata
        cg_list = column_groups.to_list()
        cg_file_count = sum(len(cg.files) for cg in cg_list)

        # Count from filesystem
        fs_file_count = count_files(temp_case_path, "**/*.parquet")

        assert (
            cg_file_count == fs_file_count
        ), f"File count mismatch: column_groups={cg_file_count}, filesystem={fs_file_count}"
        assert cg_file_count > 1, "Expected file rolling to occur"

    def test_file_rolling_memory_pressure(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
    ):
        """Rolling under memory pressure with small buffer size."""
        props = test_config.get_properties(
            **{
                PropertyKeys.WRITER_FILE_ROLLING_SIZE: 50 * 1024,
                PropertyKeys.WRITER_BUFFER_SIZE: 10 * 1024,  # Very small buffer
            }
        )

        writer = Writer(temp_case_path, simple_schema, props)
        total_rows = 0
        for i in range(15):
            writer.write(batch_generator(1000, offset=i * 1000))
            total_rows += 1000

        column_groups = writer.close()

        # Verify all data readable
        reader_props = test_config.get_properties()
        reader = Reader(column_groups, simple_schema, properties=reader_props)
        read_rows = sum(b.num_rows for b in reader.scan())
        assert read_rows == total_rows
