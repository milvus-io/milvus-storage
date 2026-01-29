"""
Compression tests.

Verify different compression codecs and settings.
"""

import pyarrow as pa
from milvus_storage import PropertyKeys, Reader, Writer


class TestCompression:
    """Test compression functionality."""

    def _write_and_read(self, path, schema, batch_generator, props, num_batches=5):
        """Helper to write data and read it back, returning (column_groups, total_rows)."""
        writer = Writer(path, schema, props)
        total_rows = 0
        for i in range(num_batches):
            batch = batch_generator(1000, offset=i * 1000)
            writer.write(batch)
            total_rows += 1000
        column_groups = writer.close()
        return column_groups, total_rows

    def _verify_roundtrip(self, column_groups, schema, total_rows, test_config):
        """Verify data can be read back correctly."""
        reader_props = test_config.get_properties()
        reader = Reader(column_groups, schema, properties=reader_props)
        read_rows = 0
        all_ids = []
        for batch in reader.scan():
            read_rows += batch.num_rows
            all_ids.extend(batch.column("id").to_pylist())
        assert read_rows == total_rows
        assert all_ids == list(range(total_rows))

    def test_compression_snappy(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
    ):
        """Snappy compression codec."""
        props = test_config.get_properties(
            **{
                PropertyKeys.WRITER_COMPRESSION: "snappy",
                PropertyKeys.WRITER_COMPRESSION_LEVEL: "-1",
            }
        )
        column_groups, total_rows = self._write_and_read(
            temp_case_path,
            simple_schema,
            batch_generator,
            props,
        )
        self._verify_roundtrip(column_groups, simple_schema, total_rows, test_config)

    def test_compression_zstd(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
    ):
        """Zstd compression codec."""
        props = test_config.get_properties(**{PropertyKeys.WRITER_COMPRESSION: "zstd"})
        column_groups, total_rows = self._write_and_read(
            temp_case_path,
            simple_schema,
            batch_generator,
            props,
        )
        self._verify_roundtrip(column_groups, simple_schema, total_rows, test_config)

    def test_compression_lz4(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
    ):
        """LZ4 compression codec."""
        props = test_config.get_properties(**{PropertyKeys.WRITER_COMPRESSION: "lz4"})
        column_groups, total_rows = self._write_and_read(
            temp_case_path,
            simple_schema,
            batch_generator,
            props,
        )
        self._verify_roundtrip(column_groups, simple_schema, total_rows, test_config)

    def test_no_compression(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
    ):
        """Uncompressed data."""
        props = test_config.get_properties(
            **{PropertyKeys.WRITER_COMPRESSION: "uncompressed"}
        )
        column_groups, total_rows = self._write_and_read(
            temp_case_path,
            simple_schema,
            batch_generator,
            props,
        )
        self._verify_roundtrip(column_groups, simple_schema, total_rows, test_config)

    def test_compression_gzip(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        test_config,
    ):
        """Gzip compression codec."""
        props = test_config.get_properties(**{PropertyKeys.WRITER_COMPRESSION: "gzip"})
        column_groups, total_rows = self._write_and_read(
            temp_case_path,
            simple_schema,
            batch_generator,
            props,
        )
        self._verify_roundtrip(column_groups, simple_schema, total_rows, test_config)

    def test_compression_ratio_verification(
        self,
        temp_case_path: str,
        test_config,
        count_files,
    ):
        """Verify compression reduces file size vs uncompressed."""
        # Create highly compressible data (many repeated values)
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("repeated", pa.string()),
            ]
        )

        def write_with_codec(path, codec):
            props = test_config.get_properties(
                **{PropertyKeys.WRITER_COMPRESSION: codec}
            )
            writer = Writer(path, schema, props)
            batch = pa.RecordBatch.from_pydict(
                {
                    "id": list(range(5000)),
                    "repeated": ["same_value_repeated"] * 5000,
                },
                schema=schema,
            )
            writer.write(batch)
            return writer.close()

        cg_none = write_with_codec(f"{temp_case_path}/none", "uncompressed")
        cg_zstd = write_with_codec(f"{temp_case_path}/zstd", "zstd")

        # Both should have data and be readable
        reader_props = test_config.get_properties()
        reader_none = Reader(cg_none, schema, properties=reader_props)
        reader_zstd = Reader(cg_zstd, schema, properties=reader_props)

        rows_none = sum(b.num_rows for b in reader_none.scan())
        rows_zstd = sum(b.num_rows for b in reader_zstd.scan())

        assert rows_none == 5000
        assert rows_zstd == 5000
