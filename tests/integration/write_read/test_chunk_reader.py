"""
Chunk reader tests.

Verify chunk-level random access and metadata.
"""

import pyarrow as pa
from milvus_storage import Reader, Writer
from milvus_storage.reader import ChunkMetadataType


class TestChunkReader:
    """Test chunk reader functionality."""

    def _write_data(
        self, path, schema, batch_generator, props, num_batches=5, rows_per_batch=1000
    ):
        """Helper to write test data and return column_groups."""
        writer = Writer(path, schema, props)
        for i in range(num_batches):
            writer.write(batch_generator(rows_per_batch, offset=i * rows_per_batch))
        return writer.close()

    def test_chunk_reader_random_access(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Access specific chunks by index."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        chunk_reader = reader.get_chunk_reader(0)

        num_chunks = chunk_reader.get_number_of_chunks()
        assert num_chunks > 0

        # Read first chunk
        chunk = chunk_reader.get_chunk(0)
        assert chunk.num_rows > 0
        assert "id" in chunk.schema.names

        # Read last chunk
        last_chunk = chunk_reader.get_chunk(num_chunks - 1)
        assert last_chunk.num_rows > 0

        chunk_reader.close()

    def test_chunk_reader_sequential_read(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Read all chunks sequentially."""
        total_written = 5000
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        chunk_reader = reader.get_chunk_reader(0)

        num_chunks = chunk_reader.get_number_of_chunks()
        total_rows = 0
        for i in range(num_chunks):
            chunk = chunk_reader.get_chunk(i)
            total_rows += chunk.num_rows

        assert total_rows == total_written
        chunk_reader.close()

    def test_chunk_reader_get_chunks(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Get multiple chunks at once."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        chunk_reader = reader.get_chunk_reader(0)

        num_chunks = chunk_reader.get_number_of_chunks()
        if num_chunks >= 2:
            indices = [0, num_chunks - 1]
            batches = chunk_reader.get_chunks(indices)
            assert len(batches) == 2
            for batch in batches:
                assert batch.num_rows > 0

        chunk_reader.close()

    def test_chunk_reader_get_chunk_rows_metadata(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Verify chunk row count metadata."""
        total_written = 5000
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        chunk_reader = reader.get_chunk_reader(0)

        metadatas = chunk_reader.get_chunk_metadatas(ChunkMetadataType.NUM_OF_ROWS)
        assert len(metadatas) > 0

        # Sum of row counts should equal total written rows
        for meta in metadatas:
            if meta.is_num_of_rows:
                total_from_meta = sum(meta.data)
                assert total_from_meta == total_written

        chunk_reader.close()

    def test_chunk_reader_get_memory_metadata(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Verify chunk estimated memory metadata."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        chunk_reader = reader.get_chunk_reader(0)

        metadatas = chunk_reader.get_chunk_metadatas(ChunkMetadataType.ESTIMATED_MEMORY)
        assert len(metadatas) > 0

        for meta in metadatas:
            if meta.is_estimated_memory:
                # Each chunk should have positive memory size
                for size in meta.data:
                    assert size > 0

        chunk_reader.close()

    def test_chunk_indices_mapping(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """Map row indices to chunk indices."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        chunk_reader = reader.get_chunk_reader(0)

        num_chunks = chunk_reader.get_number_of_chunks()

        row_indices = [0, 100, 500, 1000, 2000]
        chunk_indices = chunk_reader.get_chunk_indices(row_indices)

        assert len(chunk_indices) > 0
        # All chunk indices should be valid
        for idx in chunk_indices:
            assert 0 <= idx < num_chunks

        chunk_reader.close()

    def test_chunk_reader_context_manager(
        self,
        temp_case_path: str,
        simple_schema: pa.Schema,
        batch_generator,
        default_properties,
    ):
        """ChunkReader works as context manager."""
        column_groups = self._write_data(
            temp_case_path,
            simple_schema,
            batch_generator,
            default_properties,
        )

        reader = Reader(column_groups, simple_schema, properties=default_properties)
        with reader.get_chunk_reader(0) as chunk_reader:
            num_chunks = chunk_reader.get_number_of_chunks()
            assert num_chunks > 0
            chunk = chunk_reader.get_chunk(0)
            assert chunk.num_rows > 0
