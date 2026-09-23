"""
Chunk reader tests.

Verify chunk-level random access and metadata.
"""

import numpy as np
import pyarrow as pa
import pytest
from milvus_storage import Reader, Writer
from milvus_storage.exceptions import InvalidArgumentError
from milvus_storage.manifest import ColumnGroup, ColumnGroups
from milvus_storage.reader import ChunkMetadataType


class TestChunkReader:
    """Test chunk reader functionality."""

    def _write_data(
        self, path, schema, batch_generator, props, num_batches=5, rows_per_batch=1000,
        offset=0
    ):
        """Helper to write test data and return column_groups."""
        writer = Writer(path, schema, props)
        for i in range(num_batches):
            writer.write(batch_generator(rows_per_batch, offset=offset + i * rows_per_batch))
        return writer.close()

    def _two_file_groups(self, path, schema, batch_generator, props):
        first = self._write_data(
            f"{path}/first", schema, batch_generator, props,
            num_batches=1, rows_per_batch=10,
        )
        second = self._write_data(
            f"{path}/second", schema, batch_generator, props,
            num_batches=1, rows_per_batch=10, offset=10,
        )
        files = [first.to_list()[0].files[0], second.to_list()[0].files[0]]
        groups = ColumnGroups.from_list([ColumnGroup(schema.names, "parquet", files)])
        first.destroy()
        second.destroy()
        return groups

    @pytest.mark.parametrize(
        "indices",
        [
            pytest.param([], id="list-empty"),
            pytest.param(np.array([], dtype=np.int64), id="numpy-empty"),
            pytest.param([0], id="list-zero"),
            pytest.param(
                np.array([0], dtype=np.int64),
                id="numpy-zero",
            ),
            pytest.param([0, 1], id="list-two-chunks"),
            pytest.param(
                np.array([0, 0, 1], dtype=np.int64)[::2],
                id="numpy-strided-two-chunks",
            ),
            pytest.param(
                np.array([0, 1], dtype=np.int64),
                id="numpy-two-chunks",
            ),
        ],
    )
    @pytest.mark.e2e_smoke
    def test_get_chunks_accepts_numpy_indices(
        self, temp_case_path, simple_schema, batch_generator, default_properties, indices
    ):
        groups = self._two_file_groups(
            temp_case_path, simple_schema, batch_generator, default_properties
        )
        with groups, Reader(
            groups, simple_schema, properties=default_properties
        ) as reader:
            with reader.get_chunk_reader(0) as chunk_reader:
                if chunk_reader.get_number_of_chunks() < 2:
                    pytest.fail("The two-file fixture did not create two chunks")
                batches = chunk_reader.get_chunks(indices)
        assert len(batches) == len(indices)
        assert [batch.column("id").to_pylist() for batch in batches] == [
            list(range(10 * index, 10 * (index + 1))) for index in indices
        ]

    def test_chunk_metadata_maps_real_file_boundaries(
        self, temp_case_path, simple_schema, batch_generator, default_properties
    ):
        groups = self._two_file_groups(
            temp_case_path, simple_schema, batch_generator, default_properties
        )
        with groups, Reader(groups, simple_schema, properties=default_properties) as reader:
            with reader.get_chunk_reader(0) as chunk_reader:
                assert chunk_reader.get_number_of_chunks() == 2
                assert chunk_reader.get_chunk_indices([0, 2, 9, 10, 19, 0]).tolist() == [0, 1]

                strided_rows = np.arange(20, dtype=np.int64)[::2]
                assert chunk_reader.get_chunk_indices(strided_rows).tolist() == [0, 1]
                with pytest.raises(InvalidArgumentError, match="1-dimensional"):
                    chunk_reader.get_chunk_indices(np.array([[0, 10]], dtype=np.int64))

                metadata = chunk_reader.get_chunk_metadatas(ChunkMetadataType.ALL)
                row_counts = [item.data for item in metadata if item.is_num_of_rows]
                assert row_counts == [[10, 10]]
                assert [batch.column("id").to_pylist()
                        for batch in chunk_reader.get_chunks([0, 1])] == [
                    list(range(10)), list(range(10, 20))
                ]

    @pytest.mark.parametrize("parallelism", [1, 2, 4])
    def test_parallel_chunk_reads_preserve_values(
        self, temp_case_path, simple_schema, batch_generator, default_properties,
        parallelism
    ):
        groups = self._two_file_groups(
            temp_case_path, simple_schema, batch_generator, default_properties
        )
        with groups, Reader(groups, simple_schema, properties=default_properties) as reader:
            with reader.get_chunk_reader(0) as chunk_reader:
                batches = chunk_reader.get_chunks([0, 1], parallelism=parallelism)
        assert [batch.column("id").to_pylist() for batch in batches] == [
            list(range(10)), list(range(10, 20))
        ]
        assert [batch.column("name").to_pylist() for batch in batches] == [
            [f"name_{i}" for i in range(10)],
            [f"name_{i}" for i in range(10, 20)],
        ]

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
