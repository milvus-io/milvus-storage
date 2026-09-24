"""Predicate behavior when scanning through the Python reader."""

import pyarrow as pa
import pytest
from milvus_storage import Reader, Writer
from milvus_storage.exceptions import FFIError


@pytest.mark.parametrize("cache_enabled", ["true", "false"])
@pytest.mark.e2e_smoke
def test_vortex_predicate_changes_after_cache_warmup(
    temp_case_path, simple_schema, batch_generator, default_properties, cache_enabled
):
    writer_properties = dict(default_properties)
    writer_properties["writer.format"] = "vortex"
    with Writer(temp_case_path, simple_schema, writer_properties) as writer:
        writer.write(batch_generator(100))
        groups = writer.close()

    read_properties = dict(writer_properties)
    read_properties["reader.metadata_cache.enable"] = cache_enabled
    with groups, Reader(groups, simple_schema, properties=read_properties) as reader:

        def ids(predicate=None):
            batches = list(reader.scan(predicate=predicate))
            return [
                value for batch in batches for value in batch.column("id").to_pylist()
            ]

        assert ids() == list(range(100))
        assert ids("id > 90") == list(range(91, 100))
        assert ids("id < 5") == list(range(5))
        assert ids() == list(range(100))


def test_invalid_predicate_keeps_column_reason_and_reader_recovers(
    temp_case_path, simple_schema, batch_generator, default_properties
):
    properties = dict(default_properties)
    properties["writer.format"] = "vortex"
    with Writer(temp_case_path, simple_schema, properties) as writer:
        writer.write(batch_generator(20))
        groups = writer.close()

    with groups, Reader(groups, simple_schema, properties=properties) as reader:
        with pytest.raises((FFIError, pa.ArrowException)) as failure:
            list(reader.scan(predicate="missing > 1"))
        assert "missing" in str(failure.value)
        batches = list(reader.scan(predicate="id < 3"))
        assert [
            value for batch in batches for value in batch.column("id").to_pylist()
        ] == [0, 1, 2]
