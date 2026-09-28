"""Talon-backed reads through the public Python filesystem and reader APIs."""

import os

import pyarrow as pa
import pytest
from milvus_storage import Filesystem, Reader, Writer
from milvus_storage.exceptions import FFIError

pytestmark = pytest.mark.talon_e2e


def _talon_properties(default_properties, test_config):
    if os.environ.get("E2E_TALON") != "1":
        pytest.skip("Talon service is enabled only in its dedicated CI lane")
    assert test_config.is_s3_compatible
    return {
        **default_properties,
        "fs.talon.enabled": "true",
        "fs.talon.coordinator": os.environ["E2E_TALON_COORDINATOR"],
        "fs.talon.block_size": "8388608",
    }


def test_talon_keeps_distinct_object_versions_separate(
    temp_case_path, default_properties, test_config
):
    talon = _talon_properties(default_properties, test_config)
    first_path = f"{temp_case_path}/version-1.bin"
    second_path = f"{temp_case_path}/version-2.bin"
    first = b"first-version" * 100
    second = b"second-version-is-longer" * 100
    with Filesystem.get(properties=default_properties) as origin:
        origin.write_file(first_path, first)
        origin.write_file(second_path, second)
        first_size, first_metadata = origin.get_file_stats(first_path)
        second_size, second_metadata = origin.get_file_stats(second_path)
        assert (first_size, second_size) == (len(first), len(second))
        assert first_metadata["ETag"] != second_metadata["ETag"]
        with Filesystem.get(properties=talon) as cached:
            assert cached.read_file_all(first_path) == first
            assert cached.read_file(second_path, 11, 29) == second[11:40]
            assert cached.read_file_all(second_path) == second
            assert cached.read_file(first_path, 7, 31) == first[7:38]


def test_talon_origin_error_keeps_path_and_recovers(
    temp_case_path, default_properties, test_config
):
    talon = _talon_properties(default_properties, test_config)
    path = f"{temp_case_path}/recover-after-not-found.bin"
    with Filesystem.get(properties=talon) as cached:
        with pytest.raises(FFIError) as failure:
            cached.read_file_all(path)
        assert "recover-after-not-found.bin" in str(failure.value)
        with Filesystem.get(properties=default_properties) as origin:
            origin.write_file(path, b"now-present")
        assert cached.read_file_all(path) == b"now-present"


def test_talon_unavailable_coordinator_falls_back_to_origin(
    temp_case_path, default_properties, test_config
):
    talon = _talon_properties(default_properties, test_config)
    talon["fs.talon.coordinator"] = "127.0.0.1:1"
    path = f"{temp_case_path}/origin-fallback.bin"
    payload = b"fallback-payload" * 100
    with Filesystem.get(properties=default_properties) as origin:
        origin.write_file(path, payload)
    with Filesystem.get(properties=talon) as cached:
        assert cached.read_file_all(path) == payload


@pytest.mark.parametrize("fmt", ["parquet", "vortex"])
def test_talon_reader_keeps_all_values(
    temp_case_path, simple_schema, batch_generator, default_properties, test_config, fmt
):
    talon = _talon_properties(default_properties, test_config)
    write_properties = {**default_properties, "writer.format": fmt}
    with Writer(temp_case_path, simple_schema, write_properties) as writer:
        writer.write(batch_generator(25, offset=10))
        groups = writer.close()
    with groups, Reader(groups, simple_schema, properties=talon) as reader:
        expected = {
            "id": list(range(10, 35)),
            "name": [f"name_{row}" for row in range(10, 35)],
            "value": [row * 0.1 for row in range(10, 35)],
        }
        assert pa.Table.from_batches(list(reader.scan())).to_pydict() == expected
        selected = [0, 12, 24]
        assert pa.Table.from_batches(reader.take(selected)).to_pydict() == {
            key: [values[index] for index in selected]
            for key, values in expected.items()
        }
        with reader.get_chunk_reader(0) as chunks:
            actual = pa.Table.from_batches(
                chunks.get_chunks(list(range(chunks.get_number_of_chunks())))
            ).to_pydict()
        assert actual == expected
