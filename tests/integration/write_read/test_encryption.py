"""End-to-end Parquet encryption through the Python API."""

import base64

import pyarrow as pa
import pytest
from milvus_storage import Reader, Transaction, Writer
from milvus_storage.exceptions import FFIError
from milvus_storage.manifest import ColumnGroups


def _read_ids(reader, method):
    if method == "scan":
        batches = list(reader.scan())
    elif method == "take":
        batches = reader.take([0, 2, 4])
    else:
        with reader.get_chunk_reader(0) as chunk_reader:
            batches = [chunk_reader.get_chunk(0)]
    return [row for batch in batches for row in batch.column("id").to_pylist()]


@pytest.mark.parametrize("method", ["scan", "take", "chunk"])
@pytest.mark.parametrize("key_length", [16, 24, 32])
@pytest.mark.parametrize("algorithm", ["AES_GCM_V1", "AES_GCM_CTR_V1"])
@pytest.mark.e2e_smoke
def test_binary_encryption_key_roundtrip(
    temp_case_path, default_properties, method, key_length, algorithm
):
    schema = pa.schema([pa.field("id", pa.int64())])
    key = bytes((i * 17 + 1) % 256 for i in range(key_length))
    properties = dict(default_properties)
    properties.update(
        {
            "writer.enc.enable": "true",
            "writer.enc.key": base64.b64encode(key).decode("ascii"),
            "writer.enc.meta": "e2e-key",
            "writer.enc.algorithm": algorithm,
        }
    )
    with Writer(temp_case_path, schema, properties) as writer:
        writer.write(
            pa.record_batch([pa.array(range(5), type=pa.int64())], schema=schema)
        )
        column_groups = writer.close()

    with column_groups, Reader(
        column_groups, schema, properties=default_properties
    ) as reader:
        reader.set_key_retriever(
            lambda metadata: key if metadata == "e2e-key" else None
        )
        ids = _read_ids(reader, method)
    assert ids == ([0, 2, 4] if method == "take" else list(range(5)))


@pytest.mark.parametrize("cache_enabled", ["true", "false"])
def test_encryption_keys_are_selected_per_file(
    temp_case_path, default_properties, cache_enabled
):
    schema = pa.schema([pa.field("id", pa.int64())])
    keys = {
        "first": bytes([0, 255, 128, 1]) * 4,
        "second": bytes([1, 128, 255, 0]) * 4,
    }
    for part, (metadata, key) in enumerate(keys.items()):
        properties = dict(default_properties)
        properties.update(
            {
                "writer.enc.enable": "true",
                "writer.enc.key": base64.b64encode(key).decode("ascii"),
                "writer.enc.meta": metadata,
            }
        )
        with Writer(temp_case_path, schema, properties) as writer:
            writer.write(
                pa.record_batch(
                    [pa.array(range(part * 5, part * 5 + 5))], schema=schema
                )
            )
            written = writer.close()
        with Transaction(temp_case_path, properties) as txn:
            txn.append_files(written)
            txn.commit()
        written.destroy()

    read_properties = dict(default_properties)
    read_properties["reader.metadata_cache.enable"] = cache_enabled
    with Transaction(temp_case_path, read_properties) as txn:
        manifest = txn.get_manifest()

    seen = []

    def lookup(metadata):
        seen.append(metadata)
        return keys.get(metadata)

    with ColumnGroups.from_list(manifest.column_groups) as groups:
        with Reader(groups, schema, properties=read_properties) as reader:
            reader.set_key_retriever(lookup)
            assert _read_ids(reader, "scan") == list(range(10))
            assert [
                row
                for batch in reader.take([0, 4, 5, 9])
                for row in batch.column("id").to_pylist()
            ] == [0, 4, 5, 9]
    assert set(seen) == set(keys)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="D01: key retriever callback drops the original exception",
)
@pytest.mark.parametrize("method", ["scan", "take", "chunk"])
@pytest.mark.e2e_smoke
def test_key_retriever_failure_retains_original_reason(
    temp_case_path, default_properties, method
):
    schema = pa.schema([pa.field("id", pa.int64())])
    key = bytes([0, 255, 128, 1]) * 4
    properties = dict(default_properties)
    properties.update(
        {
            "writer.enc.enable": "true",
            "writer.enc.key": base64.b64encode(key).decode("ascii"),
            "writer.enc.meta": "e2e-key",
        }
    )
    with Writer(temp_case_path, schema, properties) as writer:
        writer.write(
            pa.record_batch([pa.array(range(5), type=pa.int64())], schema=schema)
        )
        column_groups = writer.close()

    marker = "kms-e2e-original-cause"

    def failing_retriever(metadata):
        raise RuntimeError(f"{marker}: {metadata}")

    with column_groups, Reader(
        column_groups, schema, properties=default_properties
    ) as reader:
        reader.set_key_retriever(failing_retriever)
        with pytest.raises(FFIError) as failure:
            _read_ids(reader, method)

    error = failure.value
    seen = set()
    messages = []
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        messages.append(str(error))
        error = error.__cause__ or error.__context__
    assert marker in "\n".join(messages)
