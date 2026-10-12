"""Ownership and public thread-pool lifecycle in Python readers."""

import gc
import json
import os
import subprocess
import sys

import pyarrow as pa
from milvus_storage import Reader, Transaction, Writer


def test_returned_batches_survive_reader_and_column_groups_close(
    temp_case_path, simple_schema, batch_generator, default_properties
):
    with Writer(temp_case_path, simple_schema, default_properties) as writer:
        writer.write(batch_generator(20))
        groups = writer.close()

    reader = Reader(groups, simple_schema, properties=default_properties)
    stream = reader.scan()
    first = next(iter(stream))
    stream.close()
    taken = reader.take([0, 19])
    reader.close()
    groups.destroy()
    gc.collect()

    assert first.column("id").to_pylist() == list(range(20))
    assert first.column("name").to_pylist() == [f"name_{i}" for i in range(20)]
    assert pa.Table.from_batches(taken).column("id").to_pylist() == [0, 19]


def test_thread_pool_can_be_resized_and_reinitialized_in_new_process(
    temp_case_path, simple_schema, batch_generator, default_properties
):
    with Writer(temp_case_path, simple_schema, default_properties) as writer:
        writer.write(batch_generator(10))
        groups = writer.close()
    with Transaction(temp_case_path, default_properties) as txn:
        txn.append_files(groups)
        txn.commit()
    groups.destroy()

    child_code = """
import json
import sys
from milvus_storage import Properties
Properties()
import pyarrow as pa
from milvus_storage import Reader, ThreadPool, Transaction
from milvus_storage.manifest import ColumnGroups

path, properties_json = sys.argv[1:]
properties = json.loads(properties_json)
schema = pa.schema([
    pa.field('id', pa.int64()),
    pa.field('name', pa.string()),
    pa.field('value', pa.float64()),
])
with Transaction(path, properties) as txn:
    manifest = txn.get_manifest()
with ColumnGroups.from_list(manifest.column_groups) as groups:
    for size in (1, 4, 2):
        ThreadPool.init(size)
        assert ThreadPool.is_initialized()
        with Reader(groups, schema, properties=properties) as reader:
            actual = pa.Table.from_batches(list(reader.scan())).column('id').to_pylist()
        assert actual == list(range(10))
        ThreadPool.release()
        assert not ThreadPool.is_initialized()
"""
    env = os.environ.copy()
    if "GCOV_PREFIX" in env:
        env["GCOV_PREFIX"] += "/thread-pool-child"
    if child_preload := env.get("STORAGE_E2E_CHILD_PRELOAD"):
        env["DYLD_INSERT_LIBRARIES"] = child_preload
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            child_code,
            temp_case_path,
            json.dumps(default_properties),
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
