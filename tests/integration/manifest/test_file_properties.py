"""File properties across Python, FFI, and manifest persistence."""

import json
import os
import subprocess
import sys

import pyarrow as pa
import pytest
from milvus_storage import Reader, Transaction, Writer
from milvus_storage.manifest import ColumnGroups


@pytest.mark.e2e_smoke
def test_file_properties_survive_commit_and_reopen(
    temp_case_path, simple_schema, batch_generator, default_properties
):
    batch = batch_generator(12)
    with Writer(temp_case_path, simple_schema, default_properties) as writer:
        writer.write(batch)
        written_groups = writer.close()

    written = written_groups.to_list()
    files = written[0].files
    assert len(files) == 1
    assert int(files[0].properties["file_size"]) > 0
    assert int(files[0].properties["footer_size"]) > 0
    files[0].properties["test.custom"] = "retained"
    expected_properties = dict(files[0].properties)

    with ColumnGroups.from_list(written) as groups:
        assert groups.to_list()[0].files[0].properties == expected_properties
        with Transaction(temp_case_path, default_properties) as txn:
            txn.append_files(groups)
            txn.commit()
    written_groups.destroy()

    with Transaction(temp_case_path, default_properties) as txn:
        manifest = txn.get_manifest()
    assert len(manifest.column_groups) == 1
    assert manifest.column_groups[0].files[0].properties == expected_properties

    with ColumnGroups.from_list(manifest.column_groups) as groups:
        with Reader(groups, simple_schema, properties=default_properties) as reader:
            actual = pa.Table.from_batches(list(reader.scan())).to_pydict()
    assert actual == batch.to_pydict()


def test_committed_properties_and_rows_survive_new_process(
    temp_case_path, simple_schema, batch_generator, default_properties, test_config
):
    if not test_config.is_local:
        pytest.skip("Subprocess fixture uses an isolated local filesystem")

    with Writer(temp_case_path, simple_schema, default_properties) as writer:
        writer.write(batch_generator(7))
        written_groups = writer.close()
    written = written_groups.to_list()
    written[0].files[0].properties["test.custom"] = "from-parent"
    with ColumnGroups.from_list(written) as groups:
        with Transaction(temp_case_path, default_properties) as txn:
            txn.append_files(groups)
            txn.commit()
    written_groups.destroy()

    child_code = """
import json
import sys
from milvus_storage._ffi import get_library
get_library()
import pyarrow as pa
from milvus_storage import Reader, Transaction
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
assert len(manifest.column_groups) == 1
file = manifest.column_groups[0].files[0]
assert file.properties['test.custom'] == 'from-parent'
assert int(file.properties['file_size']) > 0
with ColumnGroups.from_list(manifest.column_groups) as groups:
    with Reader(groups, schema, properties=properties) as reader:
        actual = pa.Table.from_batches(list(reader.scan())).to_pydict()
assert actual == {
    'id': list(range(7)),
    'name': [f'name_{i}' for i in range(7)],
    'value': [i * 0.1 for i in range(7)],
}
"""
    env = os.environ.copy()
    if "GCOV_PREFIX" in env:
        env["GCOV_PREFIX"] += "/manifest-child"
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
