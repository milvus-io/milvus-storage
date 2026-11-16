"""Delta snapshot descriptors through Python, FFI and manifest persistence."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import quote, unquote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from milvus_storage import ColumnGroup, ExternalTable, Reader, Transaction
from milvus_storage.manifest import ColumnGroups


@dataclass
class _DeltaTable:
    """Small generated fixtures, without a Delta writer runtime or binary files."""

    path: Path
    schema: pa.Schema
    snapshots: Dict[int, Dict[str, pa.Table]] = field(default_factory=dict)

    def commit(self, version: int, files: Dict[str, pa.Table], removed: Sequence[str] = ()) -> None:
        log = self.path / "_delta_log"
        log.mkdir(parents=True, exist_ok=True)
        actions = []
        active = self.snapshots.get(version - 1, {}).copy()
        if version == 0:
            types = {pa.int64(): "long", pa.string(): "string", pa.float64(): "double"}
            schema = {
                "type": "struct",
                "fields": [
                    {
                        "name": column.name,
                        "type": types[column.type],
                        "nullable": column.nullable,
                        "metadata": {},
                    }
                    for column in self.schema
                ],
            }
            actions.extend(
                [
                    {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
                    {
                        "metaData": {
                            "id": "00000000-0000-0000-0000-000000000001",
                            "format": {"provider": "parquet", "options": {}},
                            "schemaString": json.dumps(schema),
                            "partitionColumns": [],
                            "configuration": {},
                            "createdTime": 0,
                        }
                    },
                ]
            )
        for name in removed:
            del active[(self.path / name).as_uri()]
            actions.append(
                {"remove": {"path": quote(name), "deletionTimestamp": 0, "dataChange": True}}
            )
        for name, data in files.items():
            path = self.path / name
            pq.write_table(data, path, row_group_size=2)
            # Retain historical files and expectations when a later snapshot removes its Add.
            active[path.as_uri()] = data
            actions.append(
                {
                    "add": {
                        "path": quote(name),
                        "partitionValues": {},
                        "size": path.stat().st_size,
                        "modificationTime": 0,
                        "dataChange": True,
                        "stats": json.dumps({"numRecords": data.num_rows}),
                    }
                }
            )
        with (log / f"{version:020d}.json").open("x") as output:
            for action in actions:
                output.write(json.dumps(action) + "\n")
        self.snapshots[version] = active


@pytest.fixture
def delta_table(tmp_path, simple_schema, batch_generator, test_config):
    if not test_config.is_local:
        pytest.skip("Delta persistence fixtures use local storage")
    table = _DeltaTable(tmp_path / "source table", simple_schema)
    table.commit(
        0,
        {
            "part a.parquet": pa.Table.from_batches([batch_generator(5)]),
            "part-b.parquet": pa.Table.from_batches([batch_generator(3, offset=10)]),
        },
    )
    return table


def _file_descriptors(groups: List[ColumnGroup]) -> list:
    return [
        [file.path, file.start_index, file.end_index, file.properties]
        for group in groups
        for file in group.files
    ]


def _import_snapshot(
    table: _DeltaTable, target: Path, properties: Dict[str, str], version: int = -1
) -> List[ColumnGroup]:
    count, manifest_path = ExternalTable.explore(
        columns=table.schema.names,
        format="deltalake-table",
        base_dir=str(target) + "_explore",
        explore_dir=str(table.path),
        properties={**properties, "reader.exttable.snapshot_id": str(version)},
    )
    explored = ExternalTable.read_manifest(manifest_path, properties=properties)
    assert len(explored.column_groups) == 1
    group = explored.column_groups[0]
    assert group.format == "deltalake-table"
    assert group.columns == table.schema.names
    assert count == len(group.files) == 2
    descriptors = _file_descriptors(explored.column_groups)

    with ColumnGroups.from_list(explored.column_groups) as groups:
        assert _file_descriptors(groups.to_list()) == descriptors
        with Transaction(str(target), properties) as txn:
            txn.append_files(groups)
            txn.commit()
    with Transaction(str(target), properties) as txn:
        stored = txn.get_manifest()
    assert len(stored.column_groups) == 1
    assert stored.column_groups[0].format == group.format
    assert stored.column_groups[0].columns == group.columns
    assert _file_descriptors(stored.column_groups) == descriptors
    return stored.column_groups


def _expected_data(table: _DeltaTable, groups: List[ColumnGroup], version: int) -> pa.Table:
    files = table.snapshots[version]
    assert {file.path for group in groups for file in group.files} == set(files)
    for group in groups:
        for file in group.files:
            expected = files[file.path]
            assert file.start_index == 0
            assert file.end_index == expected.num_rows
            assert set(file.properties) == {"metadata", "file_size"}
            assert (
                int(file.properties["file_size"])
                == Path(unquote(urlsplit(file.path).path)).stat().st_size
            )
            assert file.metadata.decode("utf-8") == file.properties["metadata"]
            assert json.loads(file.properties["metadata"]) == {
                "version": 1,
                "table_uri": table.path.as_uri() + "/",
                "snapshot_version": version,
            }
    # Delta file order is unspecified; persisted manifest order defines row indices.
    return pa.concat_tables([files[file.path] for group in groups for file in group.files])


def _assert_reads(
    target: Path, schema: pa.Schema, properties: Dict[str, str], expected: pa.Table
) -> None:
    with Transaction(str(target), properties) as txn:
        manifest = txn.get_manifest()
    with ColumnGroups.from_list(manifest.column_groups) as groups:
        with Reader(groups, schema, properties=properties) as reader:
            scanned = pa.Table.from_batches(list(reader.scan()))
            assert scanned.equals(expected)
            indices = [0, expected.num_rows // 2, expected.num_rows - 1]
            taken = pa.Table.from_batches(reader.take(indices, parallelism=2))
            assert taken.equals(expected.take(indices))
            with reader.get_chunk_reader(0) as chunks:
                batches = chunks.get_chunks(
                    list(range(chunks.get_number_of_chunks())), parallelism=2
                )
            assert pa.Table.from_batches(batches).equals(expected)


def _advance_snapshot(table: _DeltaTable, batch_generator) -> None:
    table.commit(
        1,
        {
            "replacement-a.parquet": pa.Table.from_batches([batch_generator(4, offset=100)]),
            "replacement-b.parquet": pa.Table.from_batches([batch_generator(6, offset=200)]),
        },
        removed=("part a.parquet", "part-b.parquet"),
    )


@pytest.mark.e2e_smoke
@pytest.mark.parametrize("metadata_cache", [False, True])
def test_deltalake_import_preserves_metadata_and_reads(
    delta_table, tmp_path, default_properties, metadata_cache
):
    properties = {
        **default_properties,
        "fs.root_path": "/",
        "reader.metadata_cache.enable": str(metadata_cache).lower(),
    }
    target = tmp_path / "imported"
    groups = _import_snapshot(delta_table, target, properties)
    expected = _expected_data(delta_table, groups, 0)
    assert expected.num_rows == 8
    _assert_reads(target, delta_table.schema, properties, expected)


@pytest.mark.parametrize("snapshot_id", [-1, 0])
def test_deltalake_import_stays_pinned_after_source_replacement(
    delta_table, tmp_path, default_properties, batch_generator, snapshot_id
):
    properties = {**default_properties, "fs.root_path": "/"}
    old_target = tmp_path / "old_import"
    old_groups = _import_snapshot(delta_table, old_target, properties, snapshot_id)
    old_expected = _expected_data(delta_table, old_groups, 0)
    _assert_reads(old_target, delta_table.schema, properties, old_expected)

    _advance_snapshot(delta_table, batch_generator)
    new_target = tmp_path / "new_import"
    new_groups = _import_snapshot(delta_table, new_target, properties)
    new_expected = _expected_data(delta_table, new_groups, 1)
    assert new_expected.num_rows == 10

    # Reader configuration must not override a persisted descriptor's exact version.
    read_properties = {**properties, "reader.exttable.snapshot_id": "1"}
    _assert_reads(old_target, delta_table.schema, read_properties, old_expected)
    _assert_reads(new_target, delta_table.schema, read_properties, new_expected)
    pinned_groups = _import_snapshot(delta_table, tmp_path / "explicit_zero", properties, 0)
    assert sorted(_file_descriptors(pinned_groups)) == sorted(_file_descriptors(old_groups))


def test_deltalake_persisted_snapshots_reopen_in_new_process(
    delta_table, tmp_path, default_properties, batch_generator
):
    properties = {**default_properties, "fs.root_path": "/"}
    old_target = tmp_path / "old_import"
    old_groups = _import_snapshot(delta_table, old_target, properties, 0)
    _advance_snapshot(delta_table, batch_generator)
    new_target = tmp_path / "new_import"
    new_groups = _import_snapshot(delta_table, new_target, properties)

    child_code = """
import json
import sys
from milvus_storage._ffi import get_library
get_library()
import pyarrow as pa
from milvus_storage import Reader, Transaction
from milvus_storage.manifest import ColumnGroups

expected = json.load(sys.stdin)
schema = pa.schema([
    pa.field(name, pa.type_for_alias(kind), nullable=nullable)
    for name, kind, nullable in expected['schema']
])
with Transaction(expected['target'], expected['properties']) as txn:
    manifest = txn.get_manifest()
assert len(manifest.column_groups) == 1
group = manifest.column_groups[0]
assert group.format == 'deltalake-table'
assert group.columns == schema.names
assert [
    [file.path, file.start_index, file.end_index, file.properties] for file in group.files
] == expected['files']
with ColumnGroups.from_list(manifest.column_groups) as groups:
    with Reader(groups, schema, properties=expected['properties']) as reader:
        actual = pa.Table.from_batches(list(reader.scan()))
assert actual.schema.equals(schema)
assert actual.to_pydict() == expected['data']
"""
    env = os.environ.copy()
    if "GCOV_PREFIX" in env:
        env["GCOV_PREFIX"] += "/delta-child"
    if child_preload := env.get("STORAGE_E2E_CHILD_PRELOAD"):
        env["DYLD_INSERT_LIBRARIES"] = child_preload

    # No parent-process reader is opened: both children must resolve cold snapshots.
    for target, groups, version in ((old_target, old_groups, 0), (new_target, new_groups, 1)):
        expected = _expected_data(delta_table, groups, version)
        payload = {
            "target": str(target),
            "properties": {**properties, "reader.exttable.snapshot_id": "1"},
            "schema": [
                (field.name, str(field.type), field.nullable) for field in delta_table.schema
            ],
            "files": _file_descriptors(groups),
            "data": expected.to_pydict(),
        }
        child = subprocess.run(
            [sys.executable, "-c", child_code],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr
