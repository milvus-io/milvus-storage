"""Python E2E reads of independently produced Iceberg and Paimon tables."""

import os
import re
import subprocess
from pathlib import Path

import pyarrow as pa
import pytest
from milvus_storage import ExternalTable, FFIError, Filesystem, Reader, Transaction
from milvus_storage.manifest import ColumnGroups


def _make_table(table_type, path, rows, scenario=None, deletes=None, properties=None):
    tool = Path(__file__).resolve().parents[3] / "cpp/build/Release/tools/loon"
    assert tool.is_file(), f"Fixture producer is missing: {tool}"
    command = [
        str(tool),
        "demo-table",
        "--type",
        table_type,
        "--path",
        str(path),
        "--rows",
        str(rows),
    ]
    if scenario:
        command.extend(["--scenario", scenario])
    if deletes:
        command.extend(["--deletes", ",".join(map(str, deletes))])
    for key, value in (properties or {}).items():
        if key.startswith(("fs.", "extfs.")):
            command.extend(["--prop", f"{key}={value}"])
    env = os.environ.copy()
    preload = env.get("STORAGE_E2E_CHILD_PRELOAD")
    if preload:
        env["DYLD_INSERT_LIBRARIES"] = preload
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        pytest.fail(result.stderr, pytrace=False)
    return result.stdout


def _source_location(test_config, relative):
    assert test_config.is_s3_compatible
    address = test_config.backend_config["address"].split("://", 1)[-1]
    return f"s3://{address}/{test_config.bucket_name}/{relative}"


def _source_properties(default_properties, test_config):
    if test_config.is_local:
        return {**default_properties, "fs.root_path": "/"}
    return {
        **default_properties,
        **{
            f"extfs.e2e.{key[3:]}": value
            for key, value in default_properties.items()
            if key.startswith("fs.")
        },
    }


def _read_import(source, fmt, target, schema, properties):
    count, manifest_path = ExternalTable.explore(
        columns=schema.names,
        format=fmt,
        base_dir=f"{target}_explore",
        explore_dir=source,
        properties=properties,
    )
    assert count > 0
    manifest = ExternalTable.read_manifest(manifest_path, properties=properties)
    assert manifest.column_groups
    with ColumnGroups.from_list(manifest.column_groups) as groups:
        with Transaction(target, properties) as txn:
            txn.append_files(groups)
            txn.commit()
    with Transaction(target, properties) as txn:
        stored = txn.get_manifest()
    with ColumnGroups.from_list(stored.column_groups) as groups:
        with Reader(groups, schema, properties=properties) as reader:
            scanned = pa.Table.from_batches(list(reader.scan())).to_pydict()
            indices = [0, len(scanned["id"]) // 2, len(scanned["id"]) - 1]
            taken = pa.Table.from_batches(reader.take(indices)).to_pydict()
            with reader.get_chunk_reader(0) as chunks_reader:
                chunks = chunks_reader.get_chunks(
                    list(range(chunks_reader.get_number_of_chunks()))
                )
            chunked = pa.Table.from_batches(chunks).to_pydict()
    assert taken == {
        key: [values[index] for index in indices] for key, values in scanned.items()
    }
    assert chunked == scanned
    return scanned


@pytest.mark.parametrize("deletes", [[], [3, 7, 15]])
def test_iceberg_import_keeps_logical_rows(
    temp_case_path, test_config, simple_schema, default_properties, deletes
):
    if not test_config.is_local:
        pytest.skip("Iceberg fixture producer uses a local table")
    table = Path(test_config.root_path) / temp_case_path / "iceberg"
    output = _make_table("iceberg", table, 20, deletes=deletes)
    metadata = next(
        line.split(":", 1)[1].strip()
        for line in output.splitlines()
        if line.strip().startswith("metadata_location:")
    )
    properties = {**default_properties, "fs.root_path": "/"}
    actual = _read_import(
        metadata,
        "iceberg-table",
        str(table) + "_imported",
        simple_schema,
        properties,
    )
    expected_ids = [row for row in range(20) if row not in deletes]
    assert actual == {
        "id": expected_ids,
        "name": [f"row_{row}" for row in expected_ids],
        "value": [row * 1.5 for row in expected_ids],
    }


@pytest.mark.parametrize(
    "scenario,deletes,expected_ids",
    [
        ("append-only", [], list(range(12))),
        ("deletion-vector", [1, 5, 9], [0, 2, 3, 4, 6, 7, 8, 10, 11]),
        ("merge-on-read", [], list(range(12))),
    ],
)
def test_paimon_import_keeps_logical_rows(
    temp_case_path,
    test_config,
    simple_schema,
    default_properties,
    scenario,
    deletes,
    expected_ids,
):
    if not test_config.is_local:
        pytest.skip("Paimon fixture producer uses a local table")
    table = Path(test_config.root_path) / temp_case_path / "paimon"
    _make_table("paimon", table, 12, scenario=scenario, deletes=deletes)
    properties = {**default_properties, "fs.root_path": "/", "paimon.scan_mode": "auto"}
    actual = _read_import(
        str(table), "paimon-table", str(table) + "_imported", simple_schema, properties
    )
    multipliers = [
        10 if scenario == "merge-on-read" and row >= 6 else 1 for row in expected_ids
    ]
    assert actual == {
        "id": expected_ids,
        "name": [
            f"row_{row * factor}" for row, factor in zip(expected_ids, multipliers)
        ],
        "value": [row * 1.5 * factor for row, factor in zip(expected_ids, multipliers)],
    }


def test_paimon_snapshot_pinning_keeps_old_values(
    temp_case_path, test_config, simple_schema, default_properties
):
    if not test_config.is_local:
        pytest.skip("Paimon fixture producer uses a local table")
    table = Path(test_config.root_path) / temp_case_path / "paimon_snapshots"
    output = _make_table("paimon", table, 10, scenario="merge-on-read")
    snapshot_ids = [
        int(value)
        for value in re.search(r"snapshots:\s*\[([^]]+)\]", output).group(1).split(",")
    ]
    assert len(snapshot_ids) == 2

    properties = {**default_properties, "fs.root_path": "/", "paimon.scan_mode": "auto"}
    first = _read_import(
        str(table),
        "paimon-table",
        str(table) + "_first",
        simple_schema,
        {**properties, "reader.exttable.snapshot_id": str(snapshot_ids[0])},
    )
    latest = _read_import(
        str(table), "paimon-table", str(table) + "_latest", simple_schema, properties
    )
    assert first == {
        "id": list(range(10)),
        "name": [f"row_{row}" for row in range(10)],
        "value": [row * 1.5 for row in range(10)],
    }
    assert latest["id"] == first["id"]
    assert latest["name"][:5] == first["name"][:5]
    assert latest["name"][5:] == [f"row_{row * 10}" for row in range(5, 10)]

    with pytest.raises(FFIError, match="required metadata.*was not found"):
        ExternalTable.explore(
            columns=simple_schema.names,
            format="paimon-table",
            base_dir=str(table) + "_missing",
            explore_dir=str(table),
            properties={
                **properties,
                "reader.exttable.snapshot_id": str(snapshot_ids[-1] + 1000),
            },
        )


def _check_remote_table_import(
    temp_case_path, test_config, simple_schema, default_properties, table_type
):
    if not test_config.is_s3_compatible:
        pytest.skip("Remote table fixture needs an S3-compatible endpoint")
    location = _source_location(test_config, f"{temp_case_path}/{table_type}")
    properties = _source_properties(default_properties, test_config)
    _make_table(table_type, location, 9, properties=properties)
    metadata_key = (
        "metadata/v1.metadata.json" if table_type == "iceberg" else "schema/schema-0"
    )
    with Filesystem.get(properties=default_properties) as origin:
        assert origin.read_file_all(f"{temp_case_path}/{table_type}/{metadata_key}")
    source = (
        location + "/metadata/v1.metadata.json" if table_type == "iceberg" else location
    )
    actual = _read_import(
        source,
        f"{table_type}-table",
        f"{temp_case_path}/{table_type}_imported",
        simple_schema,
        properties,
    )
    assert actual == {
        "id": list(range(9)),
        "name": [f"row_{row}" for row in range(9)],
        "value": [row * 1.5 for row in range(9)],
    }


def test_remote_iceberg_table_import(
    temp_case_path, test_config, simple_schema, default_properties
):
    _check_remote_table_import(
        temp_case_path, test_config, simple_schema, default_properties, "iceberg"
    )


def test_remote_paimon_table_import(
    temp_case_path, test_config, simple_schema, default_properties
):
    _check_remote_table_import(
        temp_case_path, test_config, simple_schema, default_properties, "paimon"
    )
