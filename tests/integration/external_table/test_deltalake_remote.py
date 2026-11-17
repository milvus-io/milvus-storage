"""Delta import/read validation against the configured S3-compatible service."""

import json
import os
import struct
import zlib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.parse import quote, urlsplit

import pyarrow as pa
import pytest
from milvus_storage import Filesystem, Reader, Transaction
from milvus_storage.exceptions import FFIError
from milvus_storage.manifest import ColumnGroups

from tests.config import TestConfig as _TestConfig
from tests.integration.external_table.test_deltalake import (
    _DeletionVector,
    _DeltaTable,
    _assert_reads,
    _expected_data,
    _file_descriptors,
    _import_snapshot,
)
from tests.integration.external_table.test_iceberg_paimon import (
    _source_location,
    _source_properties,
)


@dataclass(kw_only=True)
class _RemoteDeltaTable(_DeltaTable):
    filesystem: Filesystem
    prefix: str
    source: str

    @property
    def uri(self) -> str:
        return self.source

    def file_uri(self, name: str) -> str:
        return f"{self.uri}/{quote(name)}"

    def write_file(self, name: str, data: bytes) -> None:
        super().write_file(name, data)
        self.filesystem.write_file(f"{self.prefix}/{name}", data)


def _z85_encode(data: bytes) -> str:
    alphabet = (
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
    )
    encoded = []
    for offset in range(0, len(data), 4):
        chunk = data[offset : offset + 4]
        value = int.from_bytes(chunk, "big")
        digits = ""
        for _ in range(5):
            value, digit = divmod(value, 85)
            digits = alphabet[digit] + digits
        # Match the pinned z85 crate's short-tail convention used by Kernel.
        padding = 4 - len(chunk)
        encoded.append("#" * padding + digits[padding:])
    return "".join(encoded)


def _generate_dvs(positions: Sequence[Sequence[int]]) -> Tuple[bytes, List[dict]]:
    """Encode only the small sparse Roaring array containers used by these tests."""
    container = bytearray(b"\x01" if positions else b"")
    descriptors = []
    for positions_in_file in positions:
        rows = sorted(set(positions_in_file))
        assert len(rows) <= 4096 and all(
            0 <= row < 65536 for row in rows
        ), "sparse fixture rows only"
        # Delta portable magic + RoaringTreemap's little-endian bitmap count.
        payload = struct.pack("<IQ", 0x6439D3D1, int(bool(rows)))
        if rows:
            # One high-32-bit key (0), no-run cookie, one array container with
            # high-16-bit key 0, cardinality-1, and a 16-byte bitmap header.
            payload += struct.pack("<IIIHHI", 0, 12346, 1, 0, len(rows) - 1, 16)
            payload += struct.pack(f"<{len(rows)}H", *rows)
        descriptors.append(
            {
                "offset": len(container),
                "sizeInBytes": len(payload),
                "cardinality": len(rows),
                "inline": _z85_encode(payload),
            }
        )
        # The external frame has big-endian size/CRC; inline data is payload only.
        container.extend(
            struct.pack(">I", len(payload)) + payload + struct.pack(">I", zlib.crc32(payload))
        )
    return bytes(container), descriptors


_FILE_NAMES = ("part a %2F.parquet", "nested/part 空 #?.parquet")


def _make_remote_table(
    path: Path,
    prefix: str,
    test_config: _TestConfig,
    filesystem: Filesystem,
    schema: pa.Schema,
    batch_generator,
    storage: str = "none",
    offset: int = 0,
    *,
    deleted_rows: Sequence[Sequence[int]] = ((0, 4), (1, 5)),
) -> _RemoteDeltaTable:
    table = _RemoteDeltaTable(
        path,
        schema,
        filesystem=filesystem,
        prefix=prefix,
        source=_source_location(test_config, quote(prefix)),
    )
    deletions: Dict[str, _DeletionVector] = {}
    if storage != "none":
        container, descriptors = _generate_dvs(deleted_rows)
        if storage == "uuid":
            # Z85's twenty zero digits encode the nil UUID. Both descriptors
            # reference this one container, with distinct offsets and masks.
            name = "dv/deletion_vector_00000000-0000-0000-0000-000000000000.bin"
            reference = "dv" + "0" * 20
        else:
            name = "dv/shared vectors.bin"
            reference = f"s3://{test_config.bucket_name}/{quote(prefix + '/' + name)}"
        if storage != "inline":
            table.write_file(name, container)
        for name, rows, descriptor in zip(_FILE_NAMES, deleted_rows, descriptors):
            value = {
                "storageType": {"inline": "i", "uuid": "u", "absolute": "p"}[storage],
                "pathOrInlineDv": (descriptor["inline"] if storage == "inline" else reference),
                "sizeInBytes": descriptor["sizeInBytes"],
                "cardinality": descriptor["cardinality"],
            }
            if storage != "inline":
                value["offset"] = descriptor["offset"]
            deletions[name] = _DeletionVector(value, rows)
    table.commit(
        0,
        {
            name: pa.Table.from_batches([batch_generator(6, offset=offset + index * 10)])
            for index, name in enumerate(_FILE_NAMES)
        },
        deletions=deletions,
    )
    return table


@pytest.fixture
def remote_delta_factory(tmp_path, temp_case_path, test_config, simple_schema, batch_generator):
    if not test_config.is_s3_compatible:
        pytest.skip("Remote Delta fixtures require an S3-compatible endpoint")
    with Filesystem.get(properties=test_config.get_properties()) as filesystem:

        def make(storage="none", offset=0):
            return _make_remote_table(
                tmp_path / "generated",
                f"{temp_case_path}/source 空 %2F",
                test_config,
                filesystem,
                simple_schema,
                batch_generator,
                storage,
                offset,
            )

        yield make


@contextmanager
def _persisted_reader(target, schema, properties):
    with Transaction(target, properties) as txn:
        manifest = txn.get_manifest()
    with ColumnGroups.from_list(manifest.column_groups) as groups:
        with Reader(groups, schema, properties=properties) as reader:
            yield reader


def _assert_projected_reads(target, table, properties, expected):
    for columns in (None, ["id"], ["name", "value"]):
        _assert_reads(target, table.schema, properties, expected, columns)


@pytest.mark.cloud
@pytest.mark.parametrize("storage", ["none", "inline", "uuid", "absolute"])
@pytest.mark.parametrize("metadata_cache", [False, True])
def test_remote_deltalake_import_preserves_live_values(
    remote_delta_factory,
    temp_case_path,
    default_properties,
    test_config,
    storage,
    metadata_cache,
):
    table = remote_delta_factory(storage)
    properties = {
        **_source_properties(default_properties, test_config),
        "reader.metadata_cache.enable": str(metadata_cache).lower(),
    }
    target = f"{temp_case_path}/imported"
    groups = _import_snapshot(table, target, properties)
    expected = _expected_data(table, groups, 0)
    assert expected.num_rows == (12 if storage == "none" else 8)
    # Explicit endpoint-bearing URLs plus a remote default filesystem ensure
    # persisted imports cannot silently fall back to the local producer files.
    assert all(urlsplit(file.path).scheme == "s3" for group in groups for file in group.files)
    _assert_projected_reads(target, table, properties, expected)
    with _persisted_reader(target, table.schema, properties) as reader:
        indices = [0, 3, expected.num_rows - 1]
        columns = ["id", "value"]
        actual = pa.Table.from_batches(reader.take(indices, parallelism=2, columns=columns))
        assert actual.equals(expected.select(columns).take(indices))


@pytest.mark.cloud
@pytest.mark.parametrize("metadata_cache", [False, True])
def test_remote_deltalake_import_stays_pinned(
    remote_delta_factory,
    temp_case_path,
    default_properties,
    test_config,
    batch_generator,
    metadata_cache,
):
    table = remote_delta_factory("uuid")
    properties = {
        **_source_properties(default_properties, test_config),
        "reader.metadata_cache.enable": str(metadata_cache).lower(),
    }
    old_target = f"{temp_case_path}/old"
    old_groups = _import_snapshot(table, old_target, properties, 0)
    old_expected = _expected_data(table, old_groups, 0)
    # Keep the old Reader alive; the cache-enabled case retains its snapshot too.
    with _persisted_reader(old_target, table.schema, properties) as old_reader:
        assert pa.Table.from_batches(list(old_reader.scan())).equals(old_expected)
        table.commit(
            1,
            {
                f"new-{index}.parquet": pa.Table.from_batches(
                    [batch_generator(5, offset=100 + index * 10)]
                )
                for index in range(2)
            },
            removed=_FILE_NAMES,
        )
        new_target = f"{temp_case_path}/latest"
        new_groups = _import_snapshot(table, new_target, properties)
        new_expected = _expected_data(table, new_groups, 1)
        assert new_expected.num_rows == 10
        _assert_projected_reads(new_target, table, properties, new_expected)
        assert pa.Table.from_batches(list(old_reader.scan())).equals(old_expected)
    # Reopen persisted imports after their Reader objects have been released.
    override = {**properties, "reader.exttable.snapshot_id": "1"}
    _assert_projected_reads(old_target, table, override, old_expected)
    explicit = _import_snapshot(table, f"{temp_case_path}/explicit-zero", properties, 0)
    assert sorted(_file_descriptors(explicit)) == sorted(_file_descriptors(old_groups))


@pytest.mark.cloud
def test_remote_deltalake_invalid_credentials_do_not_reuse_snapshot(
    remote_delta_factory, temp_case_path, default_properties, test_config
):
    table = remote_delta_factory("absolute")
    properties = {
        **_source_properties(default_properties, test_config),
        "reader.metadata_cache.enable": "true",
    }
    target = f"{temp_case_path}/imported"
    groups = _import_snapshot(table, target, properties)
    expected = _expected_data(table, groups, 0)
    bad = {
        **properties,
        "extfs.e2e.access_key_id": "e2e-invalid-user",
        "extfs.e2e.access_key_value": "e2e-invalid-key",
    }
    # Keep the authenticated snapshot resident while changing only source credentials.
    # The default fs remains valid so opening the collection manifest still succeeds.
    with _persisted_reader(target, table.schema, properties) as authenticated:
        assert pa.Table.from_batches(list(authenticated.scan())).equals(expected)
        with pytest.raises((FFIError, pa.ArrowException)) as failure:
            _assert_reads(target, table.schema, bad, expected)
        assert any(
            reason in str(failure.value) for reason in ("403", "AccessDenied", "InvalidAccessKey")
        )
        assert pa.Table.from_batches(list(authenticated.scan())).equals(expected)
    _assert_projected_reads(target, table, properties, expected)


@pytest.mark.cloud
@pytest.mark.parametrize("missing", ["log", "data", "dv"])
def test_remote_deltalake_missing_object_recovers(
    remote_delta_factory, temp_case_path, default_properties, test_config, missing
):
    table = remote_delta_factory("absolute")
    properties = _source_properties(default_properties, test_config)
    target = f"{temp_case_path}/imported"
    name = {
        "log": "_delta_log/00000000000000000000.json",
        "data": _FILE_NAMES[0],
        "dv": "dv/shared vectors.bin",
    }[missing]
    if missing != "log":
        groups = _import_snapshot(table, target, properties)
        expected = _expected_data(table, groups, 0)
    key = f"{table.prefix}/{name}"
    table.filesystem.delete_file(key)
    try:
        with pytest.raises((FFIError, pa.ArrowException)):
            if missing == "log":
                _import_snapshot(table, target, properties, 0)
            else:
                _assert_reads(target, table.schema, properties, expected)
    finally:
        table.filesystem.write_file(key, (table.path / name).read_bytes())
    if missing == "log":
        groups = _import_snapshot(table, target, properties, 0)
        expected = _expected_data(table, groups, 0)
    _assert_projected_reads(target, table, properties, expected)


@pytest.mark.cloud
def test_remote_deltalake_same_keys_on_two_endpoints_stay_isolated(
    remote_delta_factory,
    tmp_path,
    temp_case_path,
    test_config,
    simple_schema,
    batch_generator,
):
    secondary = os.environ.get("E2E_SECONDARY_CONFIG_FILE")
    if not secondary:
        if os.environ.get("E2E_REQUIRE_SECONDARY") == "1":
            pytest.fail("The required second remote backend is not configured")
        pytest.skip("The second remote backend is not configured")
    second_config = _TestConfig(Path(secondary))
    assert second_config.is_s3_compatible
    assert second_config.backend_config["address"] != test_config.backend_config["address"]
    first = remote_delta_factory("absolute")
    with Filesystem.get(properties=second_config.get_properties()) as second_fs:
        try:
            second = _make_remote_table(
                tmp_path / "second",
                first.prefix,
                second_config,
                second_fs,
                simple_schema,
                batch_generator,
                "absolute",
                100,
                deleted_rows=((1, 5), (0, 4)),
            )
            with ExitStack() as readers:
                for index, (table, config) in enumerate(
                    ((first, test_config), (second, second_config))
                ):
                    properties = {
                        **_source_properties(config.get_properties(), config),
                        "reader.metadata_cache.enable": "true",
                    }
                    # Match source keys to exercise Delta's filesystem identity.
                    # Destination paths must differ under the shared manifest
                    # cache's existing root/bucket-and-path uniqueness contract.
                    target = f"{temp_case_path}/imported-{index}"
                    groups = _import_snapshot(table, target, properties)
                    expected = _expected_data(table, groups, 0)
                    reader = readers.enter_context(
                        _persisted_reader(target, table.schema, properties)
                    )
                    assert pa.Table.from_batches(list(reader.scan())).equals(expected)
                    _assert_projected_reads(target, table, properties, expected)
                    if table is first:
                        first_reader, first_expected = reader, expected
                assert pa.Table.from_batches(list(first_reader.scan())).equals(first_expected)
        finally:
            for item in second_fs.list_dir(temp_case_path, recursive=True):
                if not item.is_dir:
                    second_fs.delete_file(item.path)


@dataclass
class _MemoryFilesystem:
    """Fixture-unit sink: no native filesystem, snapshot reader or service I/O."""

    files: Dict[str, bytes] = field(default_factory=dict)

    def write_file(self, name: str, data: bytes) -> None:
        self.files[name] = data


def _fixture_config(tmp_path):
    path = tmp_path / "backend.json"
    path.write_text(
        json.dumps(
            {
                "storage_backend": "minio",
                "minio": {
                    "address": "http://localhost:9000",
                    "bucket_name": "test-bucket",
                },
            }
        )
    )
    return _TestConfig(path)


def test_delta_dv_encoding_matches_kernel_reference():
    # Reference output of the existing Kernel writer at revision 913389c:
    # empty, one-row (short Z85 tail), and two-row DVs in one container.
    container, descriptors = _generate_dvs(((), (1,), (4, 0, 4)))
    assert container == bytes.fromhex(
        "01"
        "0000000cd1d339640000000000000000bf18480c"
        "00000022d1d339640100000000000000000000003a3000000100000000000000100000000100eebd85f4"
        "00000024d1d339640100000000000000000000003a300000010000000000010010000000000004009fc9660b"
    )
    assert [(item["offset"], item["sizeInBytes"], item["cardinality"]) for item in descriptors] == [
        (1, 12, 0),
        (21, 34, 1),
        (63, 36, 2),
    ]
    assert [item["inline"] for item in descriptors] == [
        "^Bg9^0000000000",
        "^Bg9^0rr910000000000iXQKl0rr91000005c8Xg##031",
        "^Bg9^0rr910000000000iXQKl0rr91000315c8Xg000c4",
    ]
    assert _generate_dvs(()) == (b"", [])
    with pytest.raises(AssertionError, match="sparse fixture"):
        _generate_dvs(((65536,),))


@pytest.mark.parametrize("storage", ["none", "inline", "uuid", "absolute"])
def test_delta_remote_fixture_publishes_log_last(tmp_path, simple_schema, batch_generator, storage):
    filesystem = _MemoryFilesystem()
    table = _make_remote_table(
        tmp_path / "producer",
        "source 空 %2F",
        _fixture_config(tmp_path),
        filesystem,
        simple_schema,
        batch_generator,
        storage,
    )
    assert table.uri == "s3://localhost:9000/test-bucket/source%20%E7%A9%BA%20%252F"
    log_key = next(reversed(filesystem.files))
    assert log_key == table.prefix + "/_delta_log/00000000000000000000.json"
    actions = [json.loads(line) for line in filesystem.files[log_key].splitlines()]
    adds = [action["add"] for action in actions if "add" in action]
    assert [add["path"] for add in adds] == [quote(name) for name in _FILE_NAMES]
    for name, add in zip(_FILE_NAMES, adds):
        assert add["size"] == len(filesystem.files[f"{table.prefix}/{name}"])
        assert json.loads(add["stats"])["numRecords"] == 6
    actual = pa.concat_tables(list(table.snapshots[0].values()))["id"].to_pylist()
    expected = (
        list(range(6)) + list(range(10, 16)) if storage == "none" else [1, 2, 3, 5, 10, 12, 13, 14]
    )
    assert actual == expected
    if storage != "none":
        assert actions[0]["protocol"]["readerFeatures"] == ["deletionVectors"]
        first, second = [add["deletionVector"] for add in adds]
        assert first["cardinality"] == second["cardinality"] == 2
        if storage == "inline":
            assert "offset" not in first and "offset" not in second
            assert len(filesystem.files) == 3
        else:
            assert first["pathOrInlineDv"] == second["pathOrInlineDv"]
            assert first["offset"] < second["offset"]
            assert len(filesystem.files) == 4
            if storage == "uuid":
                assert first["pathOrInlineDv"] == "dv" + "0" * 20


def test_delta_remote_fixture_removals_keep_dv_identity(tmp_path, simple_schema, batch_generator):
    filesystem = _MemoryFilesystem()
    table = _make_remote_table(
        tmp_path / "producer",
        "source",
        _fixture_config(tmp_path),
        filesystem,
        simple_schema,
        batch_generator,
        "uuid",
        deleted_rows=((1, 5), (0, 4)),
    )
    assert pa.concat_tables(list(table.snapshots[0].values()))["id"].to_pylist() == [
        0,
        2,
        3,
        4,
        11,
        12,
        13,
        15,
    ]
    previous = dict(table.deletions)
    table.commit(
        1,
        {
            f"new-{index}.parquet": pa.Table.from_batches(
                [batch_generator(2, offset=20 + index * 10)]
            )
            for index in range(2)
        },
        removed=_FILE_NAMES,
    )
    actions = [
        json.loads(line)
        for line in filesystem.files[
            f"{table.prefix}/_delta_log/00000000000000000001.json"
        ].splitlines()
    ]
    removals = [action["remove"] for action in actions if "remove" in action]
    assert [item["deletionVector"] for item in removals] == [
        previous[table.file_uri(name)].descriptor for name in _FILE_NAMES
    ]
    assert sum(data.num_rows for data in table.snapshots[0].values()) == 8
    assert sum(data.num_rows for data in table.snapshots[1].values()) == 4
    assert all(f"{table.prefix}/{name}" in filesystem.files for name in _FILE_NAMES)
