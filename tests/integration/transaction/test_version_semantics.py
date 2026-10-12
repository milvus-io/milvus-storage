"""Committed versions and conflict strategies through the Python API."""

import pyarrow as pa
import pytest
from milvus_storage import Reader, Transaction, Writer
from milvus_storage.exceptions import FFIError
from milvus_storage.manifest import ColumnGroups
from milvus_storage.transaction import ResolveStrategy


def _write_one(path, schema, batch_generator, props, row_id):
    with Writer(path, schema, props) as writer:
        writer.write(batch_generator(1, offset=row_id))
        return writer.close()


def _read_ids(path, schema, props, version=-1):
    with Transaction(path, props, read_version=version) as txn:
        manifest = txn.get_manifest()
    with ColumnGroups.from_list(manifest.column_groups) as groups:
        with Reader(groups, schema, properties=props) as reader:
            return pa.Table.from_batches(list(reader.scan())).column("id").to_pylist()


@pytest.mark.e2e_smoke
def test_read_versions_and_conflict_strategies(
    temp_case_path, simple_schema, batch_generator, default_properties
):
    initial = _write_one(
        temp_case_path, simple_schema, batch_generator, default_properties, 0
    )
    with Transaction(temp_case_path, default_properties) as txn:
        txn.append_files(initial)
        first_version = txn.commit()

    stale = Transaction(
        temp_case_path,
        default_properties,
        read_version=first_version,
        resolve_id=ResolveStrategy.FAIL,
    )
    later = _write_one(
        temp_case_path, simple_schema, batch_generator, default_properties, 1
    )
    stale_file = _write_one(
        temp_case_path, simple_schema, batch_generator, default_properties, 2
    )
    stale.append_files(stale_file)
    with Transaction(temp_case_path, default_properties) as txn:
        txn.append_files(later)
        second_version = txn.commit()
    with pytest.raises(FFIError, match="concurrent transaction") as conflict:
        stale.commit()
    assert f"read_version={first_version}" in str(conflict.value)
    assert f"latest_version={second_version}" in str(conflict.value)
    stale.close()

    assert _read_ids(temp_case_path, simple_schema, default_properties) == [0, 1]
    overwrite = Transaction(
        temp_case_path,
        default_properties,
        read_version=first_version,
        resolve_id=ResolveStrategy.OVERWRITE,
    )
    overwrite.append_files(stale_file)
    third_version = overwrite.commit()
    overwrite.close()

    assert third_version > second_version > first_version
    assert _read_ids(
        temp_case_path, simple_schema, default_properties, first_version
    ) == [0]
    assert _read_ids(
        temp_case_path, simple_schema, default_properties, second_version
    ) == [0, 1]
    assert _read_ids(
        temp_case_path, simple_schema, default_properties, third_version
    ) == [0, 2]


def test_closing_transaction_without_commit_keeps_last_version(
    temp_case_path, simple_schema, batch_generator, default_properties
):
    initial = _write_one(
        temp_case_path, simple_schema, batch_generator, default_properties, 0
    )
    with Transaction(temp_case_path, default_properties) as txn:
        txn.append_files(initial)
        committed_version = txn.commit()

    uncommitted = _write_one(
        temp_case_path, simple_schema, batch_generator, default_properties, 1
    )
    with Transaction(temp_case_path, default_properties) as txn:
        txn.append_files(uncommitted)

    with Transaction(temp_case_path, default_properties) as txn:
        assert txn.get_read_version() == committed_version
    assert _read_ids(temp_case_path, simple_schema, default_properties) == [0]
