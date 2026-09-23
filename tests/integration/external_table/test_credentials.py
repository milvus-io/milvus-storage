"""Remote storage identities must not share another endpoint's data."""

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from milvus_storage import Filesystem, Reader
from milvus_storage.exceptions import FFIError
from milvus_storage.manifest import ColumnGroup, ColumnGroupFile, ColumnGroups

from tests.config import TestConfig as _TestConfig


@pytest.mark.cloud
def test_invalid_credentials_do_not_reuse_authenticated_filesystem(
    temp_case_path, default_properties, test_config
):
    if not test_config.is_s3_compatible:
        pytest.skip("Credential isolation requires an S3-compatible backend")
    path = f"{temp_case_path}/protected.bin"
    bad_properties = dict(default_properties)
    bad_properties["fs.access_key_id"] = "e2e-invalid-user"
    bad_properties["fs.access_key_value"] = "e2e-invalid-key"

    with Filesystem.get(properties=default_properties) as authenticated:
        authenticated.create_dir(temp_case_path)
        authenticated.write_file(path, b"protected-data")
        with Filesystem.get(properties=bad_properties) as unauthenticated:
            with pytest.raises(FFIError) as failure:
                unauthenticated.read_file(path, 0, len(b"protected-data"))
        assert authenticated.read_file_all(path) == b"protected-data"

    message = str(failure.value)
    assert "protected.bin" in message
    assert any(
        reason in message for reason in ("403", "AccessDenied", "InvalidAccessKey")
    )


@pytest.mark.cloud
def test_same_path_on_two_remote_identities_keeps_distinct_contents(
    temp_case_path, default_properties, test_config
):
    if not test_config.is_s3_compatible:
        pytest.skip("Requires two S3-compatible backends")
    secondary_config = os.environ.get("E2E_SECONDARY_CONFIG_FILE")
    if not secondary_config:
        if os.environ.get("E2E_REQUIRE_SECONDARY") == "1":
            pytest.fail("The required second remote backend is not configured")
        pytest.skip("The second remote backend is not configured")

    second_properties = _TestConfig(secondary_config).get_properties()
    schema = pa.schema([pa.field("id", pa.int64())])
    path = f"{temp_case_path}/same-object.parquet"
    contents = ((default_properties, [1, 2, 3]), (second_properties, [101, 102, 103]))

    for properties, ids in contents:
        output = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pydict({"id": ids}, schema=schema), output)
        with Filesystem.get(properties=properties) as fs:
            fs.create_dir(temp_case_path)
            fs.write_file(path, output.getvalue().to_pybytes())

    groups = ColumnGroups.from_list(
        [ColumnGroup(["id"], "parquet", [ColumnGroupFile(path, 0, 3)])]
    )
    with groups:
        with Reader(groups, schema, properties=default_properties) as first:
            with Reader(groups, schema, properties=second_properties) as second:
                for reader, expected in ((first, [1, 2, 3]), (second, [101, 102, 103])):
                    actual = (
                        pa.Table.from_batches(list(reader.scan()))
                        .column("id")
                        .to_pylist()
                    )
                    assert actual == expected
                    taken = pa.Table.from_batches(reader.take([0, 2]))
                    assert taken.column("id").to_pylist() == [expected[0], expected[2]]
                again = (
                    pa.Table.from_batches(list(first.scan())).column("id").to_pylist()
                )
                assert again == [1, 2, 3]
