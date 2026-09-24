"""Filesystem behavior visible through the Python API."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from milvus_storage import FaultInjector, Filesystem
from milvus_storage.exceptions import FFIError


@pytest.mark.e2e_smoke
def test_binary_file_read_write_and_range(temp_case_path, default_properties):
    payload = bytes(range(256)) * 2
    path = f"{temp_case_path}/数据 1.bin"
    with Filesystem.get(properties=default_properties) as fs:
        fs.create_dir(temp_case_path)
        with fs.open_writer(path) as writer:
            writer.write(payload[:257])
            writer.flush()
            writer.write(payload[257:])
        assert fs.get_file_size(path) == len(payload)
        assert fs.read_file_all(path) == payload
        assert fs.read_file(path, 253, 9) == payload[253:262]
        with fs.open_reader(path) as reader:
            assert reader.read_at(250, 13) == payload[250:263]
            assert reader.read_at(0, 0) == b""


@pytest.mark.e2e_smoke
def test_missing_file_error_keeps_path_and_reason(temp_case_path, default_properties):
    missing = f"{temp_case_path}/missing-e2e.bin"
    with Filesystem.get(properties=default_properties) as fs:
        with pytest.raises(FFIError) as failure:
            fs.read_file(missing, 0, 1)
    message = str(failure.value)
    assert "missing-e2e.bin" in message
    assert "notfound" in message.lower().replace(" ", "")
    assert "code " in message


def test_concurrent_read_errors_keep_their_own_path(temp_case_path, default_properties):
    with Filesystem.get(properties=default_properties) as fs:
        fs.create_dir(temp_case_path)
        barrier = Barrier(2)

        def read_missing(name):
            barrier.wait()
            with pytest.raises(FFIError) as failure:
                fs.read_file(f"{temp_case_path}/{name}", 0, 1)
            return str(failure.value)

        names = ("missing-left-e2e.bin", "missing-right-e2e.bin")
        with ThreadPoolExecutor(max_workers=2) as pool:
            errors = list(pool.map(read_missing, names))
        for index, name in enumerate(names):
            assert name in errors[index]
            assert names[1 - index] not in errors[index]

        path = f"{temp_case_path}/after-errors.bin"
        fs.write_file(path, b"recovered")
        assert fs.read_file_all(path) == b"recovered"


def test_directory_listing_and_file_info(temp_case_path, default_properties):
    nested = f"{temp_case_path}/nested/deep"
    path = f"{nested}/payload.bin"
    with Filesystem.get(properties=default_properties) as fs:
        fs.create_dir(nested, recursive=True)
        fs.write_file(path, b"\x00\xffdata")
        assert fs.exists(path)
        assert fs.is_directory(nested)
        exists, is_dir, mtime_ns = fs.get_path_info(path)
        assert (exists, is_dir) == (True, False)
        assert mtime_ns > 0
        assert fs.get_file_size(path) == 6
        assert any(
            info.path.endswith("nested") and info.is_dir
            for info in fs.list_dir(temp_case_path)
        )
        assert any(
            info.path.endswith("payload.bin") and info.size == 6
            for info in fs.list_dir(temp_case_path, recursive=True)
        )
        fs.delete_file(path)


@pytest.mark.e2e_smoke
def test_exists_reports_missing_path_after_delete(temp_case_path, default_properties):
    path = f"{temp_case_path}/deleted.bin"
    with Filesystem.get(properties=default_properties) as fs:
        fs.create_dir(temp_case_path)
        fs.write_file(path, b"present")
        assert fs.exists(path)
        fs.delete_file(path)
        with pytest.raises(FFIError) as failure:
            fs.exists(path)
        assert "File not found" in str(failure.value)
        assert path in str(failure.value)


def test_conditional_write_preserves_existing_object(
    temp_case_path, default_properties
):
    path = f"{temp_case_path}/conditional.bin"
    with Filesystem.get(properties=default_properties) as fs:
        fs.create_dir(temp_case_path)
        with fs.open_writer(path, conditional=True) as writer:
            writer.write(b"first")
        with pytest.raises(FFIError):
            with fs.open_writer(path, conditional=True) as writer:
                writer.write(b"unexpected replacement")
        assert fs.read_file_all(path) == b"first"
        fs.write_file(path, b"second")
        assert fs.read_file_all(path) == b"second"


def test_concurrent_conditional_create_has_one_complete_winner(
    temp_case_path, default_properties
):
    path = f"{temp_case_path}/winner.bin"
    with Filesystem.get(properties=default_properties) as fs:
        fs.create_dir(temp_case_path)
        barrier = Barrier(2)

        def create(content):
            barrier.wait()
            try:
                with fs.open_writer(path, conditional=True) as writer:
                    writer.write(content)
                return True
            except FFIError:
                return False

        candidates = [b"left" * 50, b"right" * 50]
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(create, candidates))
        assert outcomes.count(True) == 1
        assert fs.read_file_all(path) == candidates[outcomes.index(True)]


def test_filesystems_with_different_roots_do_not_mix_data(
    temp_case_path, temp_dir, default_properties, test_config
):
    if not test_config.is_local:
        pytest.skip("Filesystem root isolation needs two local roots")
    first = Filesystem.get(properties=default_properties)
    other_properties = dict(default_properties)
    other_properties["fs.root_path"] = str(temp_dir)
    second = Filesystem.get(properties=other_properties)
    path = f"{temp_case_path}/same-name.bin"
    with first, second:
        first.create_dir(temp_case_path)
        second.create_dir(temp_case_path)
        first.write_file(path, b"first-root")
        second.write_file(path, b"second-root")
        assert first.read_file_all(path) == b"first-root"
        assert second.read_file_all(path) == b"second-root"
        size, metadata = first.get_file_stats(path)
        assert size == len(b"first-root")
        assert isinstance(metadata, dict)

    Filesystem.close_all()
    with Filesystem.get(properties=default_properties) as reopened:
        assert reopened.read_file_all(path) == b"first-root"


def test_remote_file_stats_preserve_written_metadata(
    temp_case_path, default_properties, test_config
):
    if not test_config.is_s3_compatible:
        pytest.skip("Object metadata requires an S3-compatible backend")
    path = f"{temp_case_path}/metadata.bin"
    with Filesystem.get(properties=default_properties) as fs:
        fs.create_dir(temp_case_path)
        fs.write_file(path, b"metadata-payload", metadata={"e2e-label": "stored-value"})
        size, metadata = fs.get_file_stats(path)
        assert size == len(b"metadata-payload")
        assert metadata["e2e-label"] == "stored-value"


def test_metrics_reset_and_continue_counting(temp_case_path, default_properties):
    path = f"{temp_case_path}/metrics.bin"
    with Filesystem.get(properties=default_properties) as fs:
        fs.create_dir(temp_case_path)
        fs.reset_metrics()
        fs.write_file(path, b"abcdefgh")
        assert fs.read_file(path, 2, 3) == b"cde"
        metrics = fs.get_metrics()
        assert metrics.write_count >= 1
        assert metrics.read_count >= 1
        fs.reset_metrics()
        assert fs.read_file(path, 0, 2) == b"ab"
        assert fs.get_metrics().read_count >= 1


@pytest.mark.slow
def test_remote_multipart_configuration_preserves_bytes(
    temp_case_path, default_properties, test_config
):
    if not test_config.is_s3_compatible:
        pytest.skip("Multipart upload settings require an S3-compatible backend")
    properties = dict(default_properties)
    properties.update(
        {
            "fs.use_custom_part_upload": "true",
            "fs.multi_part_upload_size": str(10 * 1024 * 1024),
            "fs.background_writes": "true",
            "fs.use_crc32c_checksum": "true",
        }
    )
    payload = bytes(range(251)) * 50_000
    assert len(payload) > 10 * 1024 * 1024
    path = f"{temp_case_path}/multipart.bin"
    with Filesystem.get(properties=properties) as fs:
        fs.create_dir(temp_case_path)
        with fs.open_writer(path) as writer:
            writer.write(payload[:7_000_000])
            writer.write(payload[7_000_000:])
        assert fs.get_file_size(path) == len(payload)
        assert fs.read_file_all(path) == payload


@pytest.mark.cloud
@pytest.mark.parametrize("body_fails", [False, True])
def test_close_failure_preserves_available_reasons(
    temp_case_path, default_properties, test_config, require_fiu, body_fails
):
    if not test_config.is_s3_compatible:
        pytest.skip("S3 writer close fault requires an S3-compatible backend")
    path = f"{temp_case_path}/close-error.bin"
    with Filesystem.get(properties=default_properties) as fs:
        fs.create_dir(temp_case_path)
        try:
            with pytest.raises((FFIError, RuntimeError)) as failure:
                with fs.open_writer(path) as writer:
                    writer.write(b"contents")
                    require_fiu.enable(
                        FaultInjector.S3FS_WRITER_CLOSE_FAIL, one_time=True
                    )
                    if body_fails:
                        raise RuntimeError("e2e-primary-failure")
        finally:
            require_fiu.disable(FaultInjector.S3FS_WRITER_CLOSE_FAIL)

    reasons = []
    error = failure.value
    while error is not None:
        reasons.append(str(error))
        error = error.__cause__ or error.__context__
    assert any("Injected fault" in reason for reason in reasons)
    if body_fails:
        assert any("e2e-primary-failure" in reason for reason in reasons)
