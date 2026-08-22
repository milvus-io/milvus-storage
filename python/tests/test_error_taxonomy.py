"""Binding-level checks for error ownership and the structured taxonomy.

The exception TYPE is this binding's ergonomics; ``err_code`` and ``category``
are the library's verdict and must reach the caller as attributes rather than as
text inside a message (``docs/error-handling-rules.md`` R4.1-R4.3).
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import milvus_storage._ffi as ffi_module
from milvus_storage.exceptions import (
    ConflictError,
    DataFormatError,
    ErrorCategory,
    FFIError,
    InvalidArgumentError,
    RetryableError,
)

# Mirrors the native table for the codes these tests use. Kept tiny and
# explicit: the fake stands in for the C ABI, so it must answer like the C ABI
# rather than re-derive the answer from the Python enum it is verifying.
_FAKE_CATEGORIES = {
    1: ErrorCategory.SYSTEM,  # LOON_INVALID_ARGS
    7: ErrorCategory.SYSTEM,  # LOON_INVALID_PROPERTIES
    10: ErrorCategory.USER,  # LOON_USER_INVALID_ARGUMENT
    102: ErrorCategory.CONFLICT,  # StorageConflict
    109: ErrorCategory.RETRYABLE,  # StorageTransientThrottling
    117: ErrorCategory.DATA_FORMAT,  # DataCorrupted
}


class _FakeResultLib:
    loon_errcode_user_invalid_argument = 10
    loon_errcode_invalid_properties = 7

    @staticmethod
    def loon_ffi_is_success(result):
        return result.err_code == 0

    @staticmethod
    def loon_ffi_get_errmsg(result):
        return result.message

    @staticmethod
    def loon_ffi_error_category(err_code):
        # An unlisted code answers UNKNOWN, exactly as the native function does
        # for a value it cannot classify.
        return int(_FAKE_CATEGORIES.get(err_code, ErrorCategory.UNKNOWN))

    @staticmethod
    def loon_ffi_free_result(result):
        # The test message is owned by cffi rather than malloc, so the fake
        # native boundary deliberately has nothing to release.
        del result


@pytest.fixture
def fake_lib(monkeypatch):
    monkeypatch.setattr(ffi_module, "get_library", lambda: SimpleNamespace(lib=_FakeResultLib()))


def _result(code, message):
    ffi = ffi_module.get_ffi()
    message_buffer = ffi.new("char[]", message.encode("utf-8"))
    result = ffi.new("LoonFFIResult*")
    result.err_code = code
    result.message = message_buffer
    return result[0], message_buffer


def _raise(code, message):
    result, message_buffer = _result(code, message)
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 -- the type is what we assert on
        ffi_module.check_result(result)
    # Keep the cffi-owned message alive through check_result.
    assert message_buffer is not None
    return excinfo.value


def test_user_invalid_argument_maps_to_specific_exception(fake_lib):
    error = _raise(10, "path must not be empty")

    assert isinstance(error, InvalidArgumentError)
    assert "path must not be empty" in str(error)
    assert error.err_code == 10
    assert error.category == ErrorCategory.USER


def test_abi_invalid_args_remains_internal_ffi_error(fake_lib):
    error = _raise(1, "handle must not be null")

    assert isinstance(error, FFIError)
    assert error.err_code == 1
    assert error.category == ErrorCategory.SYSTEM


def test_python_owned_invalid_properties_maps_to_invalid_argument(fake_lib):
    error = _raise(7, "invalid property value")

    assert isinstance(error, InvalidArgumentError)
    # The binding chooses a friendlier exception type; it does NOT rewrite the
    # library's verdict, which stays System for every other consumer.
    assert error.category == ErrorCategory.SYSTEM


@pytest.mark.parametrize(
    ("code", "expected_type", "expected_category"),
    [
        (109, RetryableError, ErrorCategory.RETRYABLE),
        (102, ConflictError, ErrorCategory.CONFLICT),
        (117, DataFormatError, ErrorCategory.DATA_FORMAT),
    ],
)
def test_category_selects_the_exception_type(fake_lib, code, expected_type, expected_category):
    error = _raise(code, "boom")

    assert isinstance(error, expected_type)
    assert isinstance(error, FFIError), "category types stay catchable as FFIError"
    assert error.err_code == code
    assert error.category == expected_category
    assert error.retryable is (expected_category == ErrorCategory.RETRYABLE)


def test_conflict_is_not_retryable(fake_lib):
    # The one classification a generic retry helper must never get wrong.
    assert _raise(102, "lost the commit race").retryable is False


def test_unknown_code_lands_in_the_conservative_bucket(fake_lib):
    # A code from a newer library: no type of its own, no retry hint invented.
    error = _raise(9999, "from the future")

    assert type(error) is FFIError
    assert error.err_code == 9999
    assert error.category == ErrorCategory.UNKNOWN
    assert error.retryable is False


def test_error_category_matches_the_native_abi():
    """The Python enum is a transcription of ffi_error_code.h; pin it."""
    header = Path(__file__).resolve().parents[2] / "cpp/include/milvus-storage/ffi_internal/ffi_error_code.h"
    if not header.exists():
        pytest.skip("running outside the source tree")

    native = {
        name: int(value)
        for name, value in re.findall(r"#define\s+LOON_ERROR_CATEGORY_([A-Z_]+)\s+(\d+)", header.read_text())
    }
    assert native, "the category defines moved -- this gate needs updating"
    assert native == {member.name: int(member) for member in ErrorCategory}
