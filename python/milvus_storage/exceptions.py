"""
Exception classes for milvus-storage Python package.

Every failure that crosses the native boundary carries its structured verdict as
attributes -- ``err_code`` (the fine-grained ``loon_errcode_*`` value) and
``category`` (the coarse handling hint) -- and picks its exception type from the
category. Nothing a program must decide on lives in the message string: see
``docs/error-handling-rules.md`` R4.1-R4.3.
"""

from enum import IntEnum
from typing import Optional


class ErrorCategory(IntEnum):
    """Mirror of ``LoonErrorCategory`` in ``ffi_c.h``.

    The values are the C ABI contract. ``tests/test_error_taxonomy.py`` pins
    them against the enum the native library actually exports, so this
    transcription cannot drift silently.
    """

    UNKNOWN = 0
    USER = 1
    RETRYABLE = 2
    CONFLICT = 3
    DATA_FORMAT = 4
    SYSTEM = 5


class MilvusStorageError(Exception):
    """Base exception for milvus-storage errors.

    ``err_code`` is ``None`` for failures raised by the Python layer itself,
    which never crossed the C ABI and therefore have no native code.
    """

    def __init__(
        self,
        message: str,
        *,
        err_code: Optional[int] = None,
        category: int = ErrorCategory.UNKNOWN,
    ):
        super().__init__(message)
        self.err_code = err_code
        try:
            # A code from a newer library can carry a category this binding has
            # never heard of. Forward compatibility is the whole reason UNKNOWN
            # exists, so degrade to it instead of raising while raising.
            self.category = ErrorCategory(category)
        except ValueError:
            self.category = ErrorCategory.UNKNOWN

    @property
    def retryable(self) -> bool:
        """Whether the observed cause was transient.

        Not a promise that replaying the operation is safe: a failed writer must
        be discarded and recreated, and a Conflict needs coordination rather
        than a replay. See ``docs/error-codes.md``.
        """
        return self.category == ErrorCategory.RETRYABLE


class FFIError(MilvusStorageError):
    """A native failure that is not the caller's input.

    Base of the category-specific native errors below, so ``except FFIError``
    covers Retryable / Conflict / DataFormat / System / Unknown in one clause.

    The one native category it does NOT cover is ``USER``: those raise
    :class:`InvalidArgumentError`, which is a sibling rather than a subclass,
    because the same exception is raised by this binding's own argument
    validation and the caller should not have to care which side detected the
    same mistake. To catch everything this library raises, catch
    :class:`MilvusStorageError`.
    """

    pass


class RetryableError(FFIError):
    """A transient cause was identified (``ErrorCategory.RETRYABLE``)."""

    pass


class ConflictError(FFIError):
    """Concurrent modification; needs coordination, not a blind replay."""

    pass


class DataFormatError(FFIError):
    """Persisted bytes do not decode."""

    pass


class ArrowError(MilvusStorageError):
    """Error during Arrow data conversion."""

    pass


class InvalidArgumentError(MilvusStorageError):
    """Invalid argument passed to API."""

    pass


class ResourceError(MilvusStorageError):
    """Resource management error (memory, handles, etc.)."""

    pass
