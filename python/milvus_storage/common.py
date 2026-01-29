"""
Common utilities for milvus-storage.
"""

from ._ffi import check_result, get_library


class ThreadPool:
    """
    Thread pool singleton for milvus-storage operations.

    Manages a global thread pool used by read/write operations.
    The thread pool is shared across all operations.

    Example:
        >>> from milvus_storage import ThreadPool
        >>>
        >>> # Initialize with 4 threads
        >>> ThreadPool.init(4)
        >>>
        >>> # ... perform operations ...
        >>>
        >>> # Release when done
        >>> ThreadPool.release()
    """

    _initialized = False

    @classmethod
    def init(cls, num_threads: int) -> None:
        """
        Initialize or update the thread pool singleton.

        If the singleton already exists, updates the number of threads.
        Otherwise, creates a new thread pool.

        Args:
            num_threads: Number of threads in the pool

        Raises:
            InvalidArgumentError: If num_threads is not positive
            FFIError: If initialization fails
        """
        from .exceptions import InvalidArgumentError

        if num_threads <= 0:
            raise InvalidArgumentError(f"num_threads must be positive, got {num_threads}")

        lib = get_library().lib
        result = lib.loon_thread_pool_singleton(num_threads)
        check_result(result)
        cls._initialized = True

    @classmethod
    def release(cls) -> None:
        """
        Release the thread pool singleton.

        Waits for all threads to complete and releases resources.
        Can be called even if not initialized (no-op in that case).
        """
        lib = get_library().lib
        lib.loon_thread_pool_singleton_release()
        cls._initialized = False

    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if the thread pool is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return cls._initialized
