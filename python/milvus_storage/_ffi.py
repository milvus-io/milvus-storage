"""
Low-level FFI bindings for milvus-storage C API.

This module provides cffi wrappers around the C API defined in ffi_c.h.
"""

import os
import platform
from typing import Optional

from cffi import FFI

# Error codes from ffi_c.h
LOON_SUCCESS = 0
LOON_INVALID_ARGS = 1
LOON_MEMORY_ERROR = 2
LOON_ARROW_ERROR = 3
LOON_LOGICAL_ERROR = 4
LOON_GOT_EXCEPTION = 5
LOON_UNREACHABLE_ERROR = 6
LOON_INVALID_PROPERTIES = 7


# Create FFI instance and define C API
_ffi = FFI()

# Define C structures and function signatures
_ffi.cdef("""
    // ==================== Arrow C Data Interface ====================
    struct ArrowSchema {
        const char* format;
        const char* name;
        const char* metadata;
        int64_t flags;
        int64_t n_children;
        struct ArrowSchema** children;
        struct ArrowSchema* dictionary;
        void (*release)(struct ArrowSchema*);
        void* private_data;
    };

    struct ArrowArray {
        int64_t length;
        int64_t null_count;
        int64_t offset;
        int64_t n_buffers;
        int64_t n_children;
        const void** buffers;
        struct ArrowArray** children;
        struct ArrowArray* dictionary;
        void (*release)(struct ArrowArray*);
        void* private_data;
    };

    struct ArrowArrayStream {
        int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
        int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);
        const char* (*get_last_error)(struct ArrowArrayStream*);
        void (*release)(struct ArrowArrayStream*);
        void* private_data;
    };

    // ==================== Result C Interface ====================
    typedef struct LoonFFIResult {
        int err_code;
        char* message;
    } LoonFFIResult;

    int loon_ffi_is_success(LoonFFIResult* result);
    const char* loon_ffi_get_errmsg(LoonFFIResult* result);
    void loon_ffi_free_result(LoonFFIResult* result);

    // ==================== Properties C Interface ====================
    typedef struct LoonProperty {
        char* key;
        char* value;
    } LoonProperty;

    typedef struct LoonProperties {
        LoonProperty* properties;
        size_t count;
    } LoonProperties;

    LoonFFIResult loon_properties_create(const char* const* keys, const char* const* values, size_t count, LoonProperties* properties);
    const char* loon_properties_get(const LoonProperties* properties, const char* key);
    void loon_properties_free(LoonProperties* properties);

    // ==================== Writer C Interface ====================
    typedef uintptr_t LoonWriterHandle;

    LoonFFIResult loon_writer_new(const char* base_path,
                         struct ArrowSchema* schema,
                         const LoonProperties* properties,
                         LoonWriterHandle* out_handle);

    LoonFFIResult loon_writer_write(LoonWriterHandle handle, struct ArrowArray* array);
    LoonFFIResult loon_writer_flush(LoonWriterHandle handle);
    LoonFFIResult loon_writer_close(LoonWriterHandle handle, char** meta_keys, char** meta_vals, uint16_t meta_len, struct LoonColumnGroups** out_columngroups);
    void loon_writer_destroy(LoonWriterHandle handle);
    void loon_free_cstr(char* c_str);

    // ==================== ChunkReader C Interface ====================
    typedef uintptr_t LoonChunkReaderHandle;

    LoonFFIResult loon_get_chunk_indices(LoonChunkReaderHandle reader,
                                const int64_t* row_indices,
                                size_t num_indices,
                                int64_t** chunk_indices,
                                size_t* num_chunk_indices);

    void loon_free_chunk_indices(int64_t* chunk_indices);

    LoonFFIResult loon_get_chunk(LoonChunkReaderHandle reader, int64_t chunk_index, struct ArrowArray* out_array);

    LoonFFIResult loon_get_chunks(LoonChunkReaderHandle reader,
                         const int64_t* chunk_indices,
                         size_t num_indices,
                         int64_t parallelism,
                         struct ArrowArray** arrays,
                         size_t* num_arrays);

    void loon_free_chunk_arrays(struct ArrowArray* arrays, size_t num_arrays);
    void loon_chunk_reader_destroy(LoonChunkReaderHandle reader);

    // ==================== Reader C Interface ====================
    typedef uintptr_t LoonReaderHandle;

    // Forward declaration for ColumnGroups
    typedef struct LoonColumnGroups LoonColumnGroups;

    LoonFFIResult loon_reader_new(const LoonColumnGroups* columngroups,
                         struct ArrowSchema* schema,
                         const char* const* needed_columns,
                         size_t num_columns,
                         const LoonProperties* properties,
                         LoonReaderHandle* out_handle);

    void loon_reader_set_keyretriever(LoonReaderHandle reader, void* key_retriever);

    LoonFFIResult loon_get_record_batch_reader(LoonReaderHandle reader,
                                      const char* predicate,
                                      struct ArrowArrayStream* out_array_stream);

    LoonFFIResult loon_get_chunk_reader(LoonReaderHandle reader, int64_t column_group_id, LoonChunkReaderHandle* out_handle);

    LoonFFIResult loon_take(LoonReaderHandle reader,
                   const int64_t* row_indices,
                   size_t num_indices,
                   int64_t parallelism,
                   struct ArrowArray** out_arrays,
                   size_t* num_arrays);

    void loon_reader_destroy(LoonReaderHandle reader);

    // ==================== Transaction C Interface ====================
    typedef uintptr_t LoonTransactionHandle;

    // FFIResult get_latest_column_groups(const char* base_path, const Properties* properties, char** out_column_groups, int64_t* read_version);

    LoonFFIResult loon_transaction_begin(const char* base_path, const LoonProperties* properties, int64_t read_version, uint32_t retry_limit, LoonTransactionHandle* out_handle);

    // FFIResult transaction_get_column_groups(TransactionHandle handle, char** out_column_groups);

    LoonFFIResult loon_transaction_commit(LoonTransactionHandle handle, int64_t* out_committed_version);

    // FFIResult transaction_abort(TransactionHandle handle);

    void loon_transaction_destroy(LoonTransactionHandle handle);

    // Column groups create/destroy
    LoonFFIResult loon_column_groups_create(const char** columns,
                                                   size_t col_lens,
                                                   char* format,
                                                   char** paths,
                                                   int64_t* start_indices,
                                                   int64_t* end_indices,
                                                   size_t file_lens,
                                                   LoonColumnGroups** out_column_groups);

    void loon_column_groups_destroy(LoonColumnGroups* cgroups);
""")


def _find_library() -> str:
    """Find the milvus-storage shared library."""
    # Determine library name based on platform
    if platform.system() == "Linux":
        lib_name = "libmilvus-storage.so"
    elif platform.system() == "Darwin":
        lib_name = "libmilvus-storage.dylib"
    elif platform.system() == "Windows":
        lib_name = "milvus-storage.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")

    # Search paths (in order of priority)
    search_paths = [
        # 1. Bundled with package
        os.path.join(os.path.dirname(__file__), "lib", lib_name),
        # 2. Development build - Debug
        os.path.join(os.path.dirname(__file__), "..", "..", "cpp", "build", "Debug", lib_name),
        # 3. Development build - Release
        os.path.join(os.path.dirname(__file__), "..", "..", "cpp", "build", "Release", lib_name),
        # 4. System library (will use LD_LIBRARY_PATH/DYLD_LIBRARY_PATH)
        lib_name,
    ]

    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)

    # Try loading from system paths
    return lib_name


class MilvusStorageLib:
    """Wrapper around the milvus-storage C library using cffi."""

    def __init__(self):
        """Load the library using cffi."""
        lib_path = _find_library()

        try:
            print(f"Loading library from: {lib_path}")
            self._lib = _ffi.dlopen(lib_path, os.RTLD_LOCAL)
            print("Successfully loaded library")
        except Exception as e:
            print(f"Failed to load library: {e}")
            raise

    @property
    def ffi(self):
        """Get the FFI instance for creating C objects."""
        return _ffi

    @property
    def lib(self):
        """Get the loaded library."""
        return self._lib


# Global library instance
_lib_instance: Optional[MilvusStorageLib] = None


def get_library() -> MilvusStorageLib:
    """Get the global library instance, loading it if necessary."""
    global _lib_instance
    if _lib_instance is None:
        _lib_instance = MilvusStorageLib()
    return _lib_instance


def get_ffi():
    """Get the FFI instance."""
    return _ffi


def check_result(result) -> None:
    """Check FFI result and raise exception if failed.

    Args:
        result: LoonFFIResult struct (passed by value from C function)
    """
    from .exceptions import FFIError

    lib = get_library().lib
    ffi = get_ffi()

    # In cffi, C functions that return structs by value return cdata objects
    # We need to pass a pointer to these functions that expect FFIResult*
    # Create a pointer to the result
    result_ptr = ffi.new("LoonFFIResult*", result)

    if not lib.loon_ffi_is_success(result_ptr):
        error_msg = lib.loon_ffi_get_errmsg(result_ptr)
        if error_msg != ffi.NULL:
            msg = ffi.string(error_msg).decode("utf-8")
        else:
            msg = "Unknown error"
        lib.loon_ffi_free_result(result_ptr)
        raise FFIError(f"FFI call failed (code {result.err_code}): {msg}")
