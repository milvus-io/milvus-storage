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
    typedef struct ffi_result {
        int err_code;
        char* message;
    } FFIResult;

    int IsSuccess(FFIResult* result);
    const char* GetErrorMessage(FFIResult* result);
    void FreeFFIResult(FFIResult* result);

    // ==================== Properties C Interface ====================
    typedef struct Property {
        char* key;
        char* value;
    } Property;

    typedef struct Properties {
        Property* properties;
        size_t count;
    } Properties;

    FFIResult properties_create(const char* const* keys, const char* const* values, size_t count, Properties* properties);
    const char* properties_get(const Properties* properties, const char* key);
    void properties_free(Properties* properties);

    // ==================== Writer C Interface ====================
    typedef uintptr_t WriterHandle;

    FFIResult writer_new(const char* base_path,
                         struct ArrowSchema* schema,
                         const Properties* properties,
                         WriterHandle* out_handle);

    FFIResult writer_write(WriterHandle handle, struct ArrowArray* array);
    FFIResult writer_flush(WriterHandle handle);
    FFIResult writer_close(WriterHandle handle, char **config_key, char **config_value, uint16_t config_len, char** out_columngroups);
    void writer_destroy(WriterHandle handle);
    void free_cstr(char* c_str);

    // ==================== ChunkReader C Interface ====================
    typedef uintptr_t ChunkReaderHandle;

    FFIResult get_chunk_indices(ChunkReaderHandle reader,
                                const int64_t* row_indices,
                                size_t num_indices,
                                int64_t** chunk_indices,
                                size_t* num_chunk_indices);

    void free_chunk_indices(int64_t* chunk_indices);

    FFIResult get_chunk(ChunkReaderHandle reader, int64_t chunk_index, struct ArrowArray* out_array);

    FFIResult get_chunks(ChunkReaderHandle reader,
                         const int64_t* chunk_indices,
                         size_t num_indices,
                         int64_t parallelism,
                         struct ArrowArray** arrays,
                         size_t* num_arrays);

    void free_chunk_arrays(struct ArrowArray* arrays, size_t num_arrays);
    void chunk_reader_destroy(ChunkReaderHandle reader);

    // ==================== Reader C Interface ====================
    typedef uintptr_t ReaderHandle;

    FFIResult reader_new(char* columngroups,
                         struct ArrowSchema* schema,
                         const char* const* needed_columns,
                         size_t num_columns,
                         const Properties* properties,
                         ReaderHandle* out_handle);

    void reader_set_keyretriever(ReaderHandle reader, void* key_retriever);

    FFIResult get_record_batch_reader(ReaderHandle reader,
                                      const char* predicate,
                                      struct ArrowArrayStream* out_array_stream);

    FFIResult get_chunk_reader(ReaderHandle reader, int64_t column_group_id, ChunkReaderHandle* out_handle);

    FFIResult take(ReaderHandle reader,
                   const int64_t* row_indices,
                   size_t num_indices,
                   int64_t parallelism,
                   struct ArrowArray* out_arrays);

    void reader_destroy(ReaderHandle reader);

    // ==================== Transaction C Interface ====================
    typedef uintptr_t TransactionHandle;

    FFIResult get_latest_column_groups(const char* base_path, const Properties* properties, char** out_column_groups, int64_t* read_version);

    FFIResult transaction_begin(const char* base_path, const Properties* properties, TransactionHandle* out_handle);

    FFIResult transaction_get_column_groups(TransactionHandle handle, char** out_column_groups);
          
    int64_t transaction_get_read_version(TransactionHandle handle);

    FFIResult transaction_commit(TransactionHandle handle, int16_t update_id, int16_t resolve_id, char* in_column_groups, bool* out_commit_result);

    FFIResult transaction_abort(TransactionHandle handle);

    void transaction_destroy(TransactionHandle handle);
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
            self._lib = _ffi.dlopen(lib_path, os.RTLD_DEEPBIND|os.RTLD_LOCAL)
            print(f"Successfully loaded library")
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
        result: FFIResult struct (passed by value from C function)
    """
    from .exceptions import FFIError

    lib = get_library().lib
    ffi = get_ffi()

    # In cffi, C functions that return structs by value return cdata objects
    # We need to pass a pointer to these functions that expect FFIResult*
    # Create a pointer to the result
    result_ptr = ffi.new("FFIResult*", result)

    if not lib.IsSuccess(result_ptr):
        error_msg = lib.GetErrorMessage(result_ptr)
        if error_msg != ffi.NULL:
            msg = ffi.string(error_msg).decode('utf-8')
        else:
            msg = "Unknown error"
        lib.FreeFFIResult(result_ptr)
        raise FFIError(f"FFI call failed (code {result.err_code}): {msg}")
