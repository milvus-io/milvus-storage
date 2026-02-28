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
LOON_ERRORCODE_MAX = 8

# Chunk metadata type flags from ffi_c.h
LOON_CHUNK_METADATA_ESTIMATED_MEMORY = 0x01
LOON_CHUNK_METADATA_NUMOFROWS = 0x02
LOON_CHUNK_METADATA_ALL = LOON_CHUNK_METADATA_ESTIMATED_MEMORY | LOON_CHUNK_METADATA_NUMOFROWS

# Transaction resolve strategies from ffi_c.h
LOON_TRANSACTION_RESOLVE_FAIL = 0
LOON_TRANSACTION_RESOLVE_MERGE = 1
LOON_TRANSACTION_RESOLVE_OVERWRITE = 2


# Create FFI instance and define C API
_ffi = FFI()

# Define C structures and function signatures
_ffi.cdef(
    """
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

    // Property key constants
    extern const char* loon_properties_format;
    extern const char* loon_properties_fs_address;
    extern const char* loon_properties_fs_bucket_name;
    extern const char* loon_properties_fs_access_key_id;
    extern const char* loon_properties_fs_access_key_value;
    extern const char* loon_properties_fs_root_path;
    extern const char* loon_properties_fs_storage_type;
    extern const char* loon_properties_fs_cloud_provider;
    extern const char* loon_properties_fs_iam_endpoint;
    extern const char* loon_properties_fs_log_level;
    extern const char* loon_properties_fs_region;
    extern const char* loon_properties_fs_use_ssl;
    extern const char* loon_properties_fs_ssl_ca_cert;
    extern const char* loon_properties_fs_use_iam;
    extern const char* loon_properties_fs_use_virtual_host;
    extern const char* loon_properties_fs_request_timeout_ms;
    extern const char* loon_properties_fs_gcp_native_without_auth;
    extern const char* loon_properties_fs_gcp_credential_json;
    extern const char* loon_properties_fs_use_custom_part_upload;
    extern const char* loon_properties_fs_max_connections;
    extern const char* loon_properties_fs_multi_part_upload_size;
    extern const char* loon_properties_writer_policy;
    extern const char* loon_properties_writer_schema_base_patterns;
    extern const char* loon_properties_writer_size_base_macs;
    extern const char* loon_properties_writer_size_base_mcig;
    extern const char* loon_properties_writer_buffer_size;
    extern const char* loon_properties_writer_file_rolling_size;
    extern const char* loon_properties_writer_compression;
    extern const char* loon_properties_writer_compression_level;
    extern const char* loon_properties_writer_enable_dictionary;
    extern const char* loon_properties_writer_enc_enable;
    extern const char* loon_properties_writer_enc_key;
    extern const char* loon_properties_writer_enc_meta;
    extern const char* loon_properties_writer_enc_algorithm;
    extern const char* loon_properties_writer_vortex_enable_statistics;
    extern const char* loon_properties_reader_record_batch_max_rows;
    extern const char* loon_properties_reader_record_batch_max_size;
    extern const char* loon_properties_reader_logical_chunk_rows;
    extern const char* loon_properties_transaction_commit_num_retries;

    // ==================== ColumnGroups C Interface ====================
    typedef struct LoonColumnGroupFile {
        const char* path;
        int64_t start_index;
        int64_t end_index;
        uint8_t* metadata;
        uint64_t metadata_size;
    } LoonColumnGroupFile;

    typedef struct LoonColumnGroup {
        const char** columns;
        uint32_t num_of_columns;
        const char* format;
        LoonColumnGroupFile* files;
        uint32_t num_of_files;
    } LoonColumnGroup;

    typedef struct LoonColumnGroups {
        LoonColumnGroup* column_group_array;
        uint32_t num_of_column_groups;
    } LoonColumnGroups;

    typedef struct LoonDeltaLogs {
        const char** delta_log_paths;
        uint32_t* delta_log_num_entries;
        uint32_t num_delta_logs;
    } LoonDeltaLogs;

    typedef struct LoonStatsLog {
        const char** stat_keys;
        const char*** stat_files;
        uint32_t* stat_file_counts;
        const char*** stat_metadata_keys;
        const char*** stat_metadata_values;
        uint32_t* stat_metadata_counts;
        uint32_t num_stats;
    } LoonStatsLog;

    typedef struct LoonManifest {
        LoonColumnGroups column_groups;
        LoonDeltaLogs delta_logs;
        LoonStatsLog stats;
    } LoonManifest;

    void loon_manifest_destroy(LoonManifest* manifest);
    char* loon_manifest_debug_string(const LoonManifest* manifest);

    void loon_column_groups_destroy(LoonColumnGroups* cgroups);
    char* loon_column_groups_debug_string(const LoonColumnGroups* cgroups);

    // ==================== ThreadPool C Interface ====================
    LoonFFIResult loon_thread_pool_singleton(size_t num_of_thread);
    void loon_thread_pool_singleton_release();

    // ==================== Writer C Interface ====================
    typedef uintptr_t LoonWriterHandle;

    LoonFFIResult loon_writer_new(const char* base_path,
                                  struct ArrowSchema* schema,
                                  const LoonProperties* properties,
                                  LoonWriterHandle* out_handle);

    LoonFFIResult loon_writer_write(LoonWriterHandle handle, struct ArrowArray* array);
    LoonFFIResult loon_writer_flush(LoonWriterHandle handle);
    LoonFFIResult loon_writer_close(LoonWriterHandle handle, char** meta_keys, char** meta_vals, uint16_t meta_len, LoonColumnGroups** out_columngroups);
    void loon_writer_destroy(LoonWriterHandle handle);
    void loon_free_cstr(char* c_str);

    // ==================== ChunkReader C Interface ====================
    typedef uintptr_t LoonChunkReaderHandle;

    typedef struct LoonChunkMetadata {
        uint32_t metadata_type;
        union {
            uint64_t estimated_memsz;
            uint64_t number_of_rows;
        } *data;
        uint64_t number_of_chunks;
    } LoonChunkMetadata;

    typedef struct LoonChunkMetadatas {
        LoonChunkMetadata* metadatas;
        uint8_t metadatas_size;
    } LoonChunkMetadatas;

    LoonFFIResult loon_get_number_of_chunks(LoonChunkReaderHandle chunk_reader, uint64_t* out_number_of_chunks);

    LoonFFIResult loon_get_chunk_metadatas(LoonChunkReaderHandle reader,
                                           uint32_t metadata_type,
                                           LoonChunkMetadatas* out_chunk_metadata);

    LoonFFIResult loon_get_chunk_indices(LoonChunkReaderHandle reader,
                                         const int64_t* row_indices,
                                         size_t num_indices,
                                         int64_t** chunk_indices,
                                         size_t* num_chunk_indices);

    void loon_free_chunk_indices(int64_t* chunk_indices);

    LoonFFIResult loon_get_chunk(LoonChunkReaderHandle reader, int64_t chunk_index, struct ArrowArray* out_array, struct ArrowSchema* out_schema);

    LoonFFIResult loon_get_chunks(LoonChunkReaderHandle reader,
                                  const int64_t* chunk_indices,
                                  size_t num_indices,
                                  size_t parallelism,
                                  struct ArrowArray** arrays,
                                  size_t* num_arrays,
                                  struct ArrowSchema* out_schema);

    void loon_free_chunk_arrays(struct ArrowArray* arrays, size_t num_arrays);
    void loon_free_chunk_metadatas(LoonChunkMetadatas* chunk_metadata);
    void loon_chunk_reader_destroy(LoonChunkReaderHandle reader);

    // ==================== Reader C Interface ====================
    typedef uintptr_t LoonReaderHandle;

    LoonFFIResult loon_reader_new(const LoonColumnGroups* column_groups,
                                  struct ArrowSchema* schema,
                                  const char* const* needed_columns,
                                  size_t num_columns,
                                  const LoonProperties* properties,
                                  LoonReaderHandle* out_handle);

    void loon_reader_set_keyretriever(LoonReaderHandle reader, const char* (*key_retriever)(const char* metadata));

    LoonFFIResult loon_get_record_batch_reader(LoonReaderHandle reader,
                                               const char* predicate,
                                               struct ArrowArrayStream* out_array_stream);

    LoonFFIResult loon_get_chunk_reader(LoonReaderHandle reader,
                                        int64_t column_group_id,
                                        const char* const* needed_columns,
                                        size_t num_columns,
                                        LoonChunkReaderHandle* out_handle);

    LoonFFIResult loon_take(LoonReaderHandle reader,
                            const int64_t* row_indices,
                            size_t num_indices,
                            size_t parallelism,
                            const char* const* needed_columns,
                            size_t num_columns,
                            struct ArrowArray** out_arrays,
                            size_t* num_arrays,
                            struct ArrowSchema* out_schema);

    void loon_reader_destroy(LoonReaderHandle reader);

    // ==================== Transaction C Interface ====================
    typedef uintptr_t LoonTransactionHandle;

    LoonFFIResult loon_transaction_begin(const char* base_path,
                                         const LoonProperties* properties,
                                         int64_t read_version,
                                         uint32_t retry_limit,
                                         LoonTransactionHandle* out_handle);

    LoonFFIResult loon_transaction_get_manifest(LoonTransactionHandle handle, LoonManifest** out_manifest);

    LoonFFIResult loon_transaction_get_read_version(LoonTransactionHandle handle, int64_t* out_read_version);

    LoonFFIResult loon_transaction_commit(LoonTransactionHandle handle, int64_t* out_committed_version);

    void loon_transaction_destroy(LoonTransactionHandle handle);

    LoonFFIResult loon_transaction_add_column_group(LoonTransactionHandle handle,
                                                    const LoonColumnGroup* column_group);

    LoonFFIResult loon_transaction_append_files(LoonTransactionHandle handle,
                                                const LoonColumnGroups* column_groups);

    LoonFFIResult loon_transaction_add_delta_log(LoonTransactionHandle handle,
                                                 const char* path,
                                                 int64_t num_entries);

    LoonFFIResult loon_transaction_update_stat(LoonTransactionHandle handle,
                                               const char* key,
                                               const char* const* files,
                                               size_t files_len,
                                               const char* const* metadata_keys,
                                               const char* const* metadata_values,
                                               size_t metadata_len);

    // ==================== External Table C Interface (ffi_exttable_c.h) ====================
    LoonFFIResult loon_exttable_explore(const char** columns,
                                        size_t col_lens,
                                        const char* format,
                                        const char* base_dir,
                                        const char* explore_dir,
                                        const LoonProperties* properties,
                                        uint64_t* out_num_of_files,
                                        char** out_column_groups_file_path);

    LoonFFIResult loon_exttable_get_file_info(const char* format,
                                              const char* file_path,
                                              const LoonProperties* properties,
                                              uint64_t* out_num_of_rows);

    LoonFFIResult loon_exttable_read_manifest(const char* manifest_file_path,
                                              const LoonProperties* properties,
                                              LoonManifest** out_manifest);

    // ==================== Filesystem C Interface (ffi_filesystem_c.h) ====================
    typedef uintptr_t FileSystemHandle;
    typedef uintptr_t FileSystemWriterHandle;
    typedef uintptr_t FileSystemReaderHandle;

    typedef struct LoonFileSystemMeta {
        char* key;
        char* value;
    } LoonFileSystemMeta;

    typedef struct LoonFileInfo {
        char* path;
        uint32_t path_len;
        bool is_dir;
        uint64_t size;
        int64_t mtime_ns;
    } LoonFileInfo;

    typedef struct LoonFileInfoList {
        LoonFileInfo* entries;
        uint32_t count;
    } LoonFileInfoList;

    LoonFFIResult loon_filesystem_get(const LoonProperties* properties,
                                      const char* path,
                                      uint32_t path_len,
                                      FileSystemHandle* out_handle);

    void loon_filesystem_destroy(FileSystemHandle handle);

    void loon_close_filesystems();

    LoonFFIResult loon_filesystem_open_writer(FileSystemHandle handle,
                                              const char* path_ptr,
                                              uint32_t path_len,
                                              const LoonFileSystemMeta* meta_array,
                                              uint32_t num_of_meta,
                                              FileSystemWriterHandle* out_handle);

    LoonFFIResult loon_filesystem_writer_write(FileSystemWriterHandle handle,
                                               const uint8_t* data,
                                               uint64_t size);

    LoonFFIResult loon_filesystem_writer_flush(FileSystemWriterHandle handle);

    LoonFFIResult loon_filesystem_writer_close(FileSystemWriterHandle handle);

    void loon_filesystem_writer_destroy(FileSystemWriterHandle handle);

    LoonFFIResult loon_filesystem_get_file_info(FileSystemHandle handle,
                                                const char* path_ptr,
                                                uint32_t path_len,
                                                uint64_t* out_size);

    LoonFFIResult loon_filesystem_read_file(FileSystemHandle handle,
                                            const char* path_ptr,
                                            uint32_t path_len,
                                            uint64_t offset,
                                            uint64_t nbytes,
                                            uint8_t* out_data);

    LoonFFIResult loon_filesystem_open_reader(FileSystemHandle handle,
                                              const char* path_ptr,
                                              uint32_t path_len,
                                              FileSystemReaderHandle* out_reader_ptr);

    LoonFFIResult loon_filesystem_reader_readat(FileSystemReaderHandle handle,
                                                uint64_t offset,
                                                uint64_t nbytes,
                                                uint8_t* out_data);

    LoonFFIResult loon_filesystem_reader_close(FileSystemReaderHandle handle);

    void loon_filesystem_reader_destroy(FileSystemReaderHandle handle);

    LoonFFIResult loon_initialize_filesystem_singleton(const LoonProperties* properties);

    LoonFFIResult loon_get_filesystem_singleton_handle(FileSystemHandle* out_handle);

    LoonFFIResult loon_filesystem_get_file_stats(FileSystemHandle handle,
                                                 const char* path_ptr,
                                                 uint32_t path_len,
                                                 uint64_t* out_size,
                                                 LoonFileSystemMeta** out_meta_array,
                                                 uint32_t* out_meta_count);

    void loon_filesystem_free_meta_array(LoonFileSystemMeta* meta_array, uint32_t meta_count);

    LoonFFIResult loon_filesystem_read_file_all(FileSystemHandle handle,
                                                const char* path_ptr,
                                                uint32_t path_len,
                                                uint8_t** out_data,
                                                uint64_t* out_size);

    LoonFFIResult loon_filesystem_write_file(FileSystemHandle handle,
                                             const char* path_ptr,
                                             uint32_t path_len,
                                             const uint8_t* data,
                                             uint64_t data_size,
                                             const LoonFileSystemMeta* meta_array,
                                             uint32_t meta_count);

    LoonFFIResult loon_filesystem_delete_file(FileSystemHandle handle,
                                              const char* path_ptr,
                                              uint32_t path_len);

    LoonFFIResult loon_filesystem_get_path_info(FileSystemHandle handle,
                                                const char* path_ptr,
                                                uint32_t path_len,
                                                bool* out_exists,
                                                bool* out_is_dir,
                                                int64_t* out_mtime_ns);

    LoonFFIResult loon_filesystem_create_dir(FileSystemHandle handle,
                                             const char* path_ptr,
                                             uint32_t path_len,
                                             bool recursive);

    LoonFFIResult loon_filesystem_list_dir(FileSystemHandle handle,
                                           const char* path_ptr,
                                           uint32_t path_len,
                                           bool recursive,
                                           LoonFileInfoList* out_list);

    void loon_filesystem_free_file_info_list(LoonFileInfoList* list);

    // ==================== Filesystem Metrics C Interface (ffi_filesystem_metrics_c.h) ====================
    typedef struct LoonFilesystemMetricsSnapshot {
        int64_t read_count;
        int64_t write_count;
        int64_t read_bytes;
        int64_t write_bytes;
        int64_t get_file_info_count;
        int64_t create_dir_count;
        int64_t delete_dir_count;
        int64_t delete_file_count;
        int64_t move_count;
        int64_t copy_file_count;
        int64_t failed_count;
        int64_t multi_part_upload_created;
        int64_t multi_part_upload_finished;
    } LoonFilesystemMetricsSnapshot;

    LoonFFIResult loon_filesystem_get_metrics(FileSystemHandle handle,
                                              LoonFilesystemMetricsSnapshot* out_metrics);

    LoonFFIResult loon_filesystem_reset_metrics(FileSystemHandle handle);

    // ==================== Fault Injection C Interface (ffi_fiu_c.h) ====================
    // Fault point key constants (exported from C library)
    extern const char* loon_fiukey_writer_write_fail;
    extern const char* loon_fiukey_writer_flush_fail;
    extern const char* loon_fiukey_writer_close_fail;
    extern const char* loon_fiukey_column_group_read_fail;
    extern const char* loon_fiukey_take_rows_fail;
    extern const char* loon_fiukey_chunk_reader_read_fail;
    extern const char* loon_fiukey_reader_open_fail;
    extern const char* loon_fiukey_manifest_commit_fail;
    extern const char* loon_fiukey_manifest_read_fail;
    extern const char* loon_fiukey_manifest_write_fail;
    extern const char* loon_fiukey_fs_open_output_fail;
    extern const char* loon_fiukey_fs_open_input_fail;
    extern const char* loon_fiukey_s3fs_create_upload_fail;
    extern const char* loon_fiukey_s3fs_part_upload_fail;
    extern const char* loon_fiukey_s3fs_complete_upload_fail;
    extern const char* loon_fiukey_s3fs_read_fail;
    extern const char* loon_fiukey_s3fs_readat_fail;
    extern const char* loon_fiukey_column_group_write_fail;

    // Fault injection functions
    LoonFFIResult loon_fiu_enable(const char* name, uint32_t name_len, int one_time);
    LoonFFIResult loon_fiu_disable(const char* name, uint32_t name_len);
    void loon_fiu_disable_all(void);
    int loon_fiu_is_enabled(void);
"""
)


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
            self._lib = _ffi.dlopen(lib_path, os.RTLD_LOCAL)
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


def manifest_debug_string(manifest) -> str:
    """
    Get a debug string representation of a LoonManifest.

    Args:
        manifest: LoonManifest pointer

    Returns:
        Debug string representation
    """
    lib = get_library().lib
    ffi = get_ffi()

    c_str = lib.loon_manifest_debug_string(manifest)
    if c_str == ffi.NULL:
        return "LoonManifest(null)"

    result = ffi.string(c_str).decode("utf-8")
    lib.loon_free_cstr(c_str)
    return result


def column_groups_debug_string(column_groups) -> str:
    """
    Get a debug string representation of a LoonColumnGroups.

    Args:
        column_groups: LoonColumnGroups pointer

    Returns:
        Debug string representation
    """
    lib = get_library().lib
    ffi = get_ffi()

    c_str = lib.loon_column_groups_debug_string(column_groups)
    if c_str == ffi.NULL:
        return "LoonColumnGroups(null)"

    result = ffi.string(c_str).decode("utf-8")
    lib.loon_free_cstr(c_str)
    return result
