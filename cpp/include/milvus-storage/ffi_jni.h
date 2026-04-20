// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MILVUS_STORAGE_FFI_JNI_H_
#define MILVUS_STORAGE_FFI_JNI_H_

#include <jni.h>

#include "milvus-storage/ffi_c.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==================== JNI-only per-batch RecordBatchReader helpers ====================
//
// These C helpers exist solely to service the JVM consumer path. They are
// intentionally NOT part of the cross-language FFI surface (`ffi_c.h`)
// because Arrow Java's C Data importer ignores `ArrowArray.offset`, forcing
// this layer to materialize sliced columns to offset=0 before export —
// a memory copy that non-JVM callers do not need. V3 and other non-JVM
// consumers should use `loon_get_record_batch_reader` (ArrowArrayStream)
// instead. The implementations live in `src/jni/reader_jni.cpp` and are
// compiled into `libmilvus-storage-jni.so` only.

typedef uintptr_t LoonRecordBatchReaderHandle;

LoonFFIResult loon_record_batch_reader_new(LoonReaderHandle reader,
                                           const char* predicate,
                                           LoonRecordBatchReaderHandle* out_handle);

LoonFFIResult loon_record_batch_reader_read_next(LoonRecordBatchReaderHandle handle,
                                                 struct ArrowArray* out_array,
                                                 struct ArrowSchema* out_schema);

void loon_record_batch_reader_destroy(LoonRecordBatchReaderHandle handle);

// ==================== JNI Result Utilities ====================

/**
 * @brief Throws a Java exception based on LoonFFIResult
 *
 * @param env JNI environment
 * @param result LoonFFIResult to convert to exception
 */
void ThrowJavaExceptionFromFFIResult(JNIEnv* env, const struct LoonFFIResult* result);

/**
 * @brief Converts string array to Java string array
 *
 * @param env JNI environment
 * @param strings C string array
 * @param count Number of strings
 * @return Java string array
 */
jobjectArray ConvertToJavaStringArray(JNIEnv* env, const char* const* strings, size_t count);

/**
 * @brief Converts Java string array to C string array
 *
 * @param env JNI environment
 * @param java_array Java string array
 * @param out_count Output count of strings
 * @return C string array (caller must free)
 */
const char** ConvertFromJavaStringArray(JNIEnv* env, jobjectArray java_array, size_t* out_count);

/**
 * @brief Frees C string array allocated by ConvertFromJavaStringArray
 *
 * @param env JNI environment
 * @param strings String array to free
 * @param count Number of strings
 */
void FreeStringArray(JNIEnv* env, const char** strings, size_t count);

// ==================== JNI Properties Interface ====================

/**
 * @brief Allocate properties
 *
 * @param env JNI environment
 * @param obj Java object
 * @return Properties pointer as long
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageProperties_allocateProperties(JNIEnv* env, jobject obj);

/**
 * @brief Create properties from Java/Scala map
 *
 * @param env JNI environment
 * @param java_map Java/Scala Map<String, String>
 * @param properties_ptr Pointer to properties
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageProperties_createProperties(JNIEnv* env,
                                                                                       jobject obj,
                                                                                       jobject java_map,
                                                                                       jlong properties_ptr);

/**
 * @brief Free properties
 *
 * @param env JNI environment
 * @param obj Java object
 * @param properties_ptr Pointer to properties
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageProperties_freeProperties(JNIEnv* env,
                                                                                     jobject obj,
                                                                                     jlong properties_ptr);

// ==================== JNI Writer Interface ====================

/**
 * @brief Create a new Writer
 *
 * @param env JNI environment
 * @param obj Java object
 * @param base_path Base path string
 * @param schema_ptr Pointer to Arrow schema
 * @param properties_ptr Pointer to properties
 * @return Writer handle as long
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerNew(
    JNIEnv* env, jobject obj, jstring base_path, jlong schema_ptr, jlong properties_ptr);

/**
 * @brief Write a record batch
 *
 * @param env JNI environment
 * @param obj Java object
 * @param writer_handle Writer handle
 * @param array_ptr Pointer to Arrow array
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerWrite(JNIEnv* env,
                                                                              jobject obj,
                                                                              jlong writer_handle,
                                                                              jlong array_ptr);

/**
 * @brief Flush the writer
 *
 * @param env JNI environment
 * @param obj Java object
 * @param writer_handle Writer handle
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerFlush(JNIEnv* env,
                                                                              jobject obj,
                                                                              jlong writer_handle);

/**
 * @brief Close the writer and return manifest
 *
 * @param env JNI environment
 * @param obj Java object
 * @param writer_handle Writer handle
 * @return Manifest raw pointer
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerClose(JNIEnv* env,
                                                                               jobject obj,
                                                                               jlong writer_handle);

/**
 * @brief Destroy the writer
 *
 * @param env JNI environment
 * @param obj Java object
 * @param writer_handle Writer handle
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerDestroy(JNIEnv* env,
                                                                                jobject obj,
                                                                                jlong writer_handle);

// ==================== JNI Reader Interface ====================

/**
 * @brief Create a new Reader
 *
 * @param env JNI environment
 * @param obj Java object
 * @param manifest Manifest raw pointer
 * @param schema_ptr Pointer to Arrow schema
 * @param needed_columns Array of column names
 * @param properties_ptr Pointer to properties
 * @return Reader handle as long
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_readerNew(
    JNIEnv* env, jobject obj, jlong manifest, jlong schema_ptr, jobjectArray needed_columns, jlong properties_ptr);

/**
 * @brief Get record batch reader
 *
 * @param env JNI environment
 * @param obj Java object
 * @param reader_handle Reader handle
 * @param predicate Predicate string
 * @param batch_size Batch size
 * @param buffer_size Buffer size
 * @return Arrow array stream pointer as long
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_getRecordBatchReader(JNIEnv* env,
                                                                                        jobject obj,
                                                                                        jlong reader_handle,
                                                                                        jstring predicate);

/**
 * @brief Get chunk reader
 *
 * @param env JNI environment
 * @param obj Java object
 * @param reader_handle Reader handle
 * @param column_group_id Column group ID
 * @param needed_columns Column names to project (null-safe jobjectArray)
 * @return Chunk reader handle as long
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_getChunkReader(
    JNIEnv* env, jobject obj, jlong reader_handle, jlong column_group_id, jobjectArray needed_columns);

/**
 * @brief Take specific rows
 *
 * @param env JNI environment
 * @param obj Java object
 * @param reader_handle Reader handle
 * @param row_indices Array of row indices
 * @param parallelism Parallelism level
 * @param needed_columns Column names to project (null-safe jobjectArray)
 * @return Arrow array pointer as long array
 */
JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageReader_take(JNIEnv* env,
                                                                             jobject obj,
                                                                             jlong reader_handle,
                                                                             jlongArray row_indices,
                                                                             jlong parallelism,
                                                                             jobjectArray needed_columns);

/**
 * @brief Open a pull-based RecordBatchReader for per-batch import.
 *
 * @param env JNI environment
 * @param obj Java object
 * @param reader_handle Reader handle
 * @param predicate Optional predicate expression (null if unused)
 * @return RecordBatchReader handle as long
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_recordBatchReaderNew(JNIEnv* env,
                                                                                        jobject obj,
                                                                                        jlong reader_handle,
                                                                                        jstring predicate);

/**
 * @brief Read the next RecordBatch into caller-allocated ArrowArray + ArrowSchema.
 *
 * @param env JNI environment
 * @param obj Java object
 * @param rbr_handle RecordBatchReader handle
 * @param array_addr Pointer (as jlong) to zero-initialized ArrowArray
 * @param schema_addr Pointer (as jlong) to zero-initialized ArrowSchema
 * @return true when a batch was produced; false on EOF
 */
JNIEXPORT jboolean JNICALL Java_io_milvus_storage_MilvusStorageReader_recordBatchReaderReadNext(
    JNIEnv* env, jobject obj, jlong rbr_handle, jlong array_addr, jlong schema_addr);

/**
 * @brief Destroy the RecordBatchReader.
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageReader_recordBatchReaderDestroy(JNIEnv* env,
                                                                                           jobject obj,
                                                                                           jlong rbr_handle);

/**
 * @brief Destroy the reader
 *
 * @param env JNI environment
 * @param obj Java object
 * @param reader_handle Reader handle
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageReader_readerDestroy(JNIEnv* env,
                                                                                jobject obj,
                                                                                jlong reader_handle);

// ==================== JNI ChunkReader Interface ====================

/**
 * @brief Get chunk indices
 *
 * @param env JNI environment
 * @param obj Java object
 * @param chunk_reader_handle Chunk reader handle
 * @param row_indices Array of row indices
 * @return Array of chunk indices
 */
JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageChunkReader_getChunkIndices(JNIEnv* env,
                                                                                             jobject obj,
                                                                                             jlong chunk_reader_handle,
                                                                                             jlongArray row_indices);

/**
 * @brief Get single chunk
 *
 * @param env JNI environment
 * @param obj Java object
 * @param chunk_reader_handle Chunk reader handle
 * @param chunk_index Chunk index
 * @return Arrow array pointer as long
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageChunkReader_getChunk(JNIEnv* env,
                                                                                 jobject obj,
                                                                                 jlong chunk_reader_handle,
                                                                                 jlong chunk_index);

/**
 * @brief Get multiple chunks
 *
 * @param env JNI environment
 * @param obj Java object
 * @param chunk_reader_handle Chunk reader handle
 * @param chunk_indices Array of chunk indices
 * @param parallelism Parallelism level
 * @return Array of Arrow array pointers
 */
JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageChunkReader_getChunks(
    JNIEnv* env, jobject obj, jlong chunk_reader_handle, jlongArray chunk_indices, jlong parallelism);

/**
 * @brief Destroy chunk reader
 *
 * @param env JNI environment
 * @param obj Java object
 * @param chunk_reader_handle Chunk reader handle
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageChunkReader_chunkReaderDestroy(JNIEnv* env,
                                                                                          jobject obj,
                                                                                          jlong chunk_reader_handle);

// ==================== JNI ArrowUtils Interface ====================

/**
 * @brief Read next batch from ArrowArrayStream
 *
 * @param env JNI environment
 * @param obj Java object
 * @param stream_ptr Pointer to ArrowArrayStream
 * @return Arrow array pointer as long (0 if end of stream)
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_ArrowUtilsNative_readNextBatch(JNIEnv* env,
                                                                              jobject obj,
                                                                              jlong stream_ptr);

/**
 * @brief Release ArrowArrayStream
 *
 * @param env JNI environment
 * @param obj Java object
 * @param stream_ptr Pointer to ArrowArrayStream
 * @param free_ptr Whether to free the pointer(If current ptr alloc in java, then false)
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_ArrowUtilsNative_releaseArrowStream(JNIEnv* env,
                                                                                  jobject obj,
                                                                                  jlong stream_ptr,
                                                                                  jboolean free_ptr);

/**
 * @brief Release ArrowArray
 *
 * @param env JNI environment
 * @param obj Java object
 * @param array_ptr Pointer to ArrowArray
 * @param free_ptr Whether to free the pointer(If current ptr alloc in java, then false)
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_ArrowUtilsNative_releaseArrowArray(JNIEnv* env,
                                                                                 jobject obj,
                                                                                 jlong array_ptr,
                                                                                 jboolean free_ptr);

/**
 * @brief Release ArrowSchema
 *
 * @param env JNI environment
 * @param obj Java object
 * @param schema_ptr Pointer to ArrowSchema
 * @param free_ptr Whether to free the pointer(If current ptr alloc in java, then false)
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_ArrowUtilsNative_releaseArrowSchema(JNIEnv* env,
                                                                                  jobject obj,
                                                                                  jlong schema_ptr,
                                                                                  jboolean free_ptr);
// ==================== JNI Manifest Interface ====================

/**
 * @brief Get latest column groups from manifest
 *
 * @param env JNI environment
 * @param obj Java object
 * @param base_path Base path string
 * @param properties_ptr Pointer to properties
 * @return Array of [columnGroupsPtr, readVersion]
 */
JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageManifestNative_getLatestColumnGroups(
    JNIEnv* env, jobject obj, jstring base_path, jlong properties_ptr);

/**
 * @brief Get column groups from manifest at a specific version
 *
 * @param env JNI environment
 * @param obj Java object
 * @param base_path Base path string
 * @param properties_ptr Pointer to properties
 * @param read_version Version to read (-1 for latest, >0 for specific version)
 * @return Array of [columnGroupsPtr, actualReadVersion]
 */
JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageManifestNative_getColumnGroupsWithVersion(
    JNIEnv* env, jobject obj, jstring base_path, jlong properties_ptr, jlong read_version);

// =============================================================================
// Transaction JNI Error-Handling Contract
// =============================================================================
// All MilvusStorageTransaction_* JNI methods signal errors via Java exceptions
// (typically RuntimeException). Exceptions are the *sole* error channel —
// callers must use try-catch; do NOT rely on return values to detect failure.
//
// Return types reflect the method's semantic output only:
//   - jlong: methods producing an output value (handle / pointer / version).
//            The sentinel value returned along the error path (e.g. -1) exists
//            only because the C++ signature requires a return statement after
//            env->ThrowNew(); Java callers see the exception first, never the
//            sentinel.
//   - void : methods without an output value (staging / lifecycle ops).
//
// This matches the Python binding where exceptions are also the sole error
// channel. Do NOT introduce status-code return values for void-semantic methods
// — that would create a redundant second error channel alongside exceptions.
// =============================================================================

/**
 * @brief Begin a transaction
 *
 * @param env JNI environment
 * @param obj Java object
 * @param base_path Base path string
 * @param properties_ptr Pointer to properties
 * @param read_version Version to read (-1 for latest)
 * @param resolve_id Conflict resolution strategy id (0=FAIL, 1=MERGE)
 * @param retry_limit Maximum retries on commit conflicts
 * @return Transaction handle as long; throws on error
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionBegin(JNIEnv* env,
                                                                                         jobject obj,
                                                                                         jstring base_path,
                                                                                         jlong properties_ptr,
                                                                                         jlong read_version,
                                                                                         jint resolve_id,
                                                                                         jint retry_limit);

/**
 * @brief Get column groups from current transaction
 *
 * @param env JNI environment
 * @param obj Java object
 * @param transaction_handle Transaction handle
 * @return Column groups raw pointer; throws on error
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionGetColumnGroups(
    JNIEnv* env, jobject obj, jlong transaction_handle);

/**
 * @brief Append files to existing column groups in the transaction
 *
 * @param env JNI environment
 * @param obj Java object
 * @param transaction_handle Transaction handle
 * @param column_groups Column groups raw pointer
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionAppendFiles(JNIEnv* env,
                                                                                              jobject obj,
                                                                                              jlong transaction_handle,
                                                                                              jlong column_groups);

/**
 * @brief Add column groups (schema evolution: add new fields) to the transaction
 *
 * @param env JNI environment
 * @param obj Java object
 * @param transaction_handle Transaction handle
 * @param column_groups Column groups raw pointer — each CG is added as a new field column group
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionAddColumnGroups(
    JNIEnv* env, jobject obj, jlong transaction_handle, jlong column_groups);

/**
 * @brief Drop a column from the transaction
 *
 * @param env JNI environment
 * @param obj Java object
 * @param transaction_handle Transaction handle
 * @param column_name Name of the column to drop
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionDropColumn(JNIEnv* env,
                                                                                             jobject obj,
                                                                                             jlong transaction_handle,
                                                                                             jstring column_name);

/**
 * @brief Commit a transaction
 *
 * @param env JNI environment
 * @param obj Java object
 * @param transaction_handle Transaction handle
 * @return Committed manifest version (>= 0); throws on error
 */
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionCommit(JNIEnv* env,
                                                                                          jobject obj,
                                                                                          jlong transaction_handle);

/**
 * @brief Abort a transaction
 *
 * @param env JNI environment
 * @param obj Java object
 * @param transaction_handle Transaction handle
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionAbort(JNIEnv* env,
                                                                                        jobject obj,
                                                                                        jlong transaction_handle);

/**
 * @brief Destroy a transaction
 *
 * @param env JNI environment
 * @param obj Java object
 * @param transaction_handle Transaction handle
 */
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionDestroy(JNIEnv* env,
                                                                                          jobject obj,
                                                                                          jlong transaction_handle);

// ==================== JNI SegmentWriter Interface ====================

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentWriter_segmentWriterNew(JNIEnv* env,
                                                                                    jobject obj,
                                                                                    jlong schema_ptr,
                                                                                    jstring segment_path,
                                                                                    jlongArray lob_field_ids,
                                                                                    jobjectArray lob_base_paths,
                                                                                    jlongArray lob_inline_thresholds,
                                                                                    jlongArray lob_max_file_bytes,
                                                                                    jlongArray lob_flush_thresholds,
                                                                                    jlong properties_ptr);

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusSegmentWriter_segmentWriterWrite(JNIEnv* env,
                                                                                     jobject obj,
                                                                                     jlong handle,
                                                                                     jlong array_ptr);

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusSegmentWriter_segmentWriterFlush(JNIEnv* env,
                                                                                     jobject obj,
                                                                                     jlong handle);

// Returns LoonSegmentWriteOutput as two values: columnGroupsPtr and lobFilesJson
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentWriter_segmentWriterClose(JNIEnv* env,
                                                                                      jobject obj,
                                                                                      jlong handle,
                                                                                      jlongArray out_lob_field_ids,
                                                                                      jobjectArray out_lob_paths,
                                                                                      jlongArray out_lob_total_rows,
                                                                                      jlongArray out_lob_valid_rows,
                                                                                      jlongArray out_lob_file_sizes);

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusSegmentWriter_segmentWriterDestroy(JNIEnv* env,
                                                                                       jobject obj,
                                                                                       jlong handle);

// ==================== JNI SegmentReader Interface ====================

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderOpen(JNIEnv* env,
                                                                                     jobject obj,
                                                                                     jstring segment_path,
                                                                                     jlong version,
                                                                                     jlong schema_ptr,
                                                                                     jobjectArray needed_columns,
                                                                                     jlongArray lob_field_ids,
                                                                                     jobjectArray lob_base_paths,
                                                                                     jlongArray lob_inline_thresholds,
                                                                                     jlongArray lob_max_file_bytes,
                                                                                     jlongArray lob_flush_thresholds,
                                                                                     jlong properties_ptr);

// Returns ArrowArrayStream pointer. TEXT columns are auto-decoded to utf8 strings.
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderGetStream(JNIEnv* env,
                                                                                          jobject obj,
                                                                                          jlong handle);

// Random access: extract specific rows by indices. Returns ArrowArrayStream pointer.
// TEXT columns are auto-decoded to utf8 strings.
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderTake(
    JNIEnv* env, jobject obj, jlong handle, jlongArray row_indices, jint parallelism);

// Sequential read with predicate filtering. Returns ArrowArrayStream pointer.
// TEXT columns are auto-decoded to utf8 strings.
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderGetFilteredStream(JNIEnv* env,
                                                                                                  jobject obj,
                                                                                                  jlong handle,
                                                                                                  jstring predicate);

// Get ChunkReader for a specific column group. Returns ChunkReader handle.
// NOTE: chunk data is NOT LOB-resolved (TEXT columns remain as binary refs).
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderGetChunkReader(
    JNIEnv* env, jobject obj, jlong handle, jlong column_group_index, jobjectArray needed_columns);

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderDestroy(JNIEnv* env,
                                                                                       jobject obj,
                                                                                       jlong handle);

#ifdef __cplusplus
}
#endif

#endif  // MILVUS_STORAGE_FFI_JNI_H_