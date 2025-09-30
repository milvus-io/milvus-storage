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

#ifdef __cplusplus
extern "C" {
#endif

// ==================== JNI Result Utilities ====================

/**
 * @brief Throws a Java exception based on FFIResult
 *
 * @param env JNI environment
 * @param result FFI result to convert to exception
 */
void ThrowJavaExceptionFromFFIResult(JNIEnv* env, const struct ffi_result* result);

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
 * @brief Create properties from Java/Scala map
 *
 * @param env JNI environment
 * @param java_map Java/Scala Map<String, String>
 * @param properties_ptr Pointer to properties
 */
JNIEXPORT void JNICALL
Java_io_milvus_storage_Properties_createProperties(JNIEnv* env, jobject obj, jobject java_map, jlong properties_ptr);

/**
 * @brief Free properties
 *
 * @param env JNI environment
 * @param properties_ptr Pointer to properties
 */
JNIEXPORT void JNICALL
Java_io_milvus_storage_Properties_freeProperties(JNIEnv* env, jobject obj, jlong properties_ptr);

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
JNIEXPORT jlong JNICALL
Java_io_milvus_storage_Writer_writerNew(JNIEnv* env, jobject obj, jstring base_path, jlong schema_ptr, jlong properties_ptr);

/**
 * @brief Write a record batch
 *
 * @param env JNI environment
 * @param obj Java object
 * @param writer_handle Writer handle
 * @param array_ptr Pointer to Arrow array
 */
JNIEXPORT void JNICALL
Java_io_milvus_storage_Writer_writerWrite(JNIEnv* env, jobject obj, jlong writer_handle, jlong array_ptr);

/**
 * @brief Flush the writer
 *
 * @param env JNI environment
 * @param obj Java object
 * @param writer_handle Writer handle
 */
JNIEXPORT void JNICALL
Java_io_milvus_storage_Writer_writerFlush(JNIEnv* env, jobject obj, jlong writer_handle);

/**
 * @brief Close the writer and return manifest
 *
 * @param env JNI environment
 * @param obj Java object
 * @param writer_handle Writer handle
 * @return Manifest as string
 */
JNIEXPORT jstring JNICALL
Java_io_milvus_storage_Writer_writerClose(JNIEnv* env, jobject obj, jlong writer_handle);

/**
 * @brief Destroy the writer
 *
 * @param env JNI environment
 * @param obj Java object
 * @param writer_handle Writer handle
 */
JNIEXPORT void JNICALL
Java_io_milvus_storage_Writer_writerDestroy(JNIEnv* env, jobject obj, jlong writer_handle);

// ==================== JNI Reader Interface ====================

/**
 * @brief Create a new Reader
 *
 * @param env JNI environment
 * @param obj Java object
 * @param manifest Manifest string
 * @param schema_ptr Pointer to Arrow schema
 * @param needed_columns Array of column names
 * @param properties_ptr Pointer to properties
 * @return Reader handle as long
 */
JNIEXPORT jlong JNICALL
Java_io_milvus_storage_Reader_readerNew(JNIEnv* env, jobject obj, jstring manifest, jlong schema_ptr, jobjectArray needed_columns, jlong properties_ptr);

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
JNIEXPORT jlong JNICALL
Java_io_milvus_storage_Reader_getRecordBatchReader(JNIEnv* env, jobject obj, jlong reader_handle, jstring predicate, jlong batch_size, jlong buffer_size);

/**
 * @brief Get chunk reader
 *
 * @param env JNI environment
 * @param obj Java object
 * @param reader_handle Reader handle
 * @param column_group_id Column group ID
 * @return Chunk reader handle as long
 */
JNIEXPORT jlong JNICALL
Java_io_milvus_storage_Reader_getChunkReader(JNIEnv* env, jobject obj, jlong reader_handle, jlong column_group_id);

/**
 * @brief Take specific rows
 *
 * @param env JNI environment
 * @param obj Java object
 * @param reader_handle Reader handle
 * @param row_indices Array of row indices
 * @param parallelism Parallelism level
 * @return Arrow array pointer as long
 */
JNIEXPORT jlong JNICALL
Java_io_milvus_storage_Reader_take(JNIEnv* env, jobject obj, jlong reader_handle, jlongArray row_indices, jlong parallelism);

/**
 * @brief Destroy the reader
 *
 * @param env JNI environment
 * @param obj Java object
 * @param reader_handle Reader handle
 */
JNIEXPORT void JNICALL
Java_io_milvus_storage_Reader_readerDestroy(JNIEnv* env, jobject obj, jlong reader_handle);

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
JNIEXPORT jlongArray JNICALL
Java_io_milvus_storage_ChunkReader_getChunkIndices(JNIEnv* env, jobject obj, jlong chunk_reader_handle, jlongArray row_indices);

/**
 * @brief Get single chunk
 *
 * @param env JNI environment
 * @param obj Java object
 * @param chunk_reader_handle Chunk reader handle
 * @param chunk_index Chunk index
 * @return Arrow array pointer as long
 */
JNIEXPORT jlong JNICALL
Java_io_milvus_storage_ChunkReader_getChunk(JNIEnv* env, jobject obj, jlong chunk_reader_handle, jlong chunk_index);

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
JNIEXPORT jlongArray JNICALL
Java_io_milvus_storage_ChunkReader_getChunks(JNIEnv* env, jobject obj, jlong chunk_reader_handle, jlongArray chunk_indices, jlong parallelism);

/**
 * @brief Destroy chunk reader
 *
 * @param env JNI environment
 * @param obj Java object
 * @param chunk_reader_handle Chunk reader handle
 */
JNIEXPORT void JNICALL
Java_io_milvus_storage_ChunkReader_chunkReaderDestroy(JNIEnv* env, jobject obj, jlong chunk_reader_handle);

#ifdef __cplusplus
}
#endif

#endif  // MILVUS_STORAGE_FFI_JNI_H_