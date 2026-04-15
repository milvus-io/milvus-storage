package io.milvus.storage

/**
 * Scala wrapper for the milvus-storage `PackedRecordBatchWriter`.
 *
 * Mirrors the Go-side `internal/storagev2/packed/packed_writer.go` in milvus.
 * Use this when you need to write one parquet file per column group at
 * caller-controlled paths — e.g. the V2 backfill writer that emits
 * `{rootPath}/insert_log/{coll}/{part}/{seg}/{fieldId}/{logId}` per new field.
 *
 * Files produced by this writer carry milvus-storage's full KV metadata
 * (`row_group_metadata`, `storage_version`, `group_field_id_list`) so the
 * resulting parquet is byte-format-compatible with what Milvus's segcore
 * compaction worker would produce.
 */
class MilvusPackedWriter {
  // Ensure native library is loaded
  NativeLibraryLoader.loadLibrary()

  private var writerHandle: Long = 0
  private var isDestroyed: Boolean = false

  /**
   * Open a writer.
   *
   * @param paths         One output parquet path per column group.
   * @param columnGroups  For each group, the indices into the schema's fields
   *                      that belong in that group. Length must equal `paths`.
   * @param schemaPtr     Pointer to an exported Arrow `ArrowSchema` covering
   *                      ALL columns referenced by any group.
   * @param properties    LoonProperties carrying filesystem + storage config.
   * @param bufferSize    Max in-memory buffer in bytes (0 = native default).
   */
  def create(
      paths: Array[String],
      columnGroups: Array[Array[Int]],
      schemaPtr: Long,
      properties: MilvusStorageProperties,
      bufferSize: Long = 0L
  ): Unit = {
    if (isDestroyed) throw new IllegalStateException("Writer has been destroyed")
    if (writerHandle != 0)
      throw new IllegalStateException("Writer already created; call destroy() before reopening")
    if (paths == null || paths.isEmpty)
      throw new IllegalArgumentException("paths must be non-empty")
    if (columnGroups == null || columnGroups.length != paths.length)
      throw new IllegalArgumentException(
        s"columnGroups length (${if (columnGroups == null) 0 else columnGroups.length}) " +
          s"must match paths length (${paths.length})"
      )

    // CSR-flatten columnGroups → (offsets, indices) for the FFI call.
    val offsets = new Array[Int](columnGroups.length + 1)
    var total = 0
    var i = 0
    while (i < columnGroups.length) {
      val grp = columnGroups(i)
      if (grp == null)
        throw new IllegalArgumentException(s"columnGroups($i) is null")
      offsets(i) = total
      total += grp.length
      i += 1
    }
    offsets(columnGroups.length) = total

    val indices = new Array[Int](total)
    var pos = 0
    var g = 0
    while (g < columnGroups.length) {
      val grp = columnGroups(g)
      System.arraycopy(grp, 0, indices, pos, grp.length)
      pos += grp.length
      g += 1
    }

    writerHandle = writerNew(paths, offsets, indices, schemaPtr, properties.getPtr, bufferSize)
  }

  /**
   * Write one Arrow record batch (containing all schema columns). Internally
   * split per column group and buffered.
   */
  def write(arrayPtr: Long): Unit = {
    if (isDestroyed) throw new IllegalStateException("Writer has been destroyed")
    if (writerHandle == 0) throw new IllegalStateException("Writer not initialized")
    writerWrite(writerHandle, arrayPtr)
  }

  /**
   * Close the writer — flushes all buffered batches and finalizes every output
   * parquet file (including milvus-storage's KV metadata).
   */
  def close(): Unit = {
    if (isDestroyed) throw new IllegalStateException("Writer has been destroyed")
    if (writerHandle == 0) throw new IllegalStateException("Writer not initialized")
    writerClose(writerHandle)
  }

  /**
   * Release the native handle. Safe to call multiple times.
   */
  def destroy(): Unit = {
    if (writerHandle != 0 && !isDestroyed) {
      writerDestroy(writerHandle)
      writerHandle = 0
      isDestroyed = true
    }
  }

  def isValid: Boolean = !isDestroyed && writerHandle != 0

  @native private def writerNew(
      paths: Array[String],
      groupOffsets: Array[Int],
      groupIndices: Array[Int],
      schemaPtr: Long,
      propertiesPtr: Long,
      bufferSize: Long
  ): Long
  @native private def writerWrite(writerHandle: Long, arrayPtr: Long): Unit
  @native private def writerClose(writerHandle: Long): Unit
  @native private def writerDestroy(writerHandle: Long): Unit
}
