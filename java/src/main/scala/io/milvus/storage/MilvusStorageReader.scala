package io.milvus.storage

/**
 * Scala wrapper for MilvusStorage Reader
 * Provides functionality to read data from Milvus Storage
 */
class MilvusStorageReader {
  // Ensure native library is loaded
  NativeLibraryLoader.loadLibrary()
  private var readerHandle: Long = 0
  private var isDestroyed: Boolean = false

  /**
   * Create a new reader instance
   * @param columnGroups The column groups raw pointer
   * @param schemaPtr Pointer to Arrow schema
   * @param neededColumns Array of column names to read (can be null for all columns)
   * @param properties MilvusStorage properties
   */
  def create(columnGroups: Long, schemaPtr: Long, neededColumns: Array[String], properties: MilvusStorageProperties): Unit = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    readerHandle = readerNew(columnGroups, schemaPtr, neededColumns, properties.getPtr)
  }

  /**
   * Create a new reader instance with properties pointer
   * @param columnGroups The column groups raw pointer
   * @param schemaPtr Pointer to Arrow schema
   * @param neededColumns Array of column names to read (can be null for all columns)
   * @param propertiesPtr Pointer to properties
   */
  def create(columnGroups: Long, schemaPtr: Long, neededColumns: Array[String], propertiesPtr: Long): Unit = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    readerHandle = readerNew(columnGroups, schemaPtr, neededColumns, propertiesPtr)
  }

  /**
   * Open a per-batch record batch reader. Each call to [[readNextBatchScala]]
   * yields one batch exported as a fresh ArrowArray+ArrowSchema pair.
   *
   * Java callers must use this path rather than an ArrowArrayStream: Arrow
   * Java's C Data importer ignores `ArrowArray.offset`, so the shared-root
   * stream pattern duplicates data whenever the underlying C++ reader emits
   * `RecordBatch::Slice` results.
   *
   * @param predicate Filter predicate (can be null).
   * @return Handle to be passed to [[readNextBatchScala]] and eventually
   *         [[destroyRecordBatchReaderScala]].
   */
  def openRecordBatchReaderScala(predicate: String): Long = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    if (readerHandle == 0) throw new IllegalStateException("Reader not initialized")
    recordBatchReaderNew(readerHandle, predicate)
  }

  def openRecordBatchReaderScala(): Long = openRecordBatchReaderScala(null)

  /**
   * Read the next batch into caller-allocated Arrow C Data structs.
   *
   * @param rbrHandle  Handle from [[openRecordBatchReaderScala]].
   * @param arrayAddr  Memory address of a zero-initialized ArrowArray
   *                   struct (typically `ArrowArray.allocateNew(allocator).memoryAddress()`).
   * @param schemaAddr Memory address of a zero-initialized ArrowSchema struct.
   * @return true if a batch was written into the structs (caller must
   *         import/release), false on EOF.
   */
  def readNextBatchScala(rbrHandle: Long, arrayAddr: Long, schemaAddr: Long): Boolean = {
    recordBatchReaderReadNext(rbrHandle, arrayAddr, schemaAddr)
  }

  /** Returns the batches exported, columns materialized by concat, and a rough
   * estimate of copied bytes, in that order. The byte estimate counts only
   * top-level output buffers; it omits nested children and dictionaries and
   * is not an exact count of bytes copied or allocated. These diagnostics
   * should be removed with concat once JVM consumers correctly handle offsets.
   */
  def recordBatchReaderStatsScala(rbrHandle: Long): Array[Long] = {
    recordBatchReaderStats(rbrHandle)
  }

  /** Destroy a per-batch record batch reader handle. Safe with 0. */
  def destroyRecordBatchReaderScala(rbrHandle: Long): Unit = {
    if (rbrHandle != 0L) recordBatchReaderDestroy(rbrHandle)
  }

  /** Takes sorted, unique, nonnegative row indices into an owned batch reader.
   *
   * The result owns its batches independently of this reader. Consume it with
   * [[readNextBatchScala]] and release it with [[destroyRecordBatchReaderScala]],
   * including when consumption stops early. Null or empty columns use the
   * projection selected when this reader was created.
   */
  def takeRecordBatchReaderScala(
      rowIndices: Array[Long],
      parallelism: Long,
      neededColumns: Array[String] = null): Long = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    if (readerHandle == 0) throw new IllegalStateException("Reader not initialized")
    takeRecordBatchReader(readerHandle, rowIndices, parallelism, neededColumns)
  }

  /**
   * Get chunk reader for a specific column group
   * @param columnGroupId Column group ID
   * @param neededColumns Optional per-call column projection (null uses default from create)
   * @return Chunk reader handle
   */
  def getChunkReaderScala(columnGroupId: Long, neededColumns: Array[String] = null): MilvusStorageChunkReader = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    if (readerHandle == 0) throw new IllegalStateException("Reader not initialized")
    val chunkReaderHandle = getChunkReader(readerHandle, columnGroupId, neededColumns)
    val chunkReader = new MilvusStorageChunkReader()
    chunkReader.setHandle(chunkReaderHandle)
    chunkReader
  }

  /**
   * Take specific rows by indices
   * @param rowIndices Array of row indices to take
   * @param parallelism Parallelism level
   * @param neededColumns Optional per-call column projection (null uses default from create)
   * @return Addresses of ArrowArray structs sharing one native allocation.
   *         Release the complete result with [[freeTakeRowsScala]], not by
   *         freeing individual addresses. Prefer [[takeRecordBatchReaderScala]]
   *         when importing into Arrow Java, which requires offset-zero batches.
   */
  def takeRows(rowIndices: Array[Long], parallelism: Long, neededColumns: Array[String] = null): Array[Long] = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    if (readerHandle == 0) throw new IllegalStateException("Reader not initialized")
    take(readerHandle, rowIndices, parallelism, neededColumns)
  }

  /**
   * Take specific rows with default parallelism
   * @param rowIndices Array of row indices to take
   * @return Pointer to Arrow array
   */
  def takeRows(rowIndices: Array[Long]): Array[Long] = {
    takeRows(rowIndices, 1, null)
  }

  /** Releases all arrays and their shared allocation returned by [[takeRows]].
   * Call exactly once, after all imported arrays have transferred ownership or
   * been released. The addresses must be the unmodified result of one take call.
   */
  def freeTakeRowsScala(arrays: Array[Long]): Unit = {
    if (arrays != null && arrays.nonEmpty) freeTakeRows(arrays)
  }

  /**
   * Get the native handle (for internal use)
   */
  def getHandle: Long = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    readerHandle
  }

  /**
   * Destroy the reader and free resources
   */
  def destroy(): Unit = {
    if (readerHandle != 0 && !isDestroyed) {
      readerDestroy(readerHandle)
      readerHandle = 0
      isDestroyed = true
    }
  }

  /**
   * Check if reader is valid
   */
  def isValid: Boolean = !isDestroyed && readerHandle != 0

  @native private def readerNew(columnGroups: Long, schemaPtr: Long, neededColumns: Array[String], propertiesPtr: Long): Long
  @native private def getChunkReader(readerHandle: Long, columnGroupId: Long, neededColumns: Array[String]): Long
  @native private def take(readerHandle: Long, rowIndices: Array[Long], parallelism: Long, neededColumns: Array[String]): Array[Long]
  @native private def takeRecordBatchReader(readerHandle: Long, rowIndices: Array[Long], parallelism: Long, neededColumns: Array[String]): Long
  @native private def freeTakeRows(arrays: Array[Long]): Unit
  @native private def readerDestroy(readerHandle: Long): Unit

  // Per-batch record batch reader (see openRecordBatchReaderScala docstring).
  @native private def recordBatchReaderNew(readerHandle: Long, predicate: String): Long
  @native private def recordBatchReaderReadNext(rbrHandle: Long, arrayAddr: Long, schemaAddr: Long): Boolean
  @native private def recordBatchReaderStats(rbrHandle: Long): Array[Long]
  @native private def recordBatchReaderDestroy(rbrHandle: Long): Unit
}
