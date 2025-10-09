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
   * @param manifest The manifest string
   * @param schemaPtr Pointer to Arrow schema
   * @param neededColumns Array of column names to read
   * @param properties MilvusStorage properties
   */
  def create(manifest: String, schemaPtr: Long, neededColumns: Array[String], properties: MilvusStorageProperties): Unit = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    readerHandle = readerNew(manifest, schemaPtr, neededColumns, properties.getPtr)
  }

  /**
   * Create a new reader instance with properties pointer
   * @param manifest The manifest string
   * @param schemaPtr Pointer to Arrow schema
   * @param neededColumns Array of column names to read
   * @param propertiesPtr Pointer to properties
   */
  def create(manifest: String, schemaPtr: Long, neededColumns: Array[String], propertiesPtr: Long): Unit = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    readerHandle = readerNew(manifest, schemaPtr, neededColumns, propertiesPtr)
  }

  /**
   * Get record batch reader
   * @param predicate Filter predicate (can be null)
   * @param batchSize Batch size for reading
   * @param bufferSize Buffer size
   * @return Pointer to Arrow stream
   */
  def getRecordBatchReaderScala(predicate: String, batchSize: Long, bufferSize: Long): Long = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    if (readerHandle == 0) throw new IllegalStateException("Reader not initialized")
    getRecordBatchReader(readerHandle, predicate, batchSize, bufferSize)
  }

  /**
   * Get record batch reader with default parameters
   * @return Pointer to Arrow stream
   */
  def getRecordBatchReaderScala(): Long = {
    getRecordBatchReaderScala(null, 1024, 4096)
  }

  /**
   * Get chunk reader for a specific column group
   * @param columnGroupId Column group ID
   * @return Chunk reader handle
   */
  def getChunkReaderScala(columnGroupId: Long): MilvusStorageChunkReader = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    if (readerHandle == 0) throw new IllegalStateException("Reader not initialized")
    val chunkReaderHandle = getChunkReader(readerHandle, columnGroupId)
    val chunkReader = new MilvusStorageChunkReader()
    chunkReader.setHandle(chunkReaderHandle)
    chunkReader
  }

  /**
   * Take specific rows by indices
   * @param rowIndices Array of row indices to take
   * @param parallelism Parallelism level
   * @return Pointer to Arrow array
   */
  def takeRows(rowIndices: Array[Long], parallelism: Long): Long = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    if (readerHandle == 0) throw new IllegalStateException("Reader not initialized")
    take(readerHandle, rowIndices, parallelism)
  }

  /**
   * Take specific rows with default parallelism
   * @param rowIndices Array of row indices to take
   * @return Pointer to Arrow array
   */
  def takeRows(rowIndices: Array[Long]): Long = {
    takeRows(rowIndices, 1)
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

  @native private def readerNew(manifest: String, schemaPtr: Long, neededColumns: Array[String], propertiesPtr: Long): Long
  @native private def getRecordBatchReader(readerHandle: Long, predicate: String, batchSize: Long, bufferSize: Long): Long
  @native private def getChunkReader(readerHandle: Long, columnGroupId: Long): Long
  @native private def take(readerHandle: Long, rowIndices: Array[Long], parallelism: Long): Long
  @native private def readerDestroy(readerHandle: Long): Unit
}