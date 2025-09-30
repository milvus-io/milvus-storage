package io.milvus.storage

/**
 * Scala wrapper for MilvusStorage ChunkReader
 * Provides functionality to read chunks from Milvus Storage
 */
class MilvusStorageChunkReader {
  // Ensure native library is loaded
  NativeLibraryLoader.loadLibrary()
  private var chunkReaderHandle: Long = 0
  private var isDestroyed: Boolean = false

  /**
   * Set the native handle (internal use only)
   * @param handle Native chunk reader handle
   */
  def setHandle(handle: Long): Unit = {
    if (isDestroyed) throw new IllegalStateException("ChunkReader has been destroyed")
    chunkReaderHandle = handle
  }

  /**
   * Get chunk indices for given row indices
   * @param rowIndices Array of row indices
   * @return Array of chunk indices
   */
  def getChunkIndicesScala(rowIndices: Array[Long]): Array[Long] = {
    if (isDestroyed) throw new IllegalStateException("ChunkReader has been destroyed")
    if (chunkReaderHandle == 0) throw new IllegalStateException("ChunkReader not initialized")
    getChunkIndices(chunkReaderHandle, rowIndices)
  }

  /**
   * Get a single chunk by index
   * @param chunkIndex Chunk index
   * @return Pointer to Arrow array
   */
  def getChunkScala(chunkIndex: Long): Long = {
    if (isDestroyed) throw new IllegalStateException("ChunkReader has been destroyed")
    if (chunkReaderHandle == 0) throw new IllegalStateException("ChunkReader not initialized")
    getChunk(chunkReaderHandle, chunkIndex)
  }

  /**
   * Get multiple chunks by indices
   * @param chunkIndices Array of chunk indices
   * @param parallelism Parallelism level
   * @return Array of pointers to Arrow arrays
   */
  def getChunksScala(chunkIndices: Array[Long], parallelism: Long): Array[Long] = {
    if (isDestroyed) throw new IllegalStateException("ChunkReader has been destroyed")
    if (chunkReaderHandle == 0) throw new IllegalStateException("ChunkReader not initialized")
    getChunks(chunkReaderHandle, chunkIndices, parallelism)
  }

  /**
   * Get multiple chunks with default parallelism
   * @param chunkIndices Array of chunk indices
   * @return Array of pointers to Arrow arrays
   */
  def getChunksScala(chunkIndices: Array[Long]): Array[Long] = {
    getChunksScala(chunkIndices, 1)
  }

  /**
   * Get the native handle (for internal use)
   */
  def getHandle: Long = {
    if (isDestroyed) throw new IllegalStateException("ChunkReader has been destroyed")
    chunkReaderHandle
  }

  /**
   * Check if chunk reader is valid
   */
  def isValid: Boolean = !isDestroyed && chunkReaderHandle != 0

  @native private def getChunkIndices(chunkReaderHandle: Long, rowIndices: Array[Long]): Array[Long]
  @native private def getChunk(chunkReaderHandle: Long, chunkIndex: Long): Long
  @native private def getChunks(chunkReaderHandle: Long, chunkIndices: Array[Long], parallelism: Long): Array[Long]
  @native private def chunkReaderDestroy(chunkReaderHandle: Long): Unit
}