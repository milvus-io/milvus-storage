package io.milvus.storage

/**
 * Scala wrapper for MilvusStorage Writer
 * Provides functionality to write data to Milvus Storage
 */
class MilvusStorageWriter {
  // Ensure native library is loaded
  NativeLibraryLoader.loadLibrary()
  private var writerHandle: Long = 0
  private var isDestroyed: Boolean = false

  private def ensureUsable(): Unit = {
    if (isDestroyed) throw new IllegalStateException("Writer has been destroyed")
    if (writerHandle == 0) throw new IllegalStateException("Writer not initialized")
  }

  /**
   * Create a new writer instance
   * @param basePath The base path for storage
   * @param schemaPtr Pointer to Arrow schema
   * @param properties MilvusStorage properties
   */
  def create(basePath: String, schemaPtr: Long, properties: MilvusStorageProperties): Unit = {
    if (isDestroyed) throw new IllegalStateException("Writer has been destroyed")
    if (writerHandle != 0) throw new IllegalStateException("Writer already initialized")
    writerHandle = writerNew(basePath, schemaPtr, properties.getPtr)
  }

  /**
   * Create a new writer instance with properties pointer
   * @param basePath The base path for storage
   * @param schemaPtr Pointer to Arrow schema
   * @param propertiesPtr Pointer to properties
   */
  def create(basePath: String, schemaPtr: Long, propertiesPtr: Long): Unit = {
    if (isDestroyed) throw new IllegalStateException("Writer has been destroyed")
    if (writerHandle != 0) throw new IllegalStateException("Writer already initialized")
    writerHandle = writerNew(basePath, schemaPtr, propertiesPtr)
  }

  /**
   * Write an Arrow array to storage
   * @param arrayPtr Pointer to Arrow array
   */
  def write(arrayPtr: Long): Unit = {
    ensureUsable()
    writerWrite(writerHandle, arrayPtr)
  }

  /**
   * Flush any pending writes
   */
  def flush(): Unit = {
    ensureUsable()
    writerFlush(writerHandle)
  }

  /**
   * Close the writer and return column groups
   * @return The column groups raw pointer
   */
  def close(): Long = {
    ensureUsable()
    // try/finally, not a bare sequence: if the native close throws, the handle
    // must still be released. destroy() is what reaches loon_*_destroy, and
    // that is where an abandoned writer's storage -- an S3 multipart upload
    // whose parts no bucket listing shows -- gets released. Skipping it on the
    // failure path leaked exactly the case the abort exists for.
    try {
      writerClose(writerHandle)
    } finally {
      destroy()
    }
  }

  /**
   * Get the native handle (for internal use)
   */
  def getHandle: Long = {
    ensureUsable()
    writerHandle
  }

  /**
   * Destroy the writer and free resources
   */
  def destroy(): Unit = {
    if (writerHandle != 0 && !isDestroyed) {
      writerDestroy(writerHandle)
      writerHandle = 0
      isDestroyed = true
    }
  }

  /**
   * Check if writer is valid
   */
  def isValid: Boolean = !isDestroyed && writerHandle != 0

  @native private def writerNew(basePath: String, schemaPtr: Long, propertiesPtr: Long): Long
  @native private def writerWrite(writerHandle: Long, arrayPtr: Long): Unit
  @native private def writerFlush(writerHandle: Long): Unit
  @native private def writerClose(writerHandle: Long): Long
  @native private def writerDestroy(writerHandle: Long): Unit
}
