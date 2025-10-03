package io.milvus.storage

import org.apache.arrow.c.{ArrowArray, ArrowSchema}
import org.apache.arrow.memory.RootAllocator

/**
 * Arrow utilities for working with Arrow C Data Interface
 * Provides core functionality for releasing Arrow structures and reading streams
 */
object ArrowUtils {

  // Ensure native library is loaded
  NativeLibraryLoader.loadLibrary()

  // Create a root allocator for Arrow memory management
  private val allocator = new RootAllocator(Long.MaxValue)

  /**
   * Get the allocator instance
   */
  def getAllocator: RootAllocator = allocator

  /**
   * Release Arrow schema
   * @param schemaPtr Memory address of ArrowSchema
   */
  def releaseArrowSchema(schemaPtr: Long): Unit = {
    if (schemaPtr != 0) {
      val arrowSchema = ArrowSchema.wrap(schemaPtr)
      arrowSchema.close()
    }
  }

  /**
   * Release Arrow array
   * @param arrayPtr Memory address of ArrowArray
   */
  def releaseArrowArray(arrayPtr: Long): Unit = {
    if (arrayPtr != 0) {
      val arrowArray = ArrowArray.wrap(arrayPtr)
      arrowArray.close()
    }
  }

  /**
   * Release Arrow stream (record batch reader)
   * @param streamPtr Pointer to ArrowArrayStream
   */
  @native def releaseArrowStream(streamPtr: Long): Unit

  /**
   * Read next batch from Arrow stream
   * @param streamPtr Pointer to ArrowArrayStream
   * @return Address of ArrowArray (0L if end of stream)
   */
  @native def readNextBatch(streamPtr: Long): Long
}
