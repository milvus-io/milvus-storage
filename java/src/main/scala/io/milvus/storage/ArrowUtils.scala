package io.milvus.storage

import org.apache.arrow.c.{ArrowArray, ArrowSchema}
import org.apache.arrow.memory.RootAllocator

/**
 * Arrow utilities for working with Arrow C Data Interface
 * Provides core functionality for releasing Arrow structures and reading streams
 */
class ArrowUtilsNative {
  @native def readNextBatch(streamPtr: Long): Long
  @native def releaseArrowSchema(schemaPtr: Long, freePtr: Boolean): Unit
  @native def releaseArrowArray(arrayPtr: Long, freePtr: Boolean): Unit
  @native def releaseArrowStream(streamPtr: Long, freePtr: Boolean): Unit
}

object ArrowUtils {
  private val arrowUtilsNative = new ArrowUtilsNative()

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
  def releaseArrowSchema(schemaPtr: Long, freePtr: Boolean): Unit = {
    arrowUtilsNative.releaseArrowSchema(schemaPtr, freePtr);
  }

  /**
   * Release Arrow array
   * @param arrayPtr Memory address of ArrowArray
   */
  def releaseArrowArray(arrayPtr: Long, freePtr: Boolean): Unit = {
    arrowUtilsNative.releaseArrowArray(arrayPtr, freePtr);
  }

  /**
   * Release Arrow stream (record batch reader)
   * @param streamPtr Pointer to ArrowArrayStream
   */
  def releaseArrowStream(streamPtr: Long, freePtr: Boolean): Unit = {
    arrowUtilsNative.releaseArrowStream(streamPtr, freePtr)
  }

  /**
   * Read next batch from Arrow stream
   * @param streamPtr Pointer to ArrowArrayStream
   * @return Address of ArrowArray (0L if end of stream)
   */
  def readNextBatch(streamPtr: Long): Long = {
    arrowUtilsNative.readNextBatch(streamPtr)
  }

}
