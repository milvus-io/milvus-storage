package io.milvus.storage

import java.util.{Map => JMap}
import scala.collection.JavaConverters._

/**
 * Scala wrapper for MilvusStorage Properties
 * Provides configuration properties for Milvus Storage operations
 */
class MilvusStorageProperties {
  // Ensure native library is loaded
  NativeLibraryLoader.loadLibrary()
  private var propertiesPtr: Long = allocateProperties()
  private var isFreed: Boolean = false

  /**
   * Create properties from a Java Map
   */
  def create(properties: JMap[String, String]): Unit = {
    if (isFreed) throw new IllegalStateException("Properties have been freed")
    createProperties(properties, propertiesPtr)
  }

  /**
   * Create properties from a Scala Map
   */
  def create(properties: Map[String, String]): Unit = {
    create(properties.asJava)
  }

  /**
   * Free native resources
   */
  def free(): Unit = {
    if (propertiesPtr != 0 && !isFreed) {
      freeProperties(propertiesPtr)
      propertiesPtr = 0
      isFreed = true
    }
  }

  /**
   * Get the native pointer (for internal use)
   */
  def getPtr: Long = {
    if (isFreed) throw new IllegalStateException("Properties have been freed")
    propertiesPtr
  }

  /**
   * Check if properties are valid
   */
  def isValid: Boolean = !isFreed && propertiesPtr != 0

  @native private def allocateProperties(): Long
  @native private def createProperties(properties: JMap[String, String], propertiesPtr: Long): Unit
  @native private def freeProperties(propertiesPtr: Long): Unit
}