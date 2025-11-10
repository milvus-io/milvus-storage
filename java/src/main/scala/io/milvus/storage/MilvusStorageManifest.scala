package io.milvus.storage

/**
 * Scala wrapper for MilvusStorage Manifest operations
 * Provides functionality to retrieve manifest information
 */
object MilvusStorageManifest {
  // Ensure native library is loaded
  NativeLibraryLoader.loadLibrary()

  /**
   * Get latest column groups from manifest
   * @param basePath The base path for storage
   * @param properties MilvusStorage properties
   * @return Column groups as JSON string
   */
  def getLatestColumnGroups(basePath: String, properties: MilvusStorageProperties): String = {
    getLatestColumnGroups(basePath, properties.getPtr)
  }

  /**
   * Get latest column groups from manifest with properties pointer
   * @param basePath The base path for storage
   * @param propertiesPtr Pointer to properties
   * @return Column groups as JSON string
   */
  def getLatestColumnGroups(basePath: String, propertiesPtr: Long): String = {
    getLatestColumnGroupsNative(basePath, propertiesPtr)
  }

  @native private def getLatestColumnGroupsNative(basePath: String, propertiesPtr: Long): String
}
