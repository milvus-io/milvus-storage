package io.milvus.storage

/**
 * Scala wrapper for MilvusStorage Manifest operations
 * Provides functionality to retrieve manifest information
 */
class MilvusStorageManifestNative {
  @native def getLatestColumnGroups(basePath: String, propertiesPtr: Long): Long
}

object MilvusStorageManifest {
  // Ensure native library is loaded
  NativeLibraryLoader.loadLibrary()

  private val milvusStorageManifestNative = new MilvusStorageManifestNative()


  /**
   * Get latest column groups from manifest
   * @param basePath The base path for storage
   * @param properties MilvusStorage properties
   * @return Column groups as raw pointer
   */
  def getLatestColumnGroupsScala(basePath: String, properties: MilvusStorageProperties): Long = {
    milvusStorageManifestNative.getLatestColumnGroups(basePath, properties.getPtr)
  }

  /**
   * Get latest column groups from manifest with properties pointer
   * @param basePath The base path for storage
   * @param propertiesPtr Pointer to properties
   * @return Column groups as raw pointer
   */
  def getLatestColumnGroupsScala(basePath: String, propertiesPtr: Long): Long = {
    milvusStorageManifestNative.getLatestColumnGroups(basePath, propertiesPtr)
  }
}
