package io.milvus.storage

/**
 * Scala wrapper for MilvusStorage Manifest operations
 * Provides functionality to retrieve manifest information
 */
class MilvusStorageManifestNative {
  /**
   * Get latest column groups from manifest
   * @param basePath The base path for storage
   * @param propertiesPtr Pointer to properties
   * @return Array of [columnGroupsPtr, readVersion]
   */
  @native def getLatestColumnGroups(basePath: String, propertiesPtr: Long): Array[Long]

  /**
   * Get column groups from manifest at a specific version
   * @param basePath The base path for storage
   * @param propertiesPtr Pointer to properties
   * @param readVersion Version to read (-1 for latest, >0 for specific version)
   * @return Array of [columnGroupsPtr, actualReadVersion]
   */
  @native def getColumnGroupsWithVersion(basePath: String, propertiesPtr: Long, readVersion: Long): Array[Long]
}

/**
 * Result of getting latest column groups from manifest
 * @param columnGroupsPtr Pointer to column groups
 * @param readVersion The manifest version read (0 means no manifest exists)
 */
case class LatestColumnGroupsResult(columnGroupsPtr: Long, readVersion: Long)

object MilvusStorageManifest {
  // Ensure native library is loaded
  NativeLibraryLoader.loadLibrary()

  private val milvusStorageManifestNative = new MilvusStorageManifestNative()


  /**
   * Get latest column groups from manifest
   * @param basePath The base path for storage
   * @param properties MilvusStorage properties
   * @return LatestColumnGroupsResult containing columnGroupsPtr and readVersion
   */
  def getLatestColumnGroupsScala(basePath: String, properties: MilvusStorageProperties): LatestColumnGroupsResult = {
    val result = milvusStorageManifestNative.getLatestColumnGroups(basePath, properties.getPtr)
    LatestColumnGroupsResult(result(0), result(1))
  }

  /**
   * Get latest column groups from manifest with properties pointer
   * @param basePath The base path for storage
   * @param propertiesPtr Pointer to properties
   * @return LatestColumnGroupsResult containing columnGroupsPtr and readVersion
   */
  def getLatestColumnGroupsScala(basePath: String, propertiesPtr: Long): LatestColumnGroupsResult = {
    val result = milvusStorageManifestNative.getLatestColumnGroups(basePath, propertiesPtr)
    LatestColumnGroupsResult(result(0), result(1))
  }

  /**
   * Get column groups from manifest at a specific version
   * @param basePath The base path for storage
   * @param properties MilvusStorage properties
   * @param readVersion Version to read (-1 for latest, >0 for specific version)
   * @return LatestColumnGroupsResult containing columnGroupsPtr and actualReadVersion
   */
  def getColumnGroupsScala(basePath: String, properties: MilvusStorageProperties, readVersion: Long): LatestColumnGroupsResult = {
    val result = milvusStorageManifestNative.getColumnGroupsWithVersion(basePath, properties.getPtr, readVersion)
    LatestColumnGroupsResult(result(0), result(1))
  }
}
