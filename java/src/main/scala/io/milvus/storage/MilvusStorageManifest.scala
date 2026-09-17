package io.milvus.storage

/** Native results contain [owned manifest pointer, read version, borrowed column groups pointer]. */
class MilvusStorageManifestNative {
  @native def getLatestColumnGroups(basePath: String, propertiesPtr: Long): Array[Long]
  @native def getColumnGroupsWithVersion(basePath: String, propertiesPtr: Long, readVersion: Long): Array[Long]
  @native def destroyManifest(manifestPtr: Long): Unit
  @native def columnGroups(manifestPtr: Long): Long
}

/** The manifest owns its column groups, deltas, statistics, LOB metadata and indexes. */
final class MilvusStorageManifestHandle private[storage] (
    private var manifestPtr: Long,
    private val groupsPtr: Long,
    val readVersion: Long
) extends AutoCloseable {
  def columnGroupsPtr: Long = {
    if (manifestPtr == 0L) throw new IllegalStateException("Manifest is closed")
    groupsPtr
  }

  override def close(): Unit = synchronized {
    if (manifestPtr != 0L) {
      MilvusStorageManifest.destroyManifest(manifestPtr)
      manifestPtr = 0L
    }
  }
}

/** Column groups are borrowed from the shared owner; close this result after its readers. */
case class LatestColumnGroupsResult(
    columnGroupsPtr: Long,
    readVersion: Long,
    private val owner: MilvusStorageManifestHandle = null
) extends AutoCloseable {
  override def close(): Unit = if (owner != null) owner.close()
}

object MilvusStorageManifest {
  NativeLibraryLoader.loadLibrary()
  private val native = new MilvusStorageManifestNative()

  def open(basePath: String, properties: MilvusStorageProperties, readVersion: Long = -1L): MilvusStorageManifestHandle = {
    val result = native.getColumnGroupsWithVersion(basePath, properties.getPtr, readVersion)
    try new MilvusStorageManifestHandle(result(0), result(2), result(1))
    catch {
      case failure: Throwable => releaseFailedManifest(result(0), failure)
    }
  }

  def getLatestColumnGroupsScala(basePath: String, properties: MilvusStorageProperties): LatestColumnGroupsResult =
    getLatestColumnGroupsScala(basePath, properties.getPtr)

  def getLatestColumnGroupsScala(basePath: String, propertiesPtr: Long): LatestColumnGroupsResult = {
    val result = native.getLatestColumnGroups(basePath, propertiesPtr)
    columnGroupsResult(result)
  }

  def getColumnGroupsScala(basePath: String, properties: MilvusStorageProperties, readVersion: Long): LatestColumnGroupsResult = {
    val result = native.getColumnGroupsWithVersion(basePath, properties.getPtr, readVersion)
    columnGroupsResult(result)
  }

  private def columnGroupsResult(result: Array[Long]): LatestColumnGroupsResult = {
    try LatestColumnGroupsResult(result(2), result(1), new MilvusStorageManifestHandle(result(0), result(2), result(1)))
    catch {
      case failure: Throwable => releaseFailedManifest(result(0), failure)
    }
  }

  private def releaseFailedManifest(manifestPtr: Long, failure: Throwable): Nothing = {
    try destroyManifest(manifestPtr)
    catch {
      case cleanupFailure: Throwable => failure.addSuppressed(cleanupFailure)
    }
    throw failure
  }

  /** Release the complete owned manifest; never pass a separately allocated column-groups pointer. */
  def destroyManifest(manifestPtr: Long): Unit = {
    if (manifestPtr != 0L) native.destroyManifest(manifestPtr)
  }

  private[storage] def borrowedColumnGroups(manifestPtr: Long): Long = native.columnGroups(manifestPtr)
}
