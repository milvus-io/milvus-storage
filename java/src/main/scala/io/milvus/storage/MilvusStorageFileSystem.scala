package io.milvus.storage

/** Metadata returned by the native filesystem without a separate lookup per file. */
final case class MilvusStorageFileInfo(path: String, isDirectory: Boolean, size: Long, modifiedNanos: Long)

/** Native filesystem operations. Properties are created by MilvusStorageProperties. */
private[storage] class MilvusStorageFileSystemNative {
  @native def create(propertiesPtr: Long, path: String): Long
  @native def destroy(handle: Long): Unit
  @native def readFileAll(handle: Long, path: String): Array[Byte]
  @native def writeFile(handle: Long, path: String, data: Array[Byte]): Unit
  @native def fileSize(handle: Long, path: String): Long
  @native def list(handle: Long, path: String, recursive: Boolean): Array[MilvusStorageFileInfo]
  @native def exists(handle: Long, path: String): Boolean
  @native def deleteFile(handle: Long, path: String): Unit
  @native def createDir(handle: Long, path: String, recursive: Boolean): Unit
  @native def openReader(handle: Long, path: String, fileSize: Long): Long
  @native def readerReadAt(handle: Long, offset: Long, length: Long): Array[Byte]
  @native def readerDestroy(handle: Long): Unit
}

/** Owns a native filesystem handle. Paths and storage configuration are interpreted in C. */
final class MilvusStorageFileSystem(properties: MilvusStorageProperties, path: String) extends AutoCloseable {
  NativeLibraryLoader.loadLibrary()
  require(properties != null, "properties must not be null")
  require(path != null, "path must not be null")
  private val native = new MilvusStorageFileSystemNative()
  private var handle = native.create(properties.getPtr, path)

  private def active: Long = {
    if (handle == 0L) throw new IllegalStateException("Filesystem is closed")
    handle
  }

  def readFileAll(path: String): Array[Byte] = native.readFileAll(active, path)
  def writeFile(path: String, data: Array[Byte]): Unit = native.writeFile(active, path, data)
  def fileSize(path: String): Long = native.fileSize(active, path)
  def list(path: String, recursive: Boolean): Array[MilvusStorageFileInfo] = native.list(active, path, recursive)
  def exists(path: String): Boolean = native.exists(active, path)
  def deleteFile(path: String): Unit = native.deleteFile(active, path)
  def createDir(path: String, recursive: Boolean): Unit = native.createDir(active, path, recursive)

  /** A negative fileSize means unknown; a known size avoids an additional metadata request. */
  def openReader(path: String, fileSize: Long = -1L): MilvusStorageFileReader = {
    require(fileSize >= -1L, "fileSize must be -1 or nonnegative")
    new MilvusStorageFileReader(native, native.openReader(active, path, fileSize))
  }

  override def close(): Unit = synchronized {
    if (handle != 0L) {
      native.destroy(handle)
      handle = 0L
    }
  }
}

/** Owns one random-access file reader. Close it before closing its filesystem. */
final class MilvusStorageFileReader private[storage] (
    native: MilvusStorageFileSystemNative,
    private var handle: Long
) extends AutoCloseable {
  def readAt(offset: Long, length: Long): Array[Byte] = {
    if (handle == 0L) throw new IllegalStateException("File reader is closed")
    require(offset >= 0L, "offset must be nonnegative")
    require(length >= 0L && length <= Int.MaxValue, "length must fit a JVM byte array")
    require(offset <= Long.MaxValue - length, "read range overflows")
    native.readerReadAt(handle, offset, length)
  }

  override def close(): Unit = synchronized {
    if (handle != 0L) {
      native.readerDestroy(handle)
      handle = 0L
    }
  }
}
