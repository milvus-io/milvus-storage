package io.milvus.storage

import java.nio.file.FileAlreadyExistsException
import java.util.{Map => JMap}
import scala.jdk.CollectionConverters._

/** Metadata returned by the native filesystem without a separate lookup per file. */
final case class MilvusStorageFileInfo(path: String, isDirectory: Boolean, size: Long, modifiedNanos: Long)

/** Native filesystem operations. Properties are created by MilvusStorageProperties. */
private[storage] class MilvusStorageFileSystemNative {
  @native def create(propertiesPtr: Long, path: String): Long
  @native def destroy(handle: Long): Unit
  @native def readFileAll(handle: Long, path: String): Array[Byte]
  @deprecated("Use openWriter and the writer APIs instead.", "0.1.0-SNAPSHOT")
  @native def writeFile(handle: Long, path: String, data: Array[Byte]): Unit
  @native def fileSize(handle: Long, path: String): Long
  @native def list(handle: Long, path: String, recursive: Boolean): Array[MilvusStorageFileInfo]
  @native def exists(handle: Long, path: String): Boolean
  @native def deleteFile(handle: Long, path: String): Unit
  @native def createDir(handle: Long, path: String, recursive: Boolean): Unit
  @native def openReader(handle: Long, path: String, fileSize: Long): Long
  @native def readerReadAt(handle: Long, offset: Long, length: Long): Array[Byte]
  @native def readerDestroy(handle: Long): Unit
  @native def openWriter(handle: Long, path: String, metadataKeys: Array[String],
                         metadataValues: Array[String], conditional: Boolean): Long
  @native def writerWrite(handle: Long, data: Array[Byte]): Unit
  @native def writerFlush(handle: Long): Unit
  @native def writerClose(handle: Long): Unit
  @native def writerDestroy(handle: Long): Unit
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

  /** Opens a writer, optionally requiring that the object does not already exist.
    *
    * Conditional write conflicts can surface when opening, writing, flushing, or closing.
    * Conflicts retain their native error code in MilvusStorageException; they do not imply
    * that the destination exists and can require starting a new upload.
    * The local filesystem uses an in-process mutex for testing only; it does not provide
    * atomic exclusion across processes.
    *
    * @param metadata backend metadata passed to the C FFI; an empty map supplies no metadata
    * @throws FileAlreadyExistsException if the backend reports a precondition failure
    * @throws UnsupportedOperationException if conditional uploads are not supported
    * @throws MilvusStorageException for other native failures, retaining their code and message
    */
  @throws[FileAlreadyExistsException]
  def openWriter(path: String, metadata: Map[String, String] = Map.empty,
                 conditional: Boolean = false): MilvusStorageFileWriter = {
    require(metadata != null, "metadata must not be null")
    val entries = metadata.toArray
    new MilvusStorageFileWriter(native,
      native.openWriter(active, path, entries.map(_._1), entries.map(_._2), conditional))
  }

  /** Opens a writer with metadata supplied by a Java caller. */
  @throws[FileAlreadyExistsException]
  def openWriter(path: String, metadata: JMap[String, String], conditional: Boolean): MilvusStorageFileWriter = {
    require(metadata != null, "metadata must not be null")
    openWriter(path, metadata.asScala.toMap, conditional)
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

/** Owns one file writer. Close it before closing its filesystem. */
final class MilvusStorageFileWriter private[storage] (
    native: MilvusStorageFileSystemNative,
    private var handle: Long
) extends AutoCloseable {
  private def active: Long = {
    if (handle == 0L) throw new IllegalStateException("File writer is closed")
    handle
  }

  /** Writes a nonempty byte array, as required by the C FFI. */
  @throws[FileAlreadyExistsException]
  def write(data: Array[Byte]): Unit = native.writerWrite(active, data)
  @throws[FileAlreadyExistsException]
  def flush(): Unit = native.writerFlush(active)

  /** Finalizes the write and releases the handle, including when finalization fails. */
  @throws[FileAlreadyExistsException]
  override def close(): Unit = synchronized {
    if (handle != 0L) {
      val writer = handle
      handle = 0L
      try native.writerClose(writer)
      finally native.writerDestroy(writer)
    }
  }
}
