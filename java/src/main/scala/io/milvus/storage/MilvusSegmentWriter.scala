package io.milvus.storage

/**
 * Configuration for a TEXT (LOB) column.
 *
 * @param fieldId         Field ID of the TEXT column
 * @param lobBasePath     Base path for LOB files: {partition_path}/lobs/{field_id}
 * @param inlineThreshold Texts smaller than this are stored inline (default 256 bytes)
 * @param maxLobFileBytes Maximum bytes per LOB file before rolling (default 64MB)
 * @param flushThresholdBytes Flush buffer when it exceeds this size (default 16MB)
 */
case class LobColumnConfig(
    fieldId: Long,
    lobBasePath: String,
    inlineThreshold: Long = 256,
    maxLobFileBytes: Long = 64 * 1024 * 1024,
    flushThresholdBytes: Long = 16 * 1024 * 1024
)

/**
 * LOB file metadata returned by SegmentWriter.close().
 */
case class LobFileResult(
    fieldId: Long,
    path: String,
    totalRows: Long,
    validRows: Long,
    fileSizeBytes: Long
)

/**
 * Output of SegmentWriter.close().
 * Contains columnGroupsPtr (for Transaction.commit) and LOB file metadata.
 */
case class SegmentWriteResult(
    columnGroupsPtr: Long,
    lobFiles: Seq[LobFileResult],
    rowsWritten: Long
)

/**
 * Scala wrapper for MilvusStorage SegmentWriter.
 *
 * Writes parquet data + LOB files for TEXT columns.
 * Close() returns ColumnGroups + LOB file info — caller commits via Transaction.
 *
 * Usage:
 * {{{
 *   val writer = new MilvusSegmentWriter()
 *   writer.create(schemaPtr, segmentPath, lobColumns, propertiesPtr)
 *   writer.write(arrayPtr)
 *   val result = writer.close()  // returns ColumnGroups + LobFiles
 *
 *   // commit via Transaction
 *   val txn = new MilvusStorageTransaction()
 *   txn.begin(basePath, propertiesPtr)
 *   txn.commit(0, 0, result.columnGroupsPtr)  // also need to add LOB files
 *   writer.destroy()
 * }}}
 */
class MilvusSegmentWriter {
  NativeLibraryLoader.loadLibrary()
  private var handle: Long = 0
  private var isDestroyed: Boolean = false

  def create(schemaPtr: Long, segmentPath: String,
             lobColumns: Seq[LobColumnConfig], propertiesPtr: Long): Unit = {
    if (isDestroyed) throw new IllegalStateException("Writer has been destroyed")

    val fieldIds = lobColumns.map(_.fieldId).toArray
    val basePaths = lobColumns.map(_.lobBasePath).toArray
    val inlines = lobColumns.map(_.inlineThreshold).toArray
    val maxFiles = lobColumns.map(_.maxLobFileBytes).toArray
    val flushes = lobColumns.map(_.flushThresholdBytes).toArray

    handle = segmentWriterNew(schemaPtr, segmentPath,
      if (fieldIds.isEmpty) null else fieldIds,
      if (basePaths.isEmpty) null else basePaths,
      if (inlines.isEmpty) null else inlines,
      if (maxFiles.isEmpty) null else maxFiles,
      if (flushes.isEmpty) null else flushes,
      propertiesPtr)
  }

  def write(arrayPtr: Long): Unit = {
    if (isDestroyed) throw new IllegalStateException("Writer has been destroyed")
    if (handle == 0) throw new IllegalStateException("Writer not initialized")
    segmentWriterWrite(handle, arrayPtr)
  }

  def flush(): Unit = {
    if (isDestroyed) throw new IllegalStateException("Writer has been destroyed")
    if (handle == 0) throw new IllegalStateException("Writer not initialized")
    segmentWriterFlush(handle)
  }

  /**
   * Close the writer and return write output.
   * Returns ColumnGroups pointer + LOB file metadata.
   * Caller must commit via MilvusStorageTransaction.
   */
  def close(): SegmentWriteResult = {
    if (isDestroyed) throw new IllegalStateException("Writer has been destroyed")
    if (handle == 0) throw new IllegalStateException("Writer not initialized")

    // Pre-allocate output arrays for LOB files (max reasonable size)
    val maxLobFiles = 64
    val lobFieldIds = new Array[Long](maxLobFiles)
    val lobPaths = new Array[String](maxLobFiles)
    val lobTotalRows = new Array[Long](maxLobFiles)
    val lobValidRows = new Array[Long](maxLobFiles)
    val lobFileSizes = new Array[Long](maxLobFiles)

    val columnGroupsPtr = segmentWriterClose(handle, lobFieldIds, lobPaths,
      lobTotalRows, lobValidRows, lobFileSizes)

    // collect non-null LOB file results
    val lobFiles = lobPaths.zipWithIndex.takeWhile(_._1 != null).map { case (path, i) =>
      LobFileResult(lobFieldIds(i), path, lobTotalRows(i), lobValidRows(i), lobFileSizes(i))
    }.toSeq

    SegmentWriteResult(columnGroupsPtr, lobFiles, 0)
  }

  def abort(): Unit = {
    if (isDestroyed) throw new IllegalStateException("Writer has been destroyed")
    if (handle == 0) throw new IllegalStateException("Writer not initialized")
    segmentWriterAbort(handle)
  }

  def destroy(): Unit = {
    if (handle != 0 && !isDestroyed) {
      segmentWriterDestroy(handle)
      handle = 0
      isDestroyed = true
    }
  }

  def isValid: Boolean = !isDestroyed && handle != 0

  @native private def segmentWriterNew(
      schemaPtr: Long, segmentPath: String,
      lobFieldIds: Array[Long], lobBasePaths: Array[String],
      lobInlineThresholds: Array[Long], lobMaxFileBytes: Array[Long], lobFlushThresholds: Array[Long],
      propertiesPtr: Long): Long

  @native private def segmentWriterWrite(handle: Long, arrayPtr: Long): Unit
  @native private def segmentWriterFlush(handle: Long): Unit
  @native private def segmentWriterClose(handle: Long,
      outLobFieldIds: Array[Long], outLobPaths: Array[String],
      outLobTotalRows: Array[Long], outLobValidRows: Array[Long], outLobFileSizes: Array[Long]): Long
  @native private def segmentWriterAbort(handle: Long): Unit
  @native private def segmentWriterDestroy(handle: Long): Unit
}
