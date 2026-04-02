package io.milvus.storage

/**
 * Scala wrapper for MilvusStorage SegmentReader.
 *
 * Reads parquet data with automatic LOBRef → TEXT resolution.
 * TEXT columns are returned as utf8 strings (LOBReferences auto-decoded).
 *
 * Usage:
 * {{{
 *   val reader = new MilvusSegmentReader()
 *   reader.open(segmentPath, version, schemaPtr, neededColumns, lobColumns, propertiesPtr)
 *   val streamPtr = reader.getStream()  // ArrowArrayStream with TEXT auto-decoded
 *   // use ArrowUtils to read batches from streamPtr
 *   reader.destroy()
 * }}}
 */
class MilvusSegmentReader {
  NativeLibraryLoader.loadLibrary()
  private var handle: Long = 0
  private var isDestroyed: Boolean = false

  /**
   * Open a SegmentReader from manifest.
   *
   * @param segmentPath   Base path where manifest and data files are stored
   * @param version       Manifest version to read (-1 = latest)
   * @param schemaPtr     Pointer to Arrow schema (TEXT columns as utf8)
   * @param neededColumns Column names to read (null = all columns)
   * @param lobColumns    TEXT column configurations for LOB resolution
   * @param propertiesPtr Pointer to storage properties
   */
  def open(segmentPath: String, version: Long, schemaPtr: Long,
           neededColumns: Array[String], lobColumns: Seq[LobColumnConfig],
           propertiesPtr: Long): Unit = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")

    val fieldIds = lobColumns.map(_.fieldId).toArray
    val basePaths = lobColumns.map(_.lobBasePath).toArray
    val inlines = lobColumns.map(_.inlineThreshold).toArray
    val maxFiles = lobColumns.map(_.maxLobFileBytes).toArray
    val flushes = lobColumns.map(_.flushThresholdBytes).toArray

    handle = segmentReaderOpen(segmentPath, version, schemaPtr, neededColumns,
      if (fieldIds.isEmpty) null else fieldIds,
      if (basePaths.isEmpty) null else basePaths,
      if (inlines.isEmpty) null else inlines,
      if (maxFiles.isEmpty) null else maxFiles,
      if (flushes.isEmpty) null else flushes,
      propertiesPtr)
  }

  /**
   * Get an ArrowArrayStream from the reader.
   * TEXT columns are automatically decoded from LOBRef to utf8 strings.
   *
   * @return Pointer to ArrowArrayStream (use ArrowUtilsNative to read batches)
   */
  def getStream(): Long = {
    if (isDestroyed) throw new IllegalStateException("Reader has been destroyed")
    if (handle == 0) throw new IllegalStateException("Reader not initialized")
    segmentReaderGetStream(handle)
  }

  def destroy(): Unit = {
    if (handle != 0 && !isDestroyed) {
      segmentReaderDestroy(handle)
      handle = 0
      isDestroyed = true
    }
  }

  def isValid: Boolean = !isDestroyed && handle != 0

  @native private def segmentReaderOpen(
      segmentPath: String, version: Long, schemaPtr: Long,
      neededColumns: Array[String],
      lobFieldIds: Array[Long], lobBasePaths: Array[String],
      lobInlineThresholds: Array[Long], lobMaxFileBytes: Array[Long], lobFlushThresholds: Array[Long],
      propertiesPtr: Long): Long

  @native private def segmentReaderGetStream(handle: Long): Long
  @native private def segmentReaderDestroy(handle: Long): Unit
}
