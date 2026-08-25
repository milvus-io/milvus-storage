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
    if (segmentPath == null) throw new IllegalArgumentException("segmentPath must not be null")
    if (segmentPath.isEmpty) throw new IllegalArgumentException("segmentPath must not be empty")
    if (version < -1) throw new IllegalArgumentException("version must be -1 or greater")
    if (schemaPtr == 0) throw new IllegalArgumentException("schemaPtr must not be 0")
    if (propertiesPtr == 0) throw new IllegalArgumentException("propertiesPtr must not be 0")
    if (neededColumns != null && neededColumns.exists(_ == null)) {
      throw new IllegalArgumentException("neededColumns must not contain null")
    }
    if (neededColumns != null && neededColumns.exists(_.isEmpty)) {
      throw new IllegalArgumentException("neededColumns must not contain empty names")
    }
    if (lobColumns == null) throw new IllegalArgumentException("lobColumns must not be null")
    if (lobColumns.exists(_ == null)) throw new IllegalArgumentException("lobColumns must not contain null")
    lobColumns.foreach { config =>
      if (config.fieldId < 0) throw new IllegalArgumentException("LOB fieldId must be greater than or equal to 0")
      if (config.lobBasePath == null) throw new IllegalArgumentException("LOB lobBasePath must not be null")
      if (config.lobBasePath.isEmpty) throw new IllegalArgumentException("LOB lobBasePath must not be empty")
      if (config.inlineThreshold <= 0) throw new IllegalArgumentException("LOB inlineThreshold must be > 0")
      if (config.maxLobFileBytes <= 0) throw new IllegalArgumentException("LOB maxLobFileBytes must be > 0")
      if (config.flushThresholdBytes <= 0) throw new IllegalArgumentException("LOB flushThresholdBytes must be > 0")
      if (config.flushThresholdBytes > config.maxLobFileBytes) {
        throw new IllegalArgumentException("LOB flushThresholdBytes must not exceed maxLobFileBytes")
      }
    }

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
