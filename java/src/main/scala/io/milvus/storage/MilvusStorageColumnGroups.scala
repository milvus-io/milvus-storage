package io.milvus.storage

/** Native bridge — do not call directly. */
class MilvusStorageColumnGroupsNative {
  @native def createFromGroups(
      columnsPerGroup: Array[Array[String]],
      filesPerGroup: Array[Array[String]],
      fileRowCountsPerGroup: Array[Array[Long]]
  ): Long
  @native def destroy(columnGroupsPtr: Long): Unit
  @native def count(columnGroupsPtr: Long): Int
  @native def columns(columnGroupsPtr: Long, groupIndex: Int): Array[String]
  @native def files(columnGroupsPtr: Long, groupIndex: Int): Array[String]
  @native def fileRowCounts(columnGroupsPtr: Long, groupIndex: Int): Array[Long]
  @native def format(columnGroupsPtr: Long, groupIndex: Int): String
}

/** Inspect column groups returned by a V3 writer or borrowed from a manifest.
  *
  * Writer results are owned and must be released with [[destroy]]. Manifest
  * column groups are borrowed: keep the manifest open while accessing them,
  * then close the manifest instead of destroying its column groups separately.
  */
object MilvusStorageColumnGroups {
  NativeLibraryLoader.loadLibrary()
  private val native = new MilvusStorageColumnGroupsNative()

  /** Construct a LoonColumnGroups from per-group column names, file paths and
    * per-file row counts.
    *
    * Legacy Storage V2 helper for Milvus 2.6 backfill. Spark readers should
    * obtain V3 column groups through [[MilvusStorageManifest.open]] instead.
    * V2 files are always parquet and have no format option or manifest.
    *
    * @param columnsPerGroup
    *   `columnsPerGroup(i)` lists column names for group `i`. For milvus-storage
    *   parquet files these are the field-ID-as-string (e.g. `Array("100","0","1")`).
    * @param filesPerGroup
    *   `filesPerGroup(i)` lists file paths for group `i`, ordered by row
    *   offset (lower `log_id` first).
    * @param fileRowCountsPerGroup
    *   `fileRowCountsPerGroup(i)(j)` = number of rows in `filesPerGroup(i)(j)`.
    *   Required by the packed reader to compute valid `(start, end)` ranges
    *   per file (it rejects negative end indices, so we cannot pass a
    *   "whole-file" sentinel).
    * @return
    *   raw pointer to the LoonColumnGroups C struct. Pass to
    *   `MilvusStorageReader.create` and eventually call [[destroy]] to free.
    */
  def createFromGroups(
      columnsPerGroup: Array[Array[String]],
      filesPerGroup: Array[Array[String]],
      fileRowCountsPerGroup: Array[Array[Long]]
  ): Long = {
    require(
      columnsPerGroup.length == filesPerGroup.length &&
        columnsPerGroup.length == fileRowCountsPerGroup.length,
      s"per-group array lengths must match: cols=${columnsPerGroup.length}, " +
        s"files=${filesPerGroup.length}, rowCounts=${fileRowCountsPerGroup.length}"
    )
    native.createFromGroups(
      columnsPerGroup,
      filesPerGroup,
      fileRowCountsPerGroup
    )
  }

  def count(columnGroupsPtr: Long): Int = native.count(columnGroupsPtr)
  def columns(columnGroupsPtr: Long, groupIndex: Int): Array[String] = native.columns(columnGroupsPtr, groupIndex)
  def files(columnGroupsPtr: Long, groupIndex: Int): Array[String] = native.files(columnGroupsPtr, groupIndex)
  def fileRowCounts(columnGroupsPtr: Long, groupIndex: Int): Array[Long] = native.fileRowCounts(columnGroupsPtr, groupIndex)
  def format(columnGroupsPtr: Long, groupIndex: Int): String = native.format(columnGroupsPtr, groupIndex)

  /** Release owned column groups returned by a writer or [[createFromGroups]].
    * Never pass borrowed manifest column groups here. Safe on a zero pointer.
    */
  def destroy(columnGroupsPtr: Long): Unit = {
    if (columnGroupsPtr != 0L) native.destroy(columnGroupsPtr)
  }
}
