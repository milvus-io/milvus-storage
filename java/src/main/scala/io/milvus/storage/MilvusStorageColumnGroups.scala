package io.milvus.storage

/** Native bridge — do not call directly. */
class MilvusStorageColumnGroupsNative {
  @native def createFromGroups(
      columnsPerGroup: Array[Array[String]],
      filesPerGroup: Array[Array[String]],
      fileRowCountsPerGroup: Array[Array[Long]]
  ): Long
  @native def destroy(columnGroupsPtr: Long): Unit
}

/** Build a `LoonColumnGroups*` directly from a caller-provided layout, without
  * resolving a milvus-storage `.milvus_manifest`.
  *
  * Used by the spark-connector's StorageV2 read path: the column-group layout
  * is recovered from the snapshot AVRO + parquet footer kv-metadata
  * (`group_field_id_list`), then fed here so the packed reader can open the
  * segment files directly.
  *
  * Caller is responsible for calling [[destroy]] once the reader no longer
  * needs the column groups — mirrors the contract of
  * `MilvusStorageManifest.getLatestColumnGroupsScala`.
  */
object MilvusStorageColumnGroups {
  NativeLibraryLoader.loadLibrary()
  private val native = new MilvusStorageColumnGroupsNative()

  /** Construct a LoonColumnGroups from per-group column names, file paths and
    * per-file row counts.
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

  /** Release the LoonColumnGroups allocated by [[createFromGroups]]. Safe on a
    * zero pointer.
    */
  def destroy(columnGroupsPtr: Long): Unit = {
    if (columnGroupsPtr != 0L) native.destroy(columnGroupsPtr)
  }
}
