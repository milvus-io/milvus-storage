package io.milvus.storage

/**
 * Scala wrapper for MilvusStorage Transaction
 * Provides transaction management functionality for manifest operations
 */
object MilvusStorageTransaction {

  /**
   * Transaction conflict resolution strategies.
   * Passed at begin() time — determines how commit() handles conflicts.
   */
  object ResolveStrategy extends Enumeration {
    type ResolveStrategy = Value
    val FAIL = Value(0)        // Fail on conflict
    val OVERWRITE = Value(2)   // Overwrite existing data on conflict
  }

  /** Read the latest committed version. */
  val LATEST_VERSION: Long = -1L
  /** Default retry limit on commit conflicts. */
  val DEFAULT_RETRY_LIMIT: Int = 1
}

/**
 * Transaction instance for manifest operations
 */
class MilvusStorageTransaction extends AutoCloseable {
  // Ensure native library is loaded
  NativeLibraryLoader.loadLibrary()

  import MilvusStorageTransaction._

  private var transactionHandle: Long = 0
  private var isDestroyed: Boolean = false
  private var ownedManifests = List.empty[Long]

  /**
   * Begin a transaction.
   *
   * @param basePath        Base path for storage
   * @param properties      MilvusStorage properties
   * @param readVersion     Version to read (-1 / LATEST_VERSION for latest)
   * @param resolveStrategy Conflict resolution strategy (default FAIL)
   * @param retryLimit      Maximum retries on commit conflicts (default 1)
   */
  def begin(basePath: String,
            properties: MilvusStorageProperties,
            readVersion: Long = LATEST_VERSION,
            resolveStrategy: ResolveStrategy.Value = ResolveStrategy.FAIL,
            retryLimit: Int = DEFAULT_RETRY_LIMIT): Unit = {
    begin(basePath, properties.getPtr, readVersion, resolveStrategy.id, retryLimit)
  }

  /**
   * Begin a transaction with raw properties pointer and integer resolve id.
   *
   * @param basePath      Base path for storage
   * @param propertiesPtr Pointer to properties
   * @param readVersion   Version to read (-1 for latest)
   * @param resolveId     Conflict resolution strategy id (0=FAIL, 2=OVERWRITE)
   * @param retryLimit    Maximum retries on commit conflicts
   */
  def begin(basePath: String,
            propertiesPtr: Long,
            readVersion: Long,
            resolveId: Int,
            retryLimit: Int): Unit = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle != 0L) throw new IllegalStateException("Transaction already initialized")
    transactionHandle = transactionBegin(basePath, propertiesPtr, readVersion, resolveId, retryLimit)
  }

  /**
   * Begin a transaction with default settings (read latest, fail on conflict, 1 retry).
   */
  def begin(basePath: String, propertiesPtr: Long): Unit = {
    begin(basePath, propertiesPtr, LATEST_VERSION, ResolveStrategy.FAIL.id, DEFAULT_RETRY_LIMIT)
  }

  /**
   * Append files to existing column groups.
   *
   * @param columnGroups Column groups as raw pointer
   */
  def appendFiles(columnGroups: Long): Unit = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle == 0) throw new IllegalStateException("Transaction not initialized")
    if (columnGroups == 0) throw new IllegalArgumentException("columnGroups must not be null")
    transactionAppendFiles(transactionHandle, columnGroups)
  }

  /**
   * Add column groups as new fields (schema evolution: add columns).
   *
   * Each column group in the provided pointer is added as a new field column group.
   *
   * @param columnGroups Column groups as raw pointer
   */
  def addColumnGroups(columnGroups: Long): Unit = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle == 0) throw new IllegalStateException("Transaction not initialized")
    if (columnGroups == 0) throw new IllegalArgumentException("columnGroups must not be null")
    transactionAddColumnGroups(transactionHandle, columnGroups)
  }

  /**
   * Drop a single column from the manifest.
   *
   * Removes the column from its column group. If the column group becomes
   * empty after removal, the entire column group is removed.
   * Automatically drops all indexes on this column.
   * Noop if the column doesn't exist.
   *
   * @param columnName Name of the column to drop
   */
  def dropColumn(columnName: String): Unit = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle == 0) throw new IllegalStateException("Transaction not initialized")
    if (columnName == null || columnName.isEmpty) throw new IllegalArgumentException("columnName must not be null or empty")
    transactionDropColumn(transactionHandle, columnName)
  }

  /**
   * Get column groups from current transaction
   * @return Borrowed column groups, valid until this transaction is destroyed
   */
  def getColumnGroups(): Long = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle == 0) throw new IllegalStateException("Transaction not initialized")
    val manifestPtr = transactionGetColumnGroups(transactionHandle)
    try {
      val groups = MilvusStorageManifest.borrowedColumnGroups(manifestPtr)
      ownedManifests = manifestPtr :: ownedManifests
      groups
    } catch {
      case error: Throwable =>
        MilvusStorageManifest.destroyManifest(manifestPtr)
        throw error
    }
  }

  /**
   * Commit the transaction.
   *
   * Updates (appendFiles / addColumnGroups / dropColumn) must be staged via their
   * dedicated methods before calling commit. Multiple stagings can be combined
   * in a single transaction (e.g. dropColumn then addColumnGroups for column replacement).
   *
   * @return Committed manifest version (>= 0 on success, -1 on failure)
   */
  def commit(): Long = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle == 0) throw new IllegalStateException("Transaction not initialized")
    transactionCommit(transactionHandle)
  }

  /**
   * Abort the transaction
   */
  def abort(): Unit = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle == 0) throw new IllegalStateException("Transaction not initialized")
    try transactionAbort(transactionHandle)
    finally {
      transactionHandle = 0L
      isDestroyed = true
      releaseManifests()
    }
  }

  /**
   * Get the native handle (for internal use)
   */
  def getHandle: Long = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    transactionHandle
  }

  /**
   * Destroy the transaction and free resources
   */
  def destroy(): Unit = synchronized {
    if (!isDestroyed) {
      val handle = transactionHandle
      transactionHandle = 0L
      isDestroyed = true
      try { if (handle != 0L) transactionDestroy(handle) }
      finally releaseManifests()
    }
  }

  override def close(): Unit = destroy()

  private def releaseManifests(): Unit = {
    val manifests = ownedManifests
    ownedManifests = Nil
    manifests.foreach(MilvusStorageManifest.destroyManifest)
  }

  /** Register stat files and their metadata in this transaction. */
  def updateStat(name: String, paths: Array[String], metadata: Map[String, String]): Unit = {
    requireActive()
    require(name != null && name.nonEmpty, "Stat name must not be empty")
    require(paths != null && metadata != null, "Stat paths and metadata must not be null")
    val entries = metadata.toArray
    transactionUpdateStat(transactionHandle, name, paths, entries.map(_._1), entries.map(_._2))
  }

  /** Register a primary-key delta log. */
  def addDeltaLog(path: String, entries: Long): Unit = {
    requireActive()
    require(path != null && path.nonEmpty, "Delta log path must not be empty")
    require(entries >= 0L, "Delta entry count must be nonnegative")
    transactionAddDeltaLog(transactionHandle, path, entries)
  }

  private def requireActive(): Unit = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle == 0L) throw new IllegalStateException("Transaction not initialized")
  }

  /**
   * Check if transaction is valid
   */
  def isValid: Boolean = !isDestroyed && transactionHandle != 0

  // Native method declarations
  @native private def transactionUpdateStat(handle: Long, name: String, paths: Array[String],
                                            metadataKeys: Array[String], metadataValues: Array[String]): Unit
  @native private def transactionAddDeltaLog(handle: Long, path: String, entries: Long): Unit
  @native private def transactionBegin(basePath: String, propertiesPtr: Long,
                                       readVersion: Long, resolveId: Int, retryLimit: Int): Long
  @native private def transactionGetColumnGroups(transactionHandle: Long): Long
  @native private def transactionAppendFiles(transactionHandle: Long, columnGroups: Long): Unit
  @native private def transactionAddColumnGroups(transactionHandle: Long, columnGroups: Long): Unit
  @native private def transactionDropColumn(transactionHandle: Long, columnName: String): Unit
  @native private def transactionCommit(transactionHandle: Long): Long
  @native private def transactionAbort(transactionHandle: Long): Unit
  @native private def transactionDestroy(transactionHandle: Long): Unit
}
