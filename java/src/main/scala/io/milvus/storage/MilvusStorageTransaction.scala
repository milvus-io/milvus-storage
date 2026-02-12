package io.milvus.storage

/**
 * Scala wrapper for MilvusStorage Transaction
 * Provides transaction management functionality for manifest operations
 */
object MilvusStorageTransaction {

  /**
   * Transaction update types
   */
  object UpdateType extends Enumeration {
    type UpdateType = Value
    val ADDFILES = Value(0)  // Add new files to manifest
    val ADDFIELD = Value(1)  // Add new field to schema
  }

  /**
   * Transaction conflict resolution strategies
   */
  object ResolveStrategy extends Enumeration {
    type ResolveStrategy = Value
    val FAIL = Value(0)   // Fail on conflict
    val MERGE = Value(1)  // Merge changes on conflict
  }
}

/**
 * Transaction instance for manifest operations
 */
class MilvusStorageTransaction {
  // Ensure native library is loaded
  NativeLibraryLoader.loadLibrary()

  import MilvusStorageTransaction._

  private var transactionHandle: Long = 0
  private var isDestroyed: Boolean = false

  /**
   * Begin a transaction
   * @param basePath The base path for storage
   * @param properties MilvusStorage properties
   */
  def begin(basePath: String, properties: MilvusStorageProperties): Unit = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    transactionHandle = transactionBegin(basePath, properties.getPtr)
  }

  /**
   * Begin a transaction with properties pointer
   * @param basePath The base path for storage
   * @param propertiesPtr Pointer to properties
   */
  def begin(basePath: String, propertiesPtr: Long): Unit = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    transactionHandle = transactionBegin(basePath, propertiesPtr)
  }

  /**
   * Get column groups from current transaction
   * @return Column groups as raw pointer
   */
  def getColumnGroups(): Long = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle == 0) throw new IllegalStateException("Transaction not initialized")
    transactionGetColumnGroups(transactionHandle)
  }

  /**
   * Commit the transaction
   * @param updateType Update operation type
   * @param resolveStrategy Conflict resolution strategy
   * @param columnGroups New column groups as raw pointer
   * @return Committed manifest version (>= 0 on success, -1 on failure)
   */
  def commit(updateType: UpdateType.Value, resolveStrategy: ResolveStrategy.Value, columnGroups: Long): Long = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle == 0) throw new IllegalStateException("Transaction not initialized")
    if (columnGroups == 0) throw new IllegalArgumentException("columnGroups must not be null")
    transactionCommit(transactionHandle, updateType.id, resolveStrategy.id, columnGroups)
  }

  /**
   * Commit the transaction with raw integer IDs
   * @param updateId Update operation type ID (0=ADDFILES, 1=ADDFIELD)
   * @param resolveId Conflict resolution strategy ID (0=FAIL, 1=MERGE)
   * @param columnGroups New column groups as raw pointer
   * @return Committed manifest version (>= 0 on success, -1 on failure)
   */
  def commit(updateId: Int, resolveId: Int, columnGroups: Long): Long = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle == 0) throw new IllegalStateException("Transaction not initialized")
    if (columnGroups == 0) throw new IllegalArgumentException("columnGroups must not be null")
    transactionCommit(transactionHandle, updateId, resolveId, columnGroups)
  }

  /**
   * Abort the transaction
   */
  def abort(): Unit = {
    if (isDestroyed) throw new IllegalStateException("Transaction has been destroyed")
    if (transactionHandle == 0) throw new IllegalStateException("Transaction not initialized")
    transactionAbort(transactionHandle)
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
  def destroy(): Unit = {
    if (transactionHandle != 0 && !isDestroyed) {
      transactionDestroy(transactionHandle)
      transactionHandle = 0
      isDestroyed = true
    }
  }

  /**
   * Check if transaction is valid
   */
  def isValid: Boolean = !isDestroyed && transactionHandle != 0

  // Native method declarations
  @native private def transactionBegin(basePath: String, propertiesPtr: Long): Long
  @native private def transactionGetColumnGroups(transactionHandle: Long): Long
  @native private def transactionCommit(transactionHandle: Long, updateId: Int, resolveId: Int, columnGroups: Long): Long
  @native private def transactionAbort(transactionHandle: Long): Unit
  @native private def transactionDestroy(transactionHandle: Long): Unit
}
