package io.milvus.storage

/** Native bridge — do not call directly. */
class MilvusStorageRuntimeNative {
  @native def ioThreadPoolCapacity(): Int
  @native def setArrowIoThreadPoolCapacity(threads: Int): Unit
}

/** Process-wide settings of the native storage runtime.
  *
  * Arrow's IO thread pool is shared by every reader in the process. Each
  * coalesced range that a Parquet read requests at once runs a blocking object
  * storage request on one of its threads, so the pool size bounds how many
  * ranges all concurrent readers of the process fetch together. Arrow creates
  * the pool with 8 threads unless `ARROW_IO_THREADS` is set, and grows or
  * shrinks it safely afterwards.
  */
object MilvusStorageRuntime {
  NativeLibraryLoader.loadLibrary()
  private val native = new MilvusStorageRuntimeNative()

  /** Threads in Arrow's IO pool now. */
  def ioThreadPoolCapacity: Int = native.ioThreadPoolCapacity()

  /** Set Arrow's IO pool to `threads`, which may be larger or smaller than the
    * current capacity. The caller owns the policy; a host with several
    * components sharing the process decides whether a later call may lower it.
    * Safe to call at any time and from any thread.
    */
  def setArrowIoThreadPoolCapacity(threads: Int): Unit = {
    require(threads > 0, s"IO thread pool capacity must be positive, got $threads")
    native.setArrowIoThreadPoolCapacity(threads)
  }
}
