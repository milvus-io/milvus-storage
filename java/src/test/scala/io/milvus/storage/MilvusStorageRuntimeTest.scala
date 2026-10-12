package io.milvus.storage

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class MilvusStorageRuntimeTest extends AnyFunSuite with Matchers {
  test("the IO thread pool takes a larger and then a smaller capacity") {
    val before = MilvusStorageRuntime.ioThreadPoolCapacity
    before should be > 0
    try {
      MilvusStorageRuntime.setArrowIoThreadPoolCapacity(before + 3)
      MilvusStorageRuntime.ioThreadPoolCapacity shouldBe before + 3
      MilvusStorageRuntime.setArrowIoThreadPoolCapacity(before)
      MilvusStorageRuntime.ioThreadPoolCapacity shouldBe before
    } finally MilvusStorageRuntime.setArrowIoThreadPoolCapacity(before)
  }

  test("a capacity that is not positive is rejected") {
    an[IllegalArgumentException] should be thrownBy MilvusStorageRuntime.setArrowIoThreadPoolCapacity(0)
  }
}
