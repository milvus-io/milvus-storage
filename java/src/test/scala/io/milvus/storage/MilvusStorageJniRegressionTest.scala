package io.milvus.storage

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.util.{AbstractMap, AbstractSet, Collections, Iterator => JavaIterator, Map => JavaMap, Set => JavaSet}
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._

import org.apache.arrow.c.{ArrowArray, ArrowSchema, Data}
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.{BigIntVector, VarCharVector, VectorSchemaRoot}
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** JNI ownership and exception regressions. Run with -Xcheck:jni and scan its
 * diagnostics as well as the test result; a warning is a failed native check.
 */
class MilvusStorageJniRegressionTest extends AnyFunSuite with Matchers {
  private def removeTree(path: Path): Unit = {
    val paths = Files.walk(path)
    try paths.iterator().asScala.toVector.reverse.foreach(Files.delete)
    finally paths.close()
  }

  Seq("entrySet", "toArray", "getKey", "getValue").foreach { failurePoint =>
    test(s"properties preserve the exception from $failurePoint and stop calling Java") {
      val expected = new IllegalStateException(s"failure in $failurePoint")
      val calls = ArrayBuffer.empty[String]
      def called(name: String): Unit = {
        calls += name
        if (name == failurePoint) throw expected
      }
      val entry = new AbstractMap.SimpleEntry[String, String]("fs.storage_type", "local") {
        override def getKey: String = { called("getKey"); super.getKey }
        override def getValue: String = { called("getValue"); super.getValue }
      }
      val entries = new AbstractSet[JavaMap.Entry[String, String]] {
        override def size(): Int = 1
        override def iterator(): JavaIterator[JavaMap.Entry[String, String]] =
          Collections.singleton[JavaMap.Entry[String, String]](entry).iterator()
        override def toArray(): Array[AnyRef] = {
          called("toArray")
          Array(entry)
        }
      }
      val map = new AbstractMap[String, String] {
        override def entrySet(): JavaSet[JavaMap.Entry[String, String]] = {
          called("entrySet")
          entries
        }
      }
      val properties = new MilvusStorageProperties()
      try {
        val actual = intercept[IllegalStateException](properties.create(map))
        actual should be theSameInstanceAs expected
        calls.toSeq shouldBe (Seq("entrySet", "toArray", "getKey", "getValue").takeWhile(_ != failurePoint) :+ failurePoint)
        properties.create(Map("fs.storage_type" -> "local"))
      } finally properties.free()
    }
  }

  test("properties reject null strings and remain usable after failure") {
    val properties = new MilvusStorageProperties()
    try {
      intercept[IllegalArgumentException](properties.create(Map("fs.storage_type" -> null)))
      intercept[IllegalArgumentException](properties.create(Map(null.asInstanceOf[String] -> "local")))
      properties.create(Map("fs.storage_type" -> "local"))
      properties.create(Map("fs.storage_type" -> "local"))
    } finally properties.free()
    properties.free()
    properties.isValid shouldBe false
  }

  test("FFI failures retain their native error code") {
    val reader = new MilvusStorageReader()
    val properties = new MilvusStorageProperties()
    try {
      properties.create(Map("fs.storage_type" -> "local"))
      val error = intercept[MilvusStorageException](reader.create(0L, 0L, null, properties))
      error.errorCode() shouldBe 1 // LOON_INVALID_ARGS
      error.getMessage should include("Invalid arguments")
    } finally {
      reader.destroy()
      properties.free()
    }
  }

  private val schema = new Schema(Seq(
    new Field("id", new FieldType(false, new ArrowType.Int(64, true), null), Collections.emptyList[Field]()),
    new Field("name", new FieldType(false, new ArrowType.Utf8(), null), Collections.emptyList[Field]())
  ).asJava)

  test("writer rejects a null path and remains usable after argument failures") {
    val work = Files.createTempDirectory("storage-jni-writer-arguments")
    val allocator = new RootAllocator(Long.MaxValue)
    val properties = new MilvusStorageProperties()
    val writer = new MilvusStorageWriter()
    val writerSchema = ArrowSchema.allocateNew(allocator)
    writerSchema.save(new ArrowSchema.Snapshot())
    try {
      properties.create(Map(
        "fs.storage_type" -> "local",
        "fs.root_path" -> work.toAbsolutePath.toString,
        "writer.policy" -> "single"
      ))
      Data.exportSchema(allocator, schema, null, writerSchema)
      intercept[IllegalArgumentException] {
        writer.create(null, writerSchema.memoryAddress(), properties)
      }
      writer.isValid shouldBe false
      val error = intercept[MilvusStorageException](writer.create("segment", 0L, properties))
      error.errorCode() shouldBe 1 // LOON_INVALID_ARGS
      writer.isValid shouldBe false
      writer.create("segment", writerSchema.memoryAddress(), properties)
      writer.isValid shouldBe true
      intercept[MilvusStorageException](writer.write(0L)).errorCode() shouldBe 1
    } finally {
      writer.destroy()
      try {
        if (writerSchema.snapshot().release != 0L) writerSchema.release()
      } finally writerSchema.close()
      properties.free()
      allocator.close()
      removeTree(work)
    }
  }

  test("packed writer rejects null paths and invalid column indices without retaining a handle") {
    val allocator = new RootAllocator(Long.MaxValue)
    val properties = new MilvusStorageProperties()
    val writer = new MilvusPackedWriter()
    val writerSchema = ArrowSchema.allocateNew(allocator)
    writerSchema.save(new ArrowSchema.Snapshot())
    try {
      properties.create(Map("fs.storage_type" -> "local"))
      Data.exportSchema(allocator, schema, null, writerSchema)
      intercept[IllegalArgumentException] {
        writer.create(Array(null.asInstanceOf[String]), Array(Array(0)), writerSchema.memoryAddress(), properties)
      }
      writer.isValid shouldBe false
      val error = intercept[MilvusStorageException] {
        writer.create(Array("invalid.parquet"), Array(Array(-1)), writerSchema.memoryAddress(), properties)
      }
      error.errorCode() shouldBe 1 // LOON_INVALID_ARGS
      error.getMessage should include("column index out of range")
      writer.isValid shouldBe false
    } finally {
      writer.destroy()
      try {
        if (writerSchema.snapshot().release != 0L) writerSchema.release()
      } finally writerSchema.close()
      properties.free()
      allocator.close()
    }
  }

  private def withReader(body: (MilvusStorageReader, RootAllocator) => Unit): Unit = {
    val work = Files.createTempDirectory("storage-jni-regression")
    val allocator = new RootAllocator(Long.MaxValue)
    val properties = new MilvusStorageProperties()
    val writer = new MilvusStorageWriter()
    val reader = new MilvusStorageReader()
    var groups = 0L
    try {
      properties.create(Map(
        "fs.storage_type" -> "local",
        "fs.root_path" -> work.toAbsolutePath.toString,
        "writer.policy" -> "single",
        "reader.record_batch_max_rows" -> "8"
      ))
      val writerSchema = ArrowSchema.allocateNew(allocator)
      try {
        Data.exportSchema(allocator, schema, null, writerSchema)
        writer.create("segment", writerSchema.memoryAddress(), properties)
      } finally writerSchema.close()
      val root = VectorSchemaRoot.create(schema, allocator)
      try {
        root.allocateNew()
        val ids = root.getVector("id").asInstanceOf[BigIntVector]
        val names = root.getVector("name").asInstanceOf[VarCharVector]
        (0 until 40).foreach { row =>
          ids.setSafe(row, row.toLong)
          names.setSafe(row, s"row-$row".getBytes(UTF_8))
        }
        root.setRowCount(40)
        val array = ArrowArray.allocateNew(allocator)
        try {
          Data.exportVectorSchemaRoot(allocator, root, null, array)
          writer.write(array.memoryAddress())
        } finally array.close()
      } finally root.close()
      groups = writer.close()
      writer.destroy()
      val readerSchema = ArrowSchema.allocateNew(allocator)
      try {
        Data.exportSchema(allocator, schema, null, readerSchema)
        reader.create(groups, readerSchema.memoryAddress(), null, properties)
      } finally readerSchema.close()
      body(reader, allocator)
    } finally {
      reader.destroy()
      writer.destroy()
      MilvusStorageColumnGroups.destroy(groups)
      properties.free()
      allocator.close()
      removeTree(work)
    }
  }

  private def readIds(reader: MilvusStorageReader, handle: Long, allocator: RootAllocator): Seq[Long] = {
    val values = ArrayBuffer.empty[Long]
    var more = true
    while (more) {
      val array = ArrowArray.allocateNew(allocator)
      val batchSchema = ArrowSchema.allocateNew(allocator)
      try {
        more = reader.readNextBatchScala(handle, array.memoryAddress(), batchSchema.memoryAddress())
        if (more) {
          val batch = Data.importVectorSchemaRoot(allocator, array, batchSchema, null)
          try {
            val ids = batch.getVector("id").asInstanceOf[BigIntVector]
            values ++= (0 until batch.getRowCount).map(ids.get)
          } finally batch.close()
        }
      } finally {
        array.close()
        batchSchema.close()
      }
    }
    values.toVector
  }

  test("batch reads preserve sliced rows and report their actual copies") {
    withReader { (reader, allocator) =>
      val handle = reader.openRecordBatchReaderScala()
      try {
        readIds(reader, handle, allocator) shouldBe (0L until 40L)
        val stats = reader.recordBatchReaderStatsScala(handle)
        stats(0) should be > 1L
        stats(1) should be > 0L
        stats(2) should be > 0L
      } finally reader.destroyRecordBatchReaderScala(handle)
    }
  }

  test("owned take survives source reader destruction and supports early close") {
    withReader { (reader, allocator) =>
      val early = reader.takeRecordBatchReaderScala(Array(0L, 3L, 39L), 1L, Array("id"))
      reader.destroyRecordBatchReaderScala(early)
      val handle = reader.takeRecordBatchReaderScala(Array(1L, 9L, 32L), 1L, Array("id"))
      reader.destroy()
      try {
        readIds(reader, handle, allocator) shouldBe Seq(1L, 9L, 32L)
        reader.recordBatchReaderStatsScala(handle)(0) should be > 0L
      } finally reader.destroyRecordBatchReaderScala(handle)
    }
  }

  test("invalid take inputs leave the reader usable and legacy take can be released as a whole") {
    withReader { (reader, allocator) =>
      Seq(Array.empty[Long], Array(-1L), Array(2L, 1L), Array(2L, 2L)).foreach { rows =>
        intercept[IllegalArgumentException](reader.takeRecordBatchReaderScala(rows, 1L))
      }
      intercept[IllegalArgumentException](reader.takeRecordBatchReaderScala(Array(1L), 0L))
      intercept[IllegalArgumentException](reader.takeRecordBatchReaderScala(Array(1L), 1L, Array(null)))
      val legacy = reader.takeRows(Array(1L, 5L), 1L, Array("id"))
      legacy should not be empty
      reader.freeTakeRowsScala(legacy)
      val handle = reader.takeRecordBatchReaderScala(Array(3L), 1L)
      try readIds(reader, handle, allocator) shouldBe Seq(3L)
      finally reader.destroyRecordBatchReaderScala(handle)
    }
  }
}
