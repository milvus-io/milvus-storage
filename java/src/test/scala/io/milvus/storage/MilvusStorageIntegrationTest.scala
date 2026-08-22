package io.milvus.storage

import org.apache.arrow.c.{ArrowArray, ArrowSchema}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll
import java.util.{HashMap => JHashMap}
import java.io.File

/**
 * Milvus Storage Scala Integration Test
 * Write and read test using JNI FFI interface
 */
class MilvusStorageIntegrationTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  private val TEST_BASE_PATH = "/tmp/milvus_storage_scala_test"

  override def beforeAll(): Unit = {
    // A missing or unloadable JNI library is a failed integration build, not a
    // reason to cancel the only suite that exercises it.
    NativeLibraryLoader.loadLibrary()

    val testDir = new File(TEST_BASE_PATH)
    if (testDir.exists()) {
      deleteRecursively(testDir)
    }
  }

  override def afterAll(): Unit = {
    // Clean up test directory
    val testDir = new File(TEST_BASE_PATH)
    if (testDir.exists()) {
      deleteRecursively(testDir)
    }
  }

  private def deleteRecursively(file: File): Unit = {
    if (file.isDirectory) {
      file.listFiles().foreach(deleteRecursively)
    }
    file.delete()
  }

  "Milvus Storage" should "perform basic write and read operations" in {
    println("\n=== Milvus Storage Write and Read Test ===\n")

    // Test data
    val int64Data = Array(1L, 2L, 3L, 4L, 5L)
    val int32Data = Array(25, 30, 35, 40, 45)
    val stringData = Array("ABC", "BCD", "DDDD", "EEEEEa", "CCCC23123")
    val schema = ArrowTestUtils.createTestStructSchema()

    val structArray = ArrowTestUtils.createTestStructArray(int64Data, int32Data, stringData)

    // Create writer properties
    val writerProperties = new MilvusStorageProperties()
    val writerProps = new JHashMap[String, String]()
    writerProps.put("writer.policy", "single")
    writerProps.put("fs.storage_type", "local")
    writerProps.put("fs.root_path", "/tmp/")
    writerProperties.create(writerProps)
    writerProperties.isValid should be(true)

    // Write data
    val writer = new MilvusStorageWriter()
    writer.create(TEST_BASE_PATH, schema, writerProperties)
    writer.write(structArray)
    writer.flush()
    val columnGroups = writer.close()
    columnGroups should not be 0
    writer.destroy()

    // Create reader properties
    val readerProperties = new MilvusStorageProperties()
    val readerProps = new JHashMap[String, String]()
    readerProps.put("fs.storage_type", "local")
    readerProps.put("fs.root_path", "/tmp/")
    readerProperties.create(readerProps)
    readerProperties.isValid should be(true)

    // read data via per-batch RecordBatchReader (the only Java-safe path — the
    // ArrowArrayStream-based API duplicates data when Arrow Java imports a
    // batch whose ArrowArray carries a non-zero offset).
    val reader = new MilvusStorageReader()
    val neededColumns = Array("int64_field", "int32_field", "string_field")
    val readerSchema = ArrowTestUtils.createTestStructSchema()
    reader.create(columnGroups, readerSchema, neededColumns, readerProperties)

    val zeroParallelism = intercept[IllegalArgumentException] {
      reader.takeRows(Array(0L), 0L)
    }
    zeroParallelism.getMessage should include("parallelism must be > 0")
    val negativeParallelism = intercept[IllegalArgumentException] {
      reader.takeRows(Array(0L), -1L)
    }
    negativeParallelism.getMessage should include("parallelism must be > 0")
    intercept[IllegalArgumentException] {
      reader.takeRows(null, 1L)
    }
    intercept[IllegalArgumentException] {
      reader.takeRows(Array.empty[Long], 1L)
    }
    intercept[IllegalArgumentException] {
      reader.takeRows(Array(0L), 1L, Array("int64_field", null))
    }
    intercept[IllegalArgumentException] {
      reader.takeRows(Array(0L), 1L, Array(""))
    }

    val segmentReader = new MilvusSegmentReader()
    intercept[IllegalArgumentException] {
      segmentReader.open(null, -1, 1L, null, Seq.empty, 1L)
    }
    intercept[IllegalArgumentException] {
      segmentReader.open("", -1, 1L, null, Seq.empty, 1L)
    }
    intercept[IllegalArgumentException] {
      segmentReader.open("segment", -1, 1L, Array("field", null), Seq.empty, 1L)
    }
    intercept[IllegalArgumentException] {
      segmentReader.open("segment", -1, 1L, null, null, 1L)
    }
    intercept[IllegalArgumentException] {
      segmentReader.open("segment", -1, 1L, null, Seq(LobColumnConfig(1L, null)), 1L)
    }
    intercept[IllegalArgumentException] {
      segmentReader.open("segment", -1, 1L, null, Seq(LobColumnConfig(1L, "lob", flushThresholdBytes = 1024,
        maxLobFileBytes = 512)), 1L)
    }

    val chunkReader = new MilvusStorageChunkReader()
    // A non-zero sentinel is sufficient: validation must run before JNI and
    // therefore must never dereference this handle.
    chunkReader.setHandle(1L)
    intercept[IllegalArgumentException] {
      chunkReader.getChunksScala(Array(0L), 0L)
    }
    intercept[IllegalArgumentException] {
      chunkReader.getChunksScala(Array(0L), -1L)
    }
    intercept[IllegalArgumentException] {
      chunkReader.getChunksScala(null, 1L)
    }
    intercept[IllegalArgumentException] {
      chunkReader.getChunksScala(Array.empty[Long], 1L)
    }
    intercept[IllegalArgumentException] {
      chunkReader.getChunkIndicesScala(null)
    }
    intercept[IllegalArgumentException] {
      chunkReader.getChunkIndicesScala(Array.empty[Long])
    }

    val rbrHandle = reader.openRecordBatchReaderScala(null)
    val batchArray = ArrowArray.allocateNew(ArrowUtils.getAllocator)
    val batchSchema = ArrowSchema.allocateNew(ArrowUtils.getAllocator)
    try {
      val hasBatch = reader.readNextBatchScala(rbrHandle, batchArray.memoryAddress(), batchSchema.memoryAddress())
      hasBatch should be(true)

      // validate data
      val (length, int64Col, int32Col, stringCol) = ArrowTestUtils.importAndExtractData(batchArray.memoryAddress())
      length should be(5)
      int64Col should equal(int64Data)
      int32Col should equal(int32Data)
      stringCol should equal(stringData)
    } finally {
      batchArray.close()
      batchSchema.close()
      reader.destroyRecordBatchReaderScala(rbrHandle)
    }

    // Cleanup, ArrowSchema is created in java, so we need to release it
    ArrowUtils.releaseArrowSchema(readerSchema, false)
    val readerSchemaWrapper = ArrowSchema.wrap(readerSchema)
    readerSchemaWrapper.close()

    reader.destroy()
    readerProperties.free()

    writerProperties.free()
    ArrowUtils.releaseArrowArray(structArray, false)
    val structArrayWrapper = ArrowArray.wrap(structArray)
    structArrayWrapper.close()

    ArrowUtils.releaseArrowSchema(schema, false)
    val schemaWrapper = ArrowSchema.wrap(schema)
    schemaWrapper.close()

    succeed
  }
}
