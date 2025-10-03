package io.milvus.storage

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
  private var libraryLoaded = false

  override def beforeAll(): Unit = {
    try {
      NativeLibraryLoader.loadLibrary()
      libraryLoaded = true

      // Clean up any existing test directory
      val testDir = new File(TEST_BASE_PATH)
      if (testDir.exists()) {
        deleteRecursively(testDir)
      }
    } catch {
      case e: Exception =>
        libraryLoaded = false
        println(s"Failed to load milvus_storage_jni library: ${e.getMessage}")
        e.printStackTrace()
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
    assume(libraryLoaded, "Native library must be loaded")

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
    val manifest = writer.close()
    manifest should not be null
    manifest should not be empty
    println(s"Writer closed. Manifest size: ${manifest.length} bytes")
    println(s"Manifest preview: ${manifest.take(200)}...")
    writer.destroy()

    // Create reader properties
    val readerProperties = new MilvusStorageProperties()
    val readerProps = new JHashMap[String, String]()
    readerProps.put("fs.storage_type", "local")
    readerProps.put("fs.root_path", "/tmp/")
    readerProperties.create(readerProps)
    readerProperties.isValid should be(true)

    // read data
    val reader = new MilvusStorageReader()
    val neededColumns = Array("int64_field", "int32_field", "string_field")
    val readerSchema = ArrowTestUtils.createTestStructSchema()
    reader.create(manifest, readerSchema, neededColumns, readerProperties)
    val recordBatchReader = reader.getRecordBatchReaderScala(null, 1024, 8 * 1024 * 1024)
    val arrowArray = ArrowUtils.readNextBatch(recordBatchReader)

    // validate data
    val (length, int64Col, int32Col, stringCol) = ArrowTestUtils.importAndExtractData(arrowArray)
    length should be(5)
    int64Col should equal(int64Data)
    int32Col should equal(int32Data)
    stringCol should equal(stringData)

    ArrowUtils.releaseArrowStream(recordBatchReader)

    // Cleanup
    ArrowUtils.releaseArrowSchema(readerSchema)
    reader.destroy()
    readerProperties.free()

    writerProperties.free()
    ArrowUtils.releaseArrowArray(structArray)
    ArrowUtils.releaseArrowSchema(schema)
    succeed
  }
}