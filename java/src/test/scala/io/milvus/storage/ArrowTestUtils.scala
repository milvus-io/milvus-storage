package io.milvus.storage

import org.apache.arrow.c.{ArrowArray, ArrowSchema, Data}
import org.apache.arrow.vector._
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import scala.collection.JavaConverters._

/**
 * Arrow test utilities for creating test schemas and arrays
 * These are used in integration tests
 */
object ArrowTestUtils {

  // Field names matching the C test implementation
  private val TABLE_NAME = "test_table"
  private val FIELD_INT64_NAME = "int64_field"
  private val FIELD_INT32_NAME = "int32_field"
  private val FIELD_STRING_NAME = "string_field"

  private val allocator = ArrowUtils.getAllocator

  /**
   * Create test struct schema matching the C FFI test
   * Creates a schema with 3 fields: int64_field, int32_field, string_field
   *
   * @return Address of ArrowSchema (to be freed by caller)
   */
  def createTestStructSchema(): Long = {
    val metadata0 = Map("PARQUET:field_id" -> "1").asJava
    val metadata1 = Map("PARQUET:field_id" -> "2").asJava
    val metadata2 = Map("PARQUET:field_id" -> "3").asJava

    val fields = List(
      new Field(FIELD_INT64_NAME, new FieldType(false, new ArrowType.Int(64, true), null, metadata0), null),
      new Field(FIELD_INT32_NAME, new FieldType(true, new ArrowType.Int(32, true), null, metadata1), null),
      new Field(FIELD_STRING_NAME, new FieldType(true, new ArrowType.Utf8(), null, metadata2), null)
    ).asJava

    val schema = new Schema(fields)
    val arrowSchema = ArrowSchema.allocateNew(allocator)
    Data.exportSchema(allocator, schema, null, arrowSchema)

    arrowSchema.memoryAddress()
  }

  /**
   * Create test struct array matching the C FFI test
   *
   * @param int64Data Array of int64 values
   * @param int32Data Array of int32 values
   * @param stringData Array of string values
   * @return Address of ArrowArray (to be freed by caller)
   */
  def createTestStructArray(int64Data: Array[Long],
                           int32Data: Array[Int],
                           stringData: Array[String]): Long = {

    val length = int64Data.length
    require(int32Data.length == length && stringData.length == length,
            "All arrays must have the same length")

    val metadata0 = Map("PARQUET:field_id" -> "1").asJava
    val metadata1 = Map("PARQUET:field_id" -> "2").asJava
    val metadata2 = Map("PARQUET:field_id" -> "3").asJava

    val fields = List(
      new Field(FIELD_INT64_NAME, new FieldType(false, new ArrowType.Int(64, true), null, metadata0), null),
      new Field(FIELD_INT32_NAME, new FieldType(true, new ArrowType.Int(32, true), null, metadata1), null),
      new Field(FIELD_STRING_NAME, new FieldType(true, new ArrowType.Utf8(), null, metadata2), null)
    ).asJava

    val schema = new Schema(fields)
    val root = VectorSchemaRoot.create(schema, allocator)
    root.allocateNew()

    // Populate int64 field
    val int64Vector = root.getVector(FIELD_INT64_NAME).asInstanceOf[BigIntVector]
    int64Vector.setValueCount(length)
    for (i <- 0 until length) {
      int64Vector.set(i, int64Data(i))
    }

    // Populate int32 field
    val int32Vector = root.getVector(FIELD_INT32_NAME).asInstanceOf[IntVector]
    int32Vector.setValueCount(length)
    for (i <- 0 until length) {
      int32Vector.set(i, int32Data(i))
    }

    // Populate string field
    val stringVector = root.getVector(FIELD_STRING_NAME).asInstanceOf[VarCharVector]
    stringVector.setValueCount(length)
    for (i <- 0 until length) {
      stringVector.setSafe(i, stringData(i).getBytes("UTF-8"))
    }

    root.setRowCount(length)

    val arrowArray = ArrowArray.allocateNew(allocator)
    val arrowSchema = ArrowSchema.allocateNew(allocator)

    Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchema)
    arrowSchema.close()

    // Keep root alive - it will be released when arrowArray is released
    keepAlive(arrowArray.memoryAddress(), root)

    arrowArray.memoryAddress()
  }

  /**
   * Import ArrowArray and extract all column data
   *
   * @param arrayPtr Pointer to ArrowArray (from readNextBatch)
   * @return Tuple of (length, int64_data, int32_data, string_data)
   */
  def importAndExtractData(arrayPtr: Long): (Int, Array[Long], Array[Int], Array[String]) = {
    if (arrayPtr == 0) {
      return (0, Array.empty[Long], Array.empty[Int], Array.empty[String])
    }

    val arrowArray = ArrowArray.wrap(arrayPtr)
    val arrowSchema = ArrowSchema.allocateNew(allocator)

    try {
      val metadata0 = Map("PARQUET:field_id" -> "1").asJava
      val metadata1 = Map("PARQUET:field_id" -> "2").asJava
      val metadata2 = Map("PARQUET:field_id" -> "3").asJava

      val fields = List(
        new Field(FIELD_INT64_NAME, new FieldType(false, new ArrowType.Int(64, true), null, metadata0), null),
        new Field(FIELD_INT32_NAME, new FieldType(true, new ArrowType.Int(32, true), null, metadata1), null),
        new Field(FIELD_STRING_NAME, new FieldType(true, new ArrowType.Utf8(), null, metadata2), null)
      ).asJava
      val schema = new Schema(fields)
      Data.exportSchema(allocator, schema, null, arrowSchema)

      val root = Data.importVectorSchemaRoot(allocator, arrowArray, arrowSchema, null)

      try {
        val rowCount = root.getRowCount

        val int64Vector = root.getVector(0).asInstanceOf[BigIntVector]
        val int64Data = new Array[Long](rowCount)
        for (i <- 0 until rowCount) {
          int64Data(i) = int64Vector.get(i)
        }

        val int32Vector = root.getVector(1).asInstanceOf[IntVector]
        val int32Data = new Array[Int](rowCount)
        for (i <- 0 until rowCount) {
          int32Data(i) = int32Vector.get(i)
        }

        val stringVector = root.getVector(2).asInstanceOf[VarCharVector]
        val stringData = new Array[String](rowCount)
        for (i <- 0 until rowCount) {
          stringData(i) = new String(stringVector.get(i), "UTF-8")
        }

        (rowCount, int64Data, int32Data, stringData)
      } finally {
        root.close()
      }
    } finally {
      arrowSchema.close()
    }
  }

  // Map to keep VectorSchemaRoot alive until ArrowArray is released
  private val keepAliveMap = new java.util.concurrent.ConcurrentHashMap[Long, VectorSchemaRoot]()

  private def keepAlive(arrayPtr: Long, root: VectorSchemaRoot): Unit = {
    keepAliveMap.put(arrayPtr, root)
  }

  /**
   * Release kept alive VectorSchemaRoot
   */
  def releaseKeptAlive(arrayPtr: Long): Unit = {
    val root = keepAliveMap.remove(arrayPtr)
    if (root != null) {
      root.close()
    }
  }
}
