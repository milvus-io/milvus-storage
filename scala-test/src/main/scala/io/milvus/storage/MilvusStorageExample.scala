package io.milvus.storage

import scala.util.{Try, Using}

/**
 * Example usage of MilvusStorage Scala API
 * Demonstrates how to use the Scala wrapper classes
 */
object MilvusStorageExample {

  /**
   * Example of writing data to MilvusStorage
   */
  def writeExample(): Unit = {
    // Create properties for local storage
    Using(new MilvusStorageProperties()) { properties =>
      properties.create(Map(
        "storage_type" -> "local",
        "uri" -> "/tmp/milvus_storage_test"
      ))

      // Create writer
      Using(new MilvusStorageWriter()) { writer =>
        // Assuming you have schema and data pointers from Arrow
        val schemaPtr: Long = 0 // Replace with actual Arrow schema pointer

        writer.create("/tmp/milvus_storage_test", schemaPtr, properties)

        // Write data (assuming you have Arrow arrays)
        val arrayPtr: Long = 0 // Replace with actual Arrow array pointer
        writer.write(arrayPtr)

        // Flush and close
        writer.flush()
        val manifest = writer.close()
        println(s"Data written successfully. Manifest: $manifest")
      }
    }
  }

  /**
   * Example of writing data to cloud storage (S3)
   */
  def writeToS3Example(): Unit = {
    Using(new MilvusStorageProperties()) { properties =>
      properties.createCloudStorage(
        storageType = "s3",
        endpoint = "s3.amazonaws.com",
        region = "us-west-2",
        accessKey = "your-access-key",
        secretKey = "your-secret-key",
        bucketName = "your-bucket"
      )

      Using(new MilvusStorageWriter()) { writer =>
        val schemaPtr: Long = 0 // Replace with actual schema
        writer.create("s3://your-bucket/path/", schemaPtr, properties)

        // Write your data...
        val manifest = writer.close()
        println(s"Data written to S3. Manifest: $manifest")
      }
    }
  }

  /**
   * Example of reading data from MilvusStorage
   */
  def readExample(): Unit = {
    val manifest = """{"version": "1.0", "files": [...]}""" // Replace with actual manifest

    Using(new MilvusStorageProperties()) { properties =>
      properties.create(Map(
        "storage_type" -> "local",
        "uri" -> "/tmp/milvus_storage_test"
      ))

      Using(new MilvusStorageReader()) { reader =>
        val schemaPtr: Long = 0 // Replace with actual schema
        val columns = Array("column1", "column2") // Replace with actual column names

        reader.create(manifest, schemaPtr, columns, properties)

        // Example 1: Get record batch reader
        val streamPtr = reader.getRecordBatchReader()
        println(s"Got record batch stream: $streamPtr")

        // Example 2: Take specific rows
        val rowIndices = Array(0L, 1L, 2L, 10L, 20L)
        val arrayPtr = reader.take(rowIndices)
        println(s"Got array for specific rows: $arrayPtr")

        // Example 3: Use chunk reader
        Using(reader.getChunkReader(0)) { chunkReader =>
          val chunkIndices = chunkReader.getChunkIndices(rowIndices)
          println(s"Chunk indices: ${chunkIndices.mkString(", ")}")

          val chunks = chunkReader.getChunks(chunkIndices)
          println(s"Got ${chunks.length} chunks")
        }
      }
    }
  }

  /**
   * Example with error handling
   */
  def robustExample(): Unit = {
    Try {
      Using.Manager { use =>
        val properties = use(new MilvusStorageProperties())
        properties.create(Map("storage_type" -> "local", "uri" -> "/tmp/test"))

        val writer = use(new MilvusStorageWriter())
        val reader = use(new MilvusStorageReader())

        // All resources will be automatically closed
        println("All operations completed successfully")
      }
    }.recover {
      case ex: Exception =>
        println(s"Error occurred: ${ex.getMessage}")
        ex.printStackTrace()
    }
  }

  def main(args: Array[String]): Unit = {
    println("=== MilvusStorage Scala API Examples ===")

    // Note: These examples assume you have valid Arrow schema and data pointers
    // In a real application, you would create these using Arrow Java libraries

    println("1. Write Example:")
    Try(writeExample()).recover { case ex => println(s"Write failed: ${ex.getMessage}") }

    println("\n2. Read Example:")
    Try(readExample()).recover { case ex => println(s"Read failed: ${ex.getMessage}") }

    println("\n3. Robust Example with Resource Management:")
    robustExample()

    println("\nExamples completed. Check the console output for results.")
  }
}