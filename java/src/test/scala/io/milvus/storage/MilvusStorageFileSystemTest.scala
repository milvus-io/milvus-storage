package io.milvus.storage

import java.nio.charset.StandardCharsets
import java.nio.file.{FileAlreadyExistsException, Files, Path}
import java.net.InetSocketAddress
import java.util.concurrent.{LinkedBlockingQueue, TimeUnit}
import com.sun.net.httpserver.HttpServer
import scala.collection.JavaConverters._
import org.scalatest.funsuite.AnyFunSuite

/** Exercises the public upstream filesystem binding against an actual local filesystem. */
class MilvusStorageFileSystemTest extends AnyFunSuite {
  private def withFileSystem(body: (MilvusStorageFileSystem, Path) => Unit): Unit = {
    val directory = Files.createTempDirectory("milvus-storage-filesystem-")
    val properties = new MilvusStorageProperties()
    var filesystem: MilvusStorageFileSystem = null
    try {
      properties.create(Map("fs.storage_type" -> "local", "fs.root_path" -> directory.toString))
      filesystem = new MilvusStorageFileSystem(properties, "")
      body(filesystem, directory)
    } finally {
      if (filesystem != null) filesystem.close()
      properties.free()
      val paths = Files.walk(directory)
      try paths.iterator().asScala.toSeq.reverse.foreach(Files.deleteIfExists(_))
      finally paths.close()
    }
  }

  test("local files preserve bytes, metadata and random-access ranges") {
    withFileSystem { (filesystem, _) =>
      val bytes = "upstream-jni-filesystem".getBytes(StandardCharsets.UTF_8)
      filesystem.createDir("nested", recursive = true)
      filesystem.writeFile("nested/data", bytes)
      assert(filesystem.exists("nested/data"))
      assert(filesystem.fileSize("nested/data") == bytes.length)
      assert(filesystem.readFileAll("nested/data").sameElements(bytes))
      val entries = filesystem.list("nested", recursive = false)
      assert(entries.length == 1)
      assert(entries.head.path.endsWith("nested/data"))
      assert(!entries.head.isDirectory)
      assert(entries.head.size == bytes.length)
      assert(entries.head.modifiedNanos > 0L)
      val reader = filesystem.openReader("nested/data", bytes.length)
      try {
        assert(reader.readAt(2L, 7L).sameElements(bytes.slice(2, 9)))
        assert(reader.readAt(bytes.length, 0L).isEmpty)
        intercept[IllegalArgumentException](reader.readAt(-1L, 1L))
        intercept[IllegalArgumentException](reader.readAt(0L, Int.MaxValue.toLong + 1L))
      } finally reader.close()
      reader.close()
      intercept[IllegalStateException](reader.readAt(0L, 1L))
      filesystem.deleteFile("nested/data")
      assert(!filesystem.exists("nested/data"))
    }
  }

  test("empty files round trip and missing files surface native failures") {
    withFileSystem { (filesystem, _) =>
      filesystem.writeFile("empty", Array.emptyByteArray)
      assert(filesystem.fileSize("empty") == 0L)
      assert(filesystem.readFileAll("empty").isEmpty)
      assert(!filesystem.exists("missing"))
      val error = intercept[MilvusStorageException](filesystem.readFileAll("missing"))
      assert(error.errorCode() != 0)
    }
  }

  test("exists propagates a backend error instead of reporting a missing path") {
    withFileSystem { (filesystem, _) =>
      // Arrow treats ELOOP as NotFound. An overlong path component produces
      // ENAMETOOLONG instead, even when the tests run as root.
      intercept[MilvusStorageException](filesystem.exists("x" * 1024))
    }
  }

  test("writers accept multiple writes and flush, and close is idempotent") {
    withFileSystem { (filesystem, _) =>
      filesystem.writeFile("data", Array[Byte](9, 9, 9, 9))
      val writer = filesystem.openWriter("data", conditional = false)
      try {
        writer.write(Array[Byte](0, 1))
        writer.flush()
        writer.write(Array[Byte](-1))
      } finally writer.close()
      writer.close()
      assert(filesystem.readFileAll("data").sameElements(Array[Byte](0, 1, -1)))
      intercept[IllegalStateException](writer.write(Array[Byte](2)))
      intercept[IllegalStateException](writer.flush())
    }
  }

  test("conditional writers preserve the local open conflict code and original bytes") {
    withFileSystem { (filesystem, _) =>
      val original = Array[Byte](0, 1, -1, 42)
      val writer = filesystem.openWriter("receipt", conditional = true)
      try writer.write(original)
      finally writer.close()
      // The local C FFI reports an existing file as a conflict, not a precondition failure.
      val error = intercept[MilvusStorageException](filesystem.openWriter("receipt", conditional = true))
      assert(error.errorCode() == 102) // loon_errcode_aws_conflict
      assert(error.getMessage.contains("File already exists"))
      assert(filesystem.readFileAll("receipt").sameElements(original))
      filesystem.writeFile("existing", original)
      val existing = intercept[MilvusStorageException](filesystem.openWriter("existing", conditional = true))
      assert(existing.errorCode() == 102)
      assert(filesystem.readFileAll("existing").sameElements(original))
    }
  }

  test("closing a conditional writer without writing creates an empty file") {
    withFileSystem { (filesystem, _) =>
      val writer = filesystem.openWriter("empty", conditional = true)
      writer.close()
      assert(filesystem.exists("empty"))
      val error = intercept[MilvusStorageException](filesystem.openWriter("empty", conditional = true))
      assert(error.errorCode() == 102)
      assert(filesystem.readFileAll("empty").isEmpty)
    }
  }

  test("writers reject invalid arguments and preserve native failures") {
    withFileSystem { (filesystem, _) =>
      intercept[IllegalArgumentException](filesystem.openWriter(null, conditional = true))
      val writer = filesystem.openWriter("data", conditional = true)
      try {
        intercept[IllegalArgumentException](writer.write(null))
        // The C FFI rejects zero-length writes. Empty files are created by closing without a write.
        val error = intercept[MilvusStorageException](writer.write(Array.emptyByteArray))
        assert(error.errorCode() == 1) // loon_errcode_invalid_args
        assert(error.getMessage.contains("size must not be null"))
        writer.write(Array[Byte](1))
      } finally writer.close()
      assert(filesystem.readFileAll("data").sameElements(Array[Byte](1)))
      val path = "x" * 1024
      val expected = intercept[MilvusStorageException](filesystem.writeFile(path, Array[Byte](1)))
      val actual = intercept[MilvusStorageException](filesystem.openWriter(path, conditional = true))
      assert(actual.errorCode() == expected.errorCode())
      assert(actual.getMessage.contains(path))
    }
  }

  test("openWriter rejects invalid metadata before opening a file") {
    withFileSystem { (filesystem, _) =>
      intercept[IllegalArgumentException] {
        filesystem.openWriter("invalid", null.asInstanceOf[Map[String, String]], conditional = true)
      }
      intercept[IllegalArgumentException] {
        filesystem.openWriter("invalid", null.asInstanceOf[java.util.Map[String, String]], conditional = true)
      }
      intercept[IllegalArgumentException] {
        filesystem.openWriter("invalid", Map(null.asInstanceOf[String] -> "value"), conditional = true)
      }
      intercept[IllegalArgumentException] {
        filesystem.openWriter("invalid", Map("key" -> null), conditional = true)
      }
      assert(!filesystem.exists("invalid"))
      val writer = filesystem.openWriter("empty-metadata", Map.empty[String, String], conditional = true)
      writer.close()
      assert(filesystem.readFileAll("empty-metadata").isEmpty)
    }
  }

  test("S3 writers forward metadata and distinguish conflicts from precondition failures") {
    val metadataRequests = new LinkedBlockingQueue[(String, String)]()
    val commitConditions = new LinkedBlockingQueue[Option[String]]()
    val server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0)
    server.createContext("/", exchange => {
      try {
        val headers = exchange.getRequestHeaders
        val query = Option(exchange.getRequestURI.getRawQuery).getOrElse("")
        val createMultipart = query == "uploads" || query.startsWith("uploads=")
        val completeMultipart = exchange.getRequestMethod == "POST" && query.contains("uploadId=")
        val putObject = exchange.getRequestMethod == "PUT" && query.isEmpty
        if (putObject || createMultipart)
          metadataRequests.add((headers.getFirst("Content-Type"), headers.getFirst("x-amz-meta-jni-key")))
        if (putObject || completeMultipart)
          commitConditions.add(Option(headers.getFirst("If-None-Match")))
        exchange.getRequestBody.readAllBytes()
        exchange.getResponseHeaders.set("ETag", "\"test-etag\"")
        val path = exchange.getRequestURI.getPath
        val status = if ((putObject || completeMultipart) && path.endsWith("/conflict-409")) 409
          else if ((putObject || completeMultipart) && path.endsWith("/conflict-412")) 412
          else 200
        // CRT streams use multipart even for small writes.
        val response = if (status == 409) {
          "<Error><Code>ConditionalRequestConflict</Code><Message>concurrent delete</Message></Error>"
        } else if (status == 412) {
          "<Error><Code>PreconditionFailed</Code><Message>object already exists</Message></Error>"
        } else if (createMultipart) {
          "<InitiateMultipartUploadResult><Bucket>metadata-test</Bucket><Key>data</Key>" +
            "<UploadId>test-upload</UploadId></InitiateMultipartUploadResult>"
        } else if (completeMultipart) {
          "<CompleteMultipartUploadResult><Bucket>metadata-test</Bucket><Key>data</Key>" +
            "<ETag>test-etag</ETag></CompleteMultipartUploadResult>"
        } else ""
        if (response.isEmpty) exchange.sendResponseHeaders(status, -1)
        else {
          val bytes = response.getBytes(StandardCharsets.UTF_8)
          exchange.getResponseHeaders.set("Content-Type", "application/xml")
          exchange.sendResponseHeaders(status, bytes.length)
          exchange.getResponseBody.write(bytes)
        }
      } finally exchange.close()
    })
    server.start()
    val properties = new MilvusStorageProperties()
    var filesystem: MilvusStorageFileSystem = null
    try {
      properties.create(Map(
        "fs.storage_type" -> "remote",
        "fs.cloud_provider" -> "aws",
        "fs.address" -> s"http://127.0.0.1:${server.getAddress.getPort}",
        "fs.bucket_name" -> "metadata-test",
        "fs.access_key_id" -> "test-key",
        "fs.access_key_value" -> "test-secret",
        "fs.region" -> "us-east-1",
        "fs.use_ssl" -> "false",
        "fs.use_iam" -> "false",
        "fs.s3.use_crt" -> sys.env.getOrElse("TEST_ENV_S3_USE_CRT", "false")
      ))
      filesystem = new MilvusStorageFileSystem(properties, "")
      for (conditional <- Seq(false, true)) {
        val metadata = Map("Content-Type" -> "application/x-jni-test", "jni-key" -> s"value-$conditional")
        val writer = if (conditional) filesystem.openWriter("data", metadata.asJava, conditional)
          else filesystem.openWriter("data", metadata, conditional)
        try writer.write(Array[Byte](1, 2, 3))
        finally writer.close()
        val request = metadataRequests.poll(5, TimeUnit.SECONDS)
        assert(request != null)
        assert(request._1 == "application/x-jni-test")
        assert(request._2 == s"value-$conditional")
        assert(commitConditions.poll(5, TimeUnit.SECONDS) == (if (conditional) Some("*") else None))
      }
      for (status <- Seq(409, 412)) {
        val writer = filesystem.openWriter(s"conflict-$status", conditional = true)
        try {
          writer.write(Array[Byte](1, 2, 3))
          if (status == 409) {
            val error = intercept[MilvusStorageException](writer.close())
            assert(error.errorCode() == 102)
            assert(error.getMessage.contains("concurrent delete"))
          } else {
            val error = intercept[FileAlreadyExistsException](writer.close())
            assert(error.getMessage.contains("object already exists"))
          }
          writer.close()
          intercept[IllegalStateException](writer.write(Array[Byte](1)))
          intercept[IllegalStateException](writer.flush())
        } finally writer.close()
      }
    } finally {
      if (filesystem != null) filesystem.close()
      properties.free()
      server.stop(0)
    }
  }

  test("a provider without conditional uploads reports unsupported") {
    val properties = new MilvusStorageProperties()
    var filesystem: MilvusStorageFileSystem = null
    try {
      // Huawei rejects conditional uploads before issuing any network request.
      properties.create(Map(
        "fs.storage_type" -> "remote",
        "fs.cloud_provider" -> "huawei",
        "fs.address" -> "http://127.0.0.1:1",
        "fs.bucket_name" -> "unused",
        "fs.access_key_id" -> "unused",
        "fs.access_key_value" -> "unused",
        "fs.region" -> "us-east-1",
        "fs.use_ssl" -> "false",
        "fs.use_iam" -> "false"
      ))
      filesystem = new MilvusStorageFileSystem(properties, "")
      val error = intercept[UnsupportedOperationException] {
        filesystem.openWriter("receipt", conditional = true)
      }
      assert(error.getMessage.contains("Conditional uploads are not supported"))
      assert(error.getMessage.contains("huawei"))
    } finally {
      if (filesystem != null) filesystem.close()
      properties.free()
    }
  }

  test("S3 conditional writers preserve existing objects on write or close conflicts") {
    assume(sys.env.get("TEST_ENV_STORAGE_TYPE").contains("remote") &&
      sys.env.get("TEST_ENV_CLOUD_PROVIDER").contains("aws"), "Requires MinIO or S3 test environment")
    val properties = new MilvusStorageProperties()
    val root = "jni-conditional-" + java.util.UUID.randomUUID().toString
    val useIam = sys.env.get("TEST_ENV_USE_IAM").exists(value => value == "true" || value == "1")
    val useSsl = sys.env.get("TEST_ENV_USE_SSL").map(value => value == "true" || value == "1")
      .getOrElse(sys.env("TEST_ENV_ADDRESS").startsWith("https://"))
    var filesystem: MilvusStorageFileSystem = null
    try {
      properties.create(Map(
        "fs.storage_type" -> "remote",
        "fs.cloud_provider" -> "aws",
        "fs.address" -> sys.env("TEST_ENV_ADDRESS"),
        "fs.bucket_name" -> sys.env("TEST_ENV_BUCKET_NAME"),
        "fs.access_key_id" -> (if (useIam) "" else sys.env("TEST_ENV_ACCESS_KEY")),
        "fs.access_key_value" -> (if (useIam) "" else sys.env("TEST_ENV_SECRET_KEY")),
        "fs.region" -> sys.env.getOrElse("TEST_ENV_REGION", "us-east-1"),
        "fs.root_path" -> root,
        "fs.use_ssl" -> useSsl.toString,
        "fs.use_iam" -> useIam.toString,
        "fs.s3.use_crt" -> sys.env.getOrElse("TEST_ENV_S3_USE_CRT", "false")
      ))
      filesystem = new MilvusStorageFileSystem(properties, "")
      // Exercise empty uploads, single-part uploads, and multipart completion.
      for (size <- Seq(0, 4, 11 * 1024 * 1024)) {
        val path = s"receipt-$size"
        val original = Array.fill[Byte](size)(42)
        val metadata = Map("Content-Type" -> "application/x-jni-test", "jni-key" -> s"receipt-$size")
        val writer = filesystem.openWriter(path, metadata, conditional = true)
        try {
          if (size != 0) writer.write(original)
          writer.flush()
        } finally writer.close()
        try {
          val conflicting = filesystem.openWriter(path, conditional = true)
          val error = intercept[FileAlreadyExistsException] {
            // Keep a write failure as the primary error if close also reports a conflict.
            scala.util.Using.resource(conflicting) { stream =>
              if (size != 0) stream.write(Array.fill[Byte](size)(7))
            }
          }
          assert(error.getMessage.contains("PreConditionFailed"))
          conflicting.close()
          intercept[IllegalStateException](conflicting.write(Array[Byte](1)))
          intercept[IllegalStateException](conflicting.flush())
          assert(filesystem.readFileAll(path).sameElements(original))
        } finally filesystem.deleteFile(path)
      }
    } finally {
      if (filesystem != null) filesystem.close()
      properties.free()
    }
  }

  test("filesystem closure is idempotent and rejects later calls") {
    withFileSystem { (filesystem, _) =>
      filesystem.close()
      filesystem.close()
      intercept[IllegalStateException](filesystem.exists("missing"))
      intercept[IllegalStateException](filesystem.readFileAll("missing"))
      intercept[IllegalStateException](filesystem.openWriter("missing", conditional = true))
    }
  }
}
