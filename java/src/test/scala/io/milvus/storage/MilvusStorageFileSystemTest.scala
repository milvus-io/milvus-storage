package io.milvus.storage

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
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

  test("filesystem closure is idempotent and rejects later calls") {
    withFileSystem { (filesystem, _) =>
      filesystem.close()
      filesystem.close()
      intercept[IllegalStateException](filesystem.exists("missing"))
      intercept[IllegalStateException](filesystem.readFileAll("missing"))
    }
  }

  test("column group metadata preserves format and rejects invalid row counts") {
    val columns = Array(Array("100", "101"))
    val files = Array(Array("one.parquet", "two.parquet"))
    val rows = Array(Array(2L, 3L))
    val groups = MilvusStorageColumnGroups.createFromGroups(columns, files, rows, "vortex")
    try {
      assert(MilvusStorageColumnGroups.count(groups) == 1)
      assert(MilvusStorageColumnGroups.columns(groups, 0).sameElements(columns(0)))
      assert(MilvusStorageColumnGroups.files(groups, 0).sameElements(files(0)))
      assert(MilvusStorageColumnGroups.fileRowCounts(groups, 0).sameElements(rows(0)))
      assert(MilvusStorageColumnGroups.format(groups, 0) == "vortex")
      intercept[IndexOutOfBoundsException](MilvusStorageColumnGroups.files(groups, 1))
    } finally MilvusStorageColumnGroups.destroy(groups)
    intercept[IllegalArgumentException] {
      MilvusStorageColumnGroups.createFromGroups(columns, files, Array(Array(-1L, 3L)))
    }
  }
}
