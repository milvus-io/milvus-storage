package io.milvus.storage

import java.net.URL
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.util.jar.{JarEntry, JarOutputStream}
import scala.jdk.CollectionConverters._

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class NativeLibraryLoaderTest extends AnyFunSuite with Matchers {
  private def removeTree(path: Path): Unit = {
    val paths = Files.walk(path)
    try paths.iterator().asScala.toVector.reverse.foreach(Files.delete)
    finally paths.close()
  }

  test("JAR extraction selects one platform and keeps nested and versioned libraries") {
    val work = Files.createTempDirectory("storage-loader-jar-test")
    val platform = NativeLibraryLoader.currentPlatform()
    val prefix = s"native/$platform/"
    val jniName = System.mapLibraryName("milvus-storage-jni")
    val jar = work.resolve("native.jar")
    val output = new JarOutputStream(Files.newOutputStream(jar))
    try {
      Seq(
        prefix + jniName -> "jni",
        prefix + "libdependency.so.1" -> "selected",
        prefix + "ossl-modules/legacy.so" -> "provider",
        "native/another-platform/libdependency.so.1" -> "wrong platform",
        "native/knowhere/1/linux-x86_64/libdependency.so.1" -> "other loader"
      ).foreach { case (name, value) =>
        output.putNextEntry(new JarEntry(name))
        output.write(value.getBytes(UTF_8))
        output.closeEntry()
      }
    } finally output.close()

    val resource = new URL(s"jar:${jar.toUri}!/$prefix$jniName")
    val first = NativeLibraryLoader.extractLibraries(resource, prefix)
    val second = NativeLibraryLoader.extractLibraries(resource, prefix)
    try {
      first should not be second
      new String(Files.readAllBytes(first.resolve("libdependency.so.1")), UTF_8) shouldBe "selected"
      new String(Files.readAllBytes(first.resolve("ossl-modules/legacy.so")), UTF_8) shouldBe "provider"
      Files.exists(first.resolve("another-platform")) shouldBe false
      Files.exists(first.resolve("knowhere")) shouldBe false
    } finally {
      removeTree(first)
      removeTree(second)
      removeTree(work)
    }
  }

  test("filesystem extraction reads only the JNI resource's platform directory") {
    val work = Files.createTempDirectory("storage-loader-directory-test")
    val prefix = s"native/${NativeLibraryLoader.currentPlatform()}/"
    val jniName = System.mapLibraryName("milvus-storage-jni")
    val jni = work.resolve(prefix + jniName)
    Files.createDirectories(jni.getParent)
    Files.write(jni, "jni".getBytes(UTF_8))
    Files.write(jni.getParent.resolve("libdependency.so.1"), "selected".getBytes(UTF_8))
    val foreign = work.resolve("native/another-platform/libdependency.so.1")
    Files.createDirectories(foreign.getParent)
    Files.write(foreign, "wrong platform".getBytes(UTF_8))

    val extracted = NativeLibraryLoader.extractLibraries(jni.toUri.toURL, prefix)
    try {
      new String(Files.readAllBytes(extracted.resolve("libdependency.so.1")), UTF_8) shouldBe "selected"
    } finally {
      removeTree(extracted)
      removeTree(work)
    }
  }
}
