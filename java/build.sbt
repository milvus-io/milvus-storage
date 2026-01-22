ThisBuild / organizationName := "zilliz"
ThisBuild / organizationHomepage := Some(url("https://zilliz.com/"))
ThisBuild / organization := "com.zilliz"

ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"

// POM metadata required for Maven Central
ThisBuild / description := "Milvus Storage JNI bindings for Java/Scala"
ThisBuild / licenses := List("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0"))
ThisBuild / homepage := Some(url("https://github.com/milvus-io/milvus-storage"))
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/milvus-io/milvus-storage"),
    "scm:git@github.com:milvus-io/milvus-storage.git"
  )
)
ThisBuild / developers := List(
  Developer(
    id = "milvus-io",
    name = "Milvus Team",
    email = "milvus-team@zilliz.com",
    url = url("https://milvus.io")
  )
)

// Publishing settings
ThisBuild / publishMavenStyle := true
ThisBuild / versionScheme := Some("early-semver")

// Sonatype Central settings for Maven Central
ThisBuild / sonatypeCredentialHost := "central.sonatype.com"
ThisBuild / sonatypeRepository := "https://central.sonatype.com/api/v1/publisher"

// publishSigned stages locally, then sonatypeCentralUpload pushes to Central
ThisBuild / publishTo := sonatypePublishToBundle.value

// GitHub Packages credentials (for publish command)
ThisBuild / credentials ++= (for {
  actor <- sys.env.get("GITHUB_ACTOR")
  token <- sys.env.get("GITHUB_TOKEN")
} yield Credentials("GitHub Package Registry", "maven.pkg.github.com", actor, token)).toSeq

// Sonatype credentials (for publishSigned command)
ThisBuild / credentials ++= (for {
  username <- sys.env.get("MAVEN_USERNAME")
  password <- sys.env.get("MAVEN_PASSWORD")
} yield Credentials("Sonatype Nexus Repository Manager", "central.sonatype.com", username, password)).toSeq

// GPG signing settings for Maven Central
ThisBuild / useGpgPinentry := true
ThisBuild / pgpPassphrase := sys.env.get("PGP_PASSPHRASE").map(_.toCharArray)

lazy val root = (project in file("."))
  .settings(
    name := "milvus-storage-jni",

    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % "3.2.17" % Test,
      "org.apache.arrow" % "arrow-vector" % "14.0.1",
      "org.apache.arrow" % "arrow-memory-netty" % "14.0.1",
      "org.apache.arrow" % "arrow-c-data" % "14.0.1",
      "org.scala-lang.modules" %% "scala-collection-compat" % "2.11.0",
      "org.apache.spark" %% "spark-sql" % "3.5.0" % "provided"
    ),

    // Fork JVM for tests to properly load native library
    Test / fork := true,
    run / fork := true,

    // JVM options for run
    run / javaOptions ++= Seq(
      "-Djava.library.path=.",
      "--add-opens=java.base/java.nio=ALL-UNNAMED"
    ),

    run / envVars := Map(
      "LD_PRELOAD" -> s"${baseDirectory.value}/../cpp/build/Release/libmilvus-storage.so"
    ),

    // Additional JVM options for better debugging and TLS handling
    Test / javaOptions ++= Seq(
      "-Xss512k",
      "-Xmx2g",
      "-verbose:jni",
      // Library path for native dependencies
      "-Djava.library.path=.",
      // Required for Arrow C Data Interface
      "--add-opens=java.base/java.nio=ALL-UNNAMED"
    ),

    Test / envVars := Map(
      "LD_PRELOAD" -> s"${baseDirectory.value}/../cpp/build/Release/libmilvus-storage.so"
    ),

    // Include native libraries in resources for fat jar
    Compile / unmanagedResourceDirectories += baseDirectory.value / "native",

    // Fat jar assembly settings
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) => xs match {
        case "MANIFEST.MF" :: Nil => MergeStrategy.discard
        case "module-info.class" :: Nil => MergeStrategy.discard
        case _ => MergeStrategy.first
      }
      case PathList("mozilla", xs @ _*) => MergeStrategy.first
      case PathList("google", "protobuf", xs @ _*) => MergeStrategy.first
      case PathList("native", xs @ _*) => MergeStrategy.first
      case x if x.endsWith(".so") => MergeStrategy.first
      case x if x.endsWith(".dylib") => MergeStrategy.first
      case x if x.endsWith(".dll") => MergeStrategy.first
      case x if x.contains("arrow") => MergeStrategy.first
      case _ => MergeStrategy.deduplicate
    },

    // Assembly jar name
    assembly / assemblyJarName := "milvus-storage-jni-fat.jar",

    // Exclude test files from fat jar
    assembly / assemblyExcludedJars := {
      val cp = (assembly / fullClasspath).value
      cp.filter(_.data.getName.contains("scalatest"))
    }
  )