ThisBuild / organizationName := "zilliz"
ThisBuild / organizationHomepage := Some(url("https://zilliz.com/"))

ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"

lazy val root = (project in file("."))
  .settings(
    name := "milvus-storage-jni-test",

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