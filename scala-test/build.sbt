ThisBuild / organizationName := "zilliz"
ThisBuild / organizationHomepage := Some(url("https://zilliz.com/"))

ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"

lazy val root = (project in file("."))
  .settings(
    name := "milvus-storage-jni-test",

    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % "3.2.17" % Test
    ),

    // Fork JVM for tests to properly load native library
    Test / fork := true,
    run / fork := true,

    // Additional JVM options for better debugging and TLS handling
    Test / javaOptions ++= Seq(
      "-Xmx2g",
      "-verbose:jni",
      // Library path for native dependencies
      "-Djava.library.path=."
    ),

    // // Fat jar assembly settings
    // assembly / assemblyMergeStrategy := {
    //   case PathList("META-INF", xs @ _*) => xs match {
    //     case ("MANIFEST.MF" :: Nil) => MergeStrategy.discard
    //     case _ => MergeStrategy.first
    //   }
    //   case "native" => MergeStrategy.first
    //   case _ => MergeStrategy.first 
    // },

    // Include native libraries in resources
    Compile / unmanagedResourceDirectories += baseDirectory.value / "native",

    // // Assembly jar name
    // assembly / assemblyJarName := "milvus-storage-jni-fat.jar"
  )