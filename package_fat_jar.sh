#!/bin/bash

# Milvus Storage JNI Fat JAR 打包脚本
set -e

echo "=== Start building Milvus Storage JNI Fat JAR ==="

PROJECT_ROOT=$(pwd)
CPP_BUILD_DIR="$PROJECT_ROOT/cpp/build"
SCALA_TEST_DIR="$PROJECT_ROOT/java"
DIST_DIR="$PROJECT_ROOT/dist"

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

cd "$PROJECT_ROOT/cpp"

export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/default-java}
make build

JNI_SO_PATH="$CPP_BUILD_DIR/Release/libmilvus_storage_jni.so"
MAIN_SO_PATH="$CPP_BUILD_DIR/Release/libmilvus-storage.so"

if [ ! -f "$JNI_SO_PATH" ]; then
    echo "build failed: libmilvus_storage_jni.so not found"
    find "$CPP_BUILD_DIR" -name "*milvus*jni*" -type f 2>/dev/null || echo "JNI library not found"
    exit 1
fi

if [ ! -f "$MAIN_SO_PATH" ]; then
    echo "build failed: libmilvus-storage.so not found"
    find "$CPP_BUILD_DIR" -name "*milvus-storage*" -type f 2>/dev/null || echo "Main library not found"
    exit 1
fi

echo "C++ libraries built successfully"

echo "prepare native libraries to Scala project"
cd "$SCALA_TEST_DIR"

mkdir -p native/linux-x86_64
mkdir -p native/linux-aarch64
mkdir -p native/darwin-x86_64
mkdir -p native/darwin-aarch64

# Copy both main library and JNI library
if [ -f "$MAIN_SO_PATH" ] && [ -f "$JNI_SO_PATH" ]; then
    cp "$MAIN_SO_PATH" native/linux-x86_64/
    cp "$JNI_SO_PATH" native/linux-x86_64/
    echo "copy main library to resource directory: $MAIN_SO_PATH"
    echo "copy JNI library to resource directory: $JNI_SO_PATH"
else
    echo "Libraries not found: $MAIN_SO_PATH or $JNI_SO_PATH"
    exit 1
fi

echo "detect system architecture and create corresponding directory"
ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

case "$OS-$ARCH" in
    linux-x86_64)
        TARGET_DIR="native/linux-x86_64"
        ;;
    linux-aarch64|linux-arm64)
        TARGET_DIR="native/linux-aarch64"
        ;;
    darwin-x86_64)
        TARGET_DIR="native/darwin-x86_64"
        ;;
    darwin-arm64)
        TARGET_DIR="native/darwin-aarch64"
        ;;
    *)
        echo "unknown platform: $OS-$ARCH, use linux-x86_64"
        TARGET_DIR="native/linux-x86_64"
        ;;
esac

echo "build Fat JAR"
echo "clean previous build..."
sbt clean

echo "compile and build Fat JAR..."
sbt assembly


ASSEMBLY_JAR="target/scala-2.13/milvus-storage-jni-fat.jar"
if [ -f "$ASSEMBLY_JAR" ]; then
    echo "Fat JAR built successfully!"

    # copy to distribution directory
    cp "$ASSEMBLY_JAR" "$DIST_DIR/"

    echo
    echo "Fat JAR location: $DIST_DIR/milvus-storage-jni-fat.jar"
    echo "file size: $(du -h "$DIST_DIR/milvus-storage-jni-fat.jar" | cut -f1)"

    # create usage example
    cat > "$DIST_DIR/usage_example.java" << 'EOF'
// usage example
import io.milvus.storage.*;

public class Example {
    public static void main(String[] args) {
        try {
            // Native library will be automatically loaded from JAR
            MilvusStorageProperties props = new MilvusStorageProperties();
            System.out.println("✓ Milvus Storage JNI initialized successfully");
        } catch (Exception e) {
            System.err.println("initialization failed: " + e.getMessage());
        }
    }
}
EOF

    echo
    echo "=== distribution package is ready ==="

else
    echo "Fat JAR build failed"
    echo "check sbt assembly output for details"
    exit 1
fi

echo
echo "=== build completed ==="