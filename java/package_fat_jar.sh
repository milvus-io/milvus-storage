#!/bin/bash

# Milvus Storage JNI Fat JAR pack script
set -e

echo "=== Start building Milvus Storage JNI Fat JAR ==="

if test -n "${ZSH_VERSION:-}"; then
    # zsh
    SCRIPT_PATH="${(%):-%x}"
elif test -n "${BASH_VERSION:-}"; then
    # bash
    SCRIPT_PATH="${BASH_SOURCE[0]}"
else
    # Unknown shell, hope below works.
    # Tested with dash
    result=$(lsof -p $$ -Fn | tail --lines=1 | xargs --max-args=2 | cut --delimiter=' ' --fields=2)
    SCRIPT_PATH=${result#n}
fi

if test -z "$SCRIPT_PATH"; then
    echo "The shell cannot be identified. Current script may not be set correctly" >&2
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" >/dev/null 2>&1 && pwd)"

PROJECT_ROOT="$SCRIPT_DIR/../"
CPP_BUILD_DIR="$PROJECT_ROOT/cpp/build"
SCALA_TEST_DIR="$PROJECT_ROOT/java"
DIST_DIR="$PROJECT_ROOT/dist"

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

cd "$PROJECT_ROOT/cpp"

export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/default-java}
make build

JNI_SO_PATH="$CPP_BUILD_DIR/Release/libmilvus-storage-jni.so"
MAIN_SO_PATH="$CPP_BUILD_DIR/Release/libmilvus-storage.so"

if [ ! -f "$JNI_SO_PATH" ]; then
    echo "build failed: libmilvus-storage-jni.so not found"
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