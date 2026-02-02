# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Milvus Storage is a multi-language columnar storage engine built on Apache Arrow. The core is implemented in C++ with FFI bindings for Python, Java, and Rust. It supports Parquet, Vortex, and Lance (read-only) formats with cloud storage backends (AWS S3, GCS, Azure, Aliyun, Tencent, Huawei).

## Build Commands

All C++ commands run from `/cpp` directory:

```bash
# Build (Release with ASAN by default)
make build

# Build variants
BUILD_TYPE=Debug make build
USE_ASAN=False make build

# Build for language bindings
make python-lib    # Python FFI
make java-lib      # Java JNI

# Run tests
make test          # Unit tests
make test-all      # Tests with MinIO

# Cloud storage tests
make test-cloud-storage        # All providers
make test-cloud-storage aws    # Specific provider (aws/gcp/azure/aliyun/tencent/huawei)

# Code quality
make fix-format    # Apply clang-format
make check-format  # Verify formatting
make fix-tidy      # Apply clang-tidy fixes
make check-tidy    # Run clang-tidy
```

Python (`/python`):
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Rust (`/rust`):
```bash
cargo build --release
cargo test
cargo fmt && cargo clippy -- -D warnings
```

Java (`/java`):
```bash
sbt compile
sbt test
```

## Architecture

```
Application Layer (Python/Java/Rust/C++) + Filesystem FFI
                    │                           │
                    ▼                           │
            FFI Layer (extern "C")              │
                    │                           │
        ┌───────────┴───────────┐               │
        ▼                       ▼               │
   Writer API              Reader API           │
        │                       │               │
        └───────────┬───────────┘               │
                    ▼                           │
           Transaction Layer                    │
    (Manifest/Conflict Resolution)              │
                    │                           │
        ┌───────────┴───────────┐               │
        ▼                       ▼               │
 Column Group Writer    Column Group Reader     │
                    │                           │
                    ▼                           │
             Format Layer                       │
      (Parquet/Vortex/Lance)                    │
                    │                           │
                    ▼                           ▼
              Filesystem Layer
    (Local/S3/GCS/Azure/Aliyun/Tencent/Huawei)
```

## Key Source Locations

- **Public Headers**: `cpp/include/milvus-storage/`
  - `writer.h`, `reader.h` - High-level APIs
  - `transaction/transaction.h` - Transaction support
  - `manifest.h`, `column_groups.h` - Metadata structures
  - `properties.h` - Configuration properties
  - `ffi_c.h`, `ffi_filesystem_c.h` - FFI interfaces

- **Core Implementation**: `cpp/src/`
  - `format/` - Parquet, Vortex format readers/writers
  - `filesystem/` - Cloud storage backends (s3/, azure/)
  - `packed/` - Column group management
  - `ffi/` - FFI layer implementation
  - `jni/` - Java JNI bindings

- **Language Bindings**:
  - `python/milvus_storage/_ffi.py` - Python FFI wrapper
  - `rust/src/ffi.rs` - Rust FFI bindings
  - `java/src/main/scala/` - Scala JNI wrappers

- **Tests**: `cpp/test/`
  - `api_writer_reader_test.cpp` - Core API tests
  - `api_transaction_test.cpp` - Transaction tests
  - `packed/packed_integration_test.cpp` - Cloud integration tests

## Build Prerequisites

- CMake >= 3.20.0
- C++17 compiler (GCC 8+, Clang 6+)
- Conan >= 1.60.0 and < 2.0.0

Setup Conan remote before first build:
```bash
conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local --insert 0
```

## Code Conventions

- C++ follows Google style with 120 column limit (`.clang-format`)
- Tests use Google Test framework
- Arrow Status/Result pattern for error handling
