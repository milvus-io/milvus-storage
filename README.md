# Milvus Storage

A high-performance columnar storage engine built on Apache Arrow, designed for vector databases and analytical workloads.

[![C++ CI](https://github.com/milvus-io/milvus-storage/actions/workflows/cpp-ci.yml/badge.svg)](https://github.com/milvus-io/milvus-storage/actions/workflows/cpp-ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Architecture

```
┌───────────────────────────────────────────────────────────────┬─────────────┐
│                       Application Layer                       │ Filesystem  │
│                   (Python / Java / Rust / C++)                │ FFI (C ABI) │
└───────────────────────────────────────────────────────────────┴──────┬──────┘
                                      │                                │
                                      ▼                                │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FFI Layer (extern "C")                              │
│                    (Cross-language bindings via C ABI)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │                                │
                          ┌───────────┴───────────┐                    │
                          ▼                       ▼                    │
┌─────────────────────────────────────┐ ┌─────────────────────────────────────┐
│            Writer API               │ │            Reader API               │
│  ┌───────────────────────────────┐  │ │  ┌───────────────────────────────┐  │
│  │    Column Group Policy        │  │ │  │   RecordBatchReader (Scan)    │  │
│  │  (Single/Schema/Size Based)   │  │ │  │   ChunkReader (Random Access) │  │
│  └───────────────────────────────┘  │ │  │   Take (Row Indices)          │  │
└─────────────────────────────────────┘ └─────────────────────────────────────┘
                          │                       │                    │
                          └───────────┬───────────┘                    │
                                      ▼                                │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Transaction Layer                                 │
│         (Manifest Versioning / Conflict Resolution / Delta Logs)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │                                │
                          ┌───────────┴───────────┐                    │
                          ▼                       ▼                    │
┌─────────────────────────────────────┐ ┌─────────────────────────────────────┐
│       Column Group Writer           │ │        Column Group Reader          │
│  ┌───────────────────────────────┐  │ │  ┌───────────────────────────────┐  │
│  │  Buffer Management            │  │ │  │  Chunk Management             │  │
│  │  Row Group Sizing             │  │ │  │  Column Projection            │  │
│  │  File Rolling                 │  │ │  │  Predicate Pushdown           │  │
│  └───────────────────────────────┘  │ │  └───────────────────────────────┘  │
└─────────────────────────────────────┘ └─────────────────────────────────────┘
                          │                       │                    │
                          └───────────┬───────────┘                    │
                                      ▼                                │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Format Layer                                     │
│    ┌─────────────┐      ┌─────────────┐      ┌───────────────────────────┐ │
│    │   Parquet   │      │   Vortex    │      │    Lance (Read Only)      │ │
│    └─────────────┘      └─────────────┘      └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │                                │
                                      ▼                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Filesystem Layer                                   │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌────────┐ ┌────────┐  │
│  │ Local │ │AWS S3 │ │  GCS  │ │ Azure │ │Aliyun │ │Tencent │ │ Huawei │  │
│  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └────────┘ └────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Column Group Storage** - Organize columns into groups for optimal I/O and compression
- **Multi-Format Support** - Parquet (primary), Vortex, and Lance formats
- **Transaction Support** - ACID-like semantics with manifest versioning and conflict resolution
- **Cloud Native** - Built-in support for major cloud storage providers
- **Multi-Language SDKs** - Python, Java/Scala, Rust, and C++ bindings
- **Encryption & Compression** - Data-at-rest encryption and configurable compression

## Cloud Storage Support

| Provider | Status |
|----------|--------|
| AWS S3 | Supported (including S3-compatible: MinIO, Cloudflare R2) |
| Google Cloud Storage | Supported |
| Azure Blob Storage | Supported |
| Aliyun OSS | Supported |
| Tencent COS | Supported |
| Huawei Cloud OBS | Supported |

## Language SDKs

| Language | Status | Notes |
|----------|--------|-------|
| C++ | Primary | Core implementation |
| Python | Supported | FFI bindings with PyArrow integration |
| Java/Scala | Supported | JNI bindings |
| Rust | Supported | DataFusion TableProvider integration |

## Development

### Prerequisites

- CMake >= 3.20.0
- C++17 compiler (GCC 8+, Clang 6+)
- Conan >= 1.60.0 and <= 2.0.0


### Build from Source (C++)

```bash
git clone https://github.com/milvus-io/milvus-storage.git
cd milvus-storage/cpp

# Setup Conan remote
conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local --insert 0

# Build
make build

# Test
make test

# Test with minio
make test-all
```

#### Build Options

| Option | Description |
|--------|-------------|
| `BUILD_TYPE=Debug/Release` | Build type |
| `WITH_AZURE_FS=ON` | Azure filesystem support |
| `WITH_JNI=ON` | Java JNI bindings |
| `WITH_PYTHON_BINDING=ON` | Python bindings |

## Quick Start

See [python/tests/test_write_read.py](python/tests/test_write_read.py) for Python usage examples.

For old storage(packed interface) integration, see [cpp/test/packed/packed_integration_test.cpp](cpp/test/packed/packed_integration_test.cpp).

### Code Style

```bash
make fix-format
make fix-tidy
```

## Contributing

Contributions are welcome. Please ensure code follows the project style and tests pass.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
