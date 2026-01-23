# Milvus Storage Python Integration Test Design

## Overview

This document describes the design of a comprehensive integration test suite for milvus-storage, using pytest framework based on Python FFI. The test suite covers core functionality and stress testing scenarios.

---

## Implementation Checklist

### Prerequisites

| Item | Description | Status |
|------|-------------|--------|
| Python FFI Complete | Python FFI fully synchronized with C FFI, including Transaction, External Table APIs | ✅ |
| pytest Environment | pytest + pyarrow + cffi dependencies installed | ✅ |
| C++ Library Build | `make python-lib` builds successfully | ✅ |
| MinIO Environment | MinIO service available in CI environment | ⬜ |

### Feature Dependencies

| Test Category | Required Feature | Status |
|---------------|------------------|--------|
| write_read | Writer/Reader basic API | ✅ |
| transaction | Transaction.append_files, add_column_group | ✅ |
| manifest | Manifest version read | ✅ |
| schema_evolution | ColumnGroupPolicy API | ⬜ |
| external_table | loon_exttable_* FFI interfaces | ⬜ |
| recovery | **Fault Injection Mechanism** (see below) | ⬜ |

### Fault Injection Mechanism (Required for recovery tests)

> ⚠️ **Note**: `recovery/` tests require a fault injection mechanism to be implemented first.

Use [libfiu](https://blitiri.com.ar/p/libfiu/) (Fault Injection in Userspace) as the fault injection framework.

**C++ Integration:**

```cpp
// Include libfiu header
#include <fiu.h>
#include <fiu-control.h>

// Add fault points in C++ code
Status Writer::Flush() {
    fiu_return_on("writer.flush.fail", Status::IOError("Injected fault"));
    // ... normal flush logic
}

Status Manifest::Commit() {
    fiu_return_on("manifest.commit.fail", Status::IOError("Injected fault"));
    // ... normal commit logic
}
```

**Fault Point Definitions:**

| Fault Point | Location | Description |
|-------------|----------|-------------|
| `writer.flush.fail` | Writer::Flush | Fail during flush |
| `writer.close.fail` | Writer::Close | Fail during close |
| `manifest.commit.fail` | Manifest::Commit | Fail during commit |
| `manifest.read.fail` | Manifest::Read | Fail during read |
| `fs.write.fail` | FileSystem::Write | Fail during file write |

**FFI Interface:**

```cpp
// cpp/include/milvus-storage/ffi_c.h
FFI_EXPORT LoonFFIResult loon_fiu_enable(const char* name, int failnum);
FFI_EXPORT LoonFFIResult loon_fiu_disable(const char* name);
FFI_EXPORT void loon_fiu_disable_all();
```

**Python Usage Example:**

```python
import pytest

@pytest.fixture
def fiu():
    """Fault injection fixture"""
    yield FaultInjector()
    # Cleanup: disable all fault points after test
    loon_fiu_disable_all()

def test_recovery_after_flush_fail(fiu):
    # Enable fault point (fail once)
    loon_fiu_enable("writer.flush.fail", 1)

    writer = Writer(path, schema, properties)
    writer.write(batch)

    with pytest.raises(IOError):
        writer.flush()  # Fails here

    # Retry should succeed (failnum exhausted)
    writer.flush()
    writer.close()
```

### Test Data Dependencies

| Test Category | Required Test Data | Status |
|---------------|-------------------|--------|
| manifest/version_upgrade | Legacy manifest files (v1, v2 format) | ⬜ |
| external_table | Pre-generated Parquet/Vortex/Lance test files | ⬜ |

---

## Test Categories

The integration tests are organized into **6 categories**:

| Category | Description | Test Modules |
|----------|-------------|--------------|
| **write_read** | Core read/write functionality | file_rolling, reader_advanced, chunk_reader, compression, encryption, boundary_conditions |
| **transaction** | Transaction workflows | append, add_field, mix_workflow, concurrent_mix_workflow |
| **manifest** | Manifest operations | version_upgrade |
| **schema_evolution** | Schema evolution | schema_evolution, column_group_policy |
| **external_table** | External table import | parquet_import, vortex_import, lance_import |
| **recovery** | Fault tolerance and recovery | crash_recovery, data_validation |

## Test Directory Structure

```
milvus-storage/                      # Project root
├── cpp/                             # C++ implementation
├── python/                          # Python FFI bindings
├── java/                            # Java bindings
├── docs/                            # Documentation
└── tests/                           # Integration tests (NEW)
    ├── __init__.py
    ├── conftest.py                  # Global pytest fixtures
    ├── pytest.ini                   # pytest configuration
    ├── config.yaml                  # Storage backend configuration (DEFAULT)
    ├── config.py                    # Configuration loader
    ├── integration/
    │   ├── __init__.py
    │   ├── conftest.py              # Integration test fixtures
    │   │
    │   ├── write_read/              # Core read/write tests
    │   │   ├── __init__.py
    │   │   ├── test_file_rolling.py
    │   │   ├── test_reader_advanced.py
    │   │   ├── test_chunk_reader.py
    │   │   ├── test_compression.py
    │   │   ├── test_encryption.py
    │   │   └── test_boundary_conditions.py
    │   │
    │   ├── transaction/             # Transaction workflow tests
    │   │   ├── __init__.py
    │   │   ├── test_append.py
    │   │   ├── test_add_field.py
    │   │   ├── test_mix_workflow.py
    │   │   └── test_concurrent_mix_workflow.py
    │   │
    │   ├── manifest/                # Manifest operation tests
    │   │   ├── __init__.py
    │   │   └── test_version_upgrade.py
    │   │
    │   ├── schema_evolution/        # Schema evolution tests
    │   │   ├── __init__.py
    │   │   ├── test_schema_evolution.py
    │   │   └── test_column_group_policy.py
    │   │
    │   ├── external_table/          # External table import tests
    │   │   ├── __init__.py
    │   │   ├── test_parquet_import.py
    │   │   ├── test_vortex_import.py
    │   │   └── test_lance_import.py
    │   │
    │   ├── recovery/                # Recovery and fault tolerance tests
    │   │   ├── __init__.py
    │   │   ├── test_crash_recovery.py
    │   │   └── test_data_validation.py
    │   │
    │   └── test_error_handling.py   # Cross-category error handling
    │
    └── stress/                      # Stress tests
        ├── __init__.py
        ├── conftest.py
        ├── test_large_scale_write.py
        ├── test_high_concurrency.py
        └── test_long_running.py
```

---

## Storage Backend Configuration

All tests can run against different storage backends by switching a single configuration file.

### Supported Storage Backends

| Backend | Type | Description |
|---------|------|-------------|
| `local` | Local | Local filesystem |
| `minio` | S3-compatible | MinIO (for CI/local testing) |
| `aws` | S3 | Amazon S3 |
| `gcs` | GCS | Google Cloud Storage |
| `azure` | Azure | Azure Blob Storage |
| `aliyun` | S3-compatible | Alibaba Cloud OSS |
| `tencent` | S3-compatible | Tencent Cloud COS |
| `huawei` | S3-compatible | Huawei Cloud OBS |

### Configuration File (`tests/config.yaml`)

```yaml
# Storage backend configuration
# Copy to config.local.yaml for local overrides (gitignored)

# Data format: parquet | vortex
format: parquet

# Active backend: local | minio | aws | gcs | azure | aliyun | tencent | huawei
storage_backend: local

# Base path (relative to SubtreeFilesystem root)
# For local: relative to root_path
# For cloud: relative to bucket_name
base_path: integration-tests

# Local filesystem
local:
  root_path: /tmp/milvus-storage-test

# MinIO (S3-compatible, for CI)
minio:
  cloud_provider: aws
  address: http://localhost:9000
  bucket_name: milvus-test
  access_key: minioadmin
  secret_key: minioadmin
  region: us-east-1

# Amazon S3
aws:
  cloud_provider: aws
  address: ${AWS_ADDRESS}
  bucket_name: ${AWS_BUCKET_NAME}
  access_key: ${AWS_ACCESS_KEY}
  secret_key: ${AWS_SECRET_KEY}
  region: ${AWS_REGION}

# Google Cloud Storage
gcs:
  cloud_provider: gcp
  address: ${GCS_ADDRESS}
  bucket_name: ${GCS_BUCKET_NAME}
  access_key: ${GCS_ACCESS_KEY}
  secret_key: ${GCS_SECRET_KEY}
  region: ${GCS_REGION}

# Azure Blob Storage
azure:
  cloud_provider: azure
  address: ${AZURE_ADDRESS}
  bucket_name: ${AZURE_BUCKET_NAME}
  access_key: ${AZURE_ACCESS_KEY}
  secret_key: ${AZURE_SECRET_KEY}
  region: ${AZURE_REGION}

# Alibaba Cloud OSS
aliyun:
  cloud_provider: aliyun
  address: ${ALIYUN_ADDRESS}
  bucket_name: ${ALIYUN_BUCKET_NAME}
  access_key: ${ALIYUN_ACCESS_KEY}
  secret_key: ${ALIYUN_SECRET_KEY}
  region: ${ALIYUN_REGION}

# Tencent Cloud COS
tencent:
  cloud_provider: tencent
  address: ${TENCENT_ADDRESS}
  bucket_name: ${TENCENT_BUCKET_NAME}
  access_key: ${TENCENT_ACCESS_KEY}
  secret_key: ${TENCENT_SECRET_KEY}
  region: ${TENCENT_REGION}

# Huawei Cloud OBS
huawei:
  cloud_provider: huawei
  address: ${HUAWEI_ADDRESS}
  bucket_name: ${HUAWEI_BUCKET_NAME}
  access_key: ${HUAWEI_ACCESS_KEY}
  secret_key: ${HUAWEI_SECRET_KEY}
  region: ${HUAWEI_REGION}
```

### Switching Backends

```bash
# Local filesystem (default)
pytest tests/integration/ -v

# MinIO (CI)
TEST_CONFIG_FILE=tests/config.minio.yaml pytest tests/integration/ -v

# Alibaba Cloud OSS
export ALIYUN_ACCESS_KEY_ID=xxx
export ALIYUN_ACCESS_KEY_SECRET=xxx
export ALIYUN_OSS_BUCKET=my-bucket
TEST_CONFIG_FILE=tests/config.aliyun.yaml pytest tests/integration/ -v

# Or use config.local.yaml for persistent overrides
cp tests/config.yaml tests/config.local.yaml
# Edit storage_backend to desired backend
pytest tests/integration/ -v
```

### Configuration Priority

1. `TEST_CONFIG_FILE` environment variable
2. `tests/config.local.yaml` (gitignored)
3. `tests/config.yaml` (default)

---

## Python FFI Interface Assumptions

This design assumes the Python FFI layer (`python/milvus_storage/`) is complete and provides full access to:

- **Writer API**: Create writer, write batches, flush, close
- **Reader API**: Create reader, scan, take by indices, chunk reader
- **Transaction API**: Begin, get column groups, commit with resolver, abort
- **Properties API**: Configure writer/reader behavior

### Key Properties Used in Tests

```python
PROPERTIES = {
    "writer.file_rolling.size": "1048576",      # 1MB file rolling
    "writer.buffer_size": "16777216",            # 16MB buffer
    "transaction.commit.num-retries": "3",       # Retry count
}
```

> **Note**: Conflict resolution strategy (fail/merge/overwrite) is set via `Transaction.Open()` resolver parameter, not via properties.

---

## Test Case Design

### Category 1: write_read

Core read/write functionality tests including file operations, compression, encryption, and data format handling.

#### 1.1 File Rolling Tests (`write_read/test_file_rolling.py`)

**Goal**: Verify file rolling behavior under different configurations

```python
class TestFileRolling:

    def test_file_rolling_by_size(self):
        """Roll by size"""
        # Set file_rolling_size = 1MB
        # Write 5MB data
        # Verify multiple files generated

    def test_file_rolling_small_threshold(self):
        """Small threshold rolling"""
        # file_rolling_size = 100KB
        # Verify frequent rolling

    def test_file_rolling_large_threshold(self):
        """Large threshold rolling"""
        # file_rolling_size = 100MB
        # Verify single file

    def test_file_rolling_exact_boundary(self):
        """Exact boundary rolling"""
        # Write exactly the amount of data to trigger rolling

    def test_file_rolling_with_compression(self):
        """File rolling with compression"""
        # Verify compressed size calculation

    def test_file_rolling_multiple_column_groups(self):
        """File rolling with multiple column groups"""
        # Different column groups roll independently

    def test_file_rolling_preserves_data(self):
        """Data integrity after rolling"""
        # Combined data from all files should be complete

    def test_file_rolling_memory_pressure(self):
        """Rolling under memory pressure"""
        # Set small buffer_size
        # Verify timely flushing

    def test_file_rolling_count_verification(self):
        """Rolling file count verification"""
        # Predict file count based on data volume and threshold
```

#### 1.2 Reader Advanced Tests (`write_read/test_reader_advanced.py`)

```python
class TestReaderAdvanced:

    def test_column_projection_performance(self):
        """Column projection performance"""
        # Read only subset of columns

    def test_predicate_pushdown(self):
        """Predicate pushdown"""
        # Scan with filter conditions

    def test_take_with_indices(self):
        """Take data by indices"""

    def test_take_with_large_index_list(self):
        """Take with large index list"""
        # Take 10000+ indices

    def test_take_across_multiple_files(self):
        """Take data spanning multiple files"""
        # Indices spread across file boundaries

    def test_parallel_scan(self):
        """Parallel scan"""

    def test_read_with_missing_columns(self):
        """Read schema with columns not in data"""
        # Should fill with nulls
```

#### 1.3 Chunk Reader Tests (`write_read/test_chunk_reader.py`)

```python
class TestChunkReader:

    def test_chunk_reader_random_access(self):
        """Chunk reader random access"""

    def test_chunk_reader_get_chunk_size(self):
        """Verify chunk size metadata"""

    def test_chunk_reader_get_chunk_rows(self):
        """Verify chunk row count metadata"""

    def test_chunk_reader_sequential_read(self):
        """Sequential chunk reading"""

    def test_chunk_reader_parallel_read(self):
        """Parallel chunk reading"""

    def test_chunk_indices_mapping(self):
        """Map row indices to chunk indices"""
```

#### 1.4 Compression Tests (`write_read/test_compression.py`)

```python
class TestCompression:

    def test_compression_snappy(self):
        """Snappy compression codec"""

    def test_compression_gzip(self):
        """Gzip compression codec"""

    def test_compression_zstd(self):
        """Zstd compression codec"""

    def test_compression_lz4(self):
        """LZ4 compression codec"""

    def test_compression_level(self):
        """Different compression levels"""

    def test_dictionary_encoding(self):
        """Dictionary encoding for strings"""

    def test_no_compression(self):
        """Uncompressed data"""

    def test_compression_ratio_verification(self):
        """Verify compression reduces file size"""
```

#### 1.5 Encryption Tests (`write_read/test_encryption.py`)

**Goal**: Verify encryption/decryption functionality

```python
class TestEncryption:

    def test_write_read_with_encryption(self):
        """Basic encrypted write and read"""
        # Enable encryption, write, read back

    def test_encryption_aes_gcm_v1(self):
        """AES-GCM-V1 encryption algorithm"""

    def test_encryption_aes_gcm_ctr_v1(self):
        """AES-GCM-CTR-V1 encryption algorithm"""

    def test_encryption_key_16_bytes(self):
        """16-byte encryption key"""

    def test_encryption_key_24_bytes(self):
        """24-byte encryption key"""

    def test_encryption_key_32_bytes(self):
        """32-byte encryption key"""

    def test_read_encrypted_without_key_fails(self):
        """Reading encrypted data without key should fail"""

    def test_read_encrypted_wrong_key_fails(self):
        """Reading encrypted data with wrong key should fail"""

    def test_key_retriever_callback(self):
        """Key retriever callback mechanism"""
        # Verify callback is called with correct key_id

    def test_encryption_metadata_persistence(self):
        """Encryption metadata persisted in column group"""

    def test_encryption_with_multiple_column_groups(self):
        """Encryption with multiple column groups"""
        # All groups should be encrypted
```

#### 1.6 Boundary Conditions Tests (`write_read/test_boundary_conditions.py`)

**Goal**: Test edge cases and boundary conditions

```python
class TestBoundaryConditions:

    def test_single_row_write_read(self):
        """Single row write and read"""

    def test_single_column_schema(self):
        """Schema with single column"""

    def test_very_wide_table(self):
        """Very wide table (500+ columns)"""

    def test_deeply_nested_types(self):
        """Deeply nested list/struct types"""
        # list<list<list<int>>>

    def test_special_characters_in_column_names(self):
        """Special characters in column names"""
        # Unicode, spaces, etc.

    def test_unicode_string_values(self):
        """Unicode string values"""
        # CJK, emoji, RTL text

    def test_max_int64_values(self):
        """Maximum int64 values"""

    def test_nan_and_inf_float_values(self):
        """NaN and Inf float values"""

    def test_empty_string_values(self):
        """Empty string values"""

    def test_null_values_all_columns(self):
        """All null values in a batch"""

    def test_very_long_file_paths(self):
        """Very long file paths"""

    def test_rapid_open_close_cycles(self):
        """Rapid writer/reader open/close cycles"""
        # Test resource cleanup
```

---

### Category 2: transaction

Transaction workflow tests.

#### 2.1 Append Tests (`transaction/test_append.py`)

**Goal**: Verify data append workflow via Transaction

```python
class TestAppend:

    def test_single_append(self):
        """Single append operation"""
        # 1. Write initial data
        # 2. Append new data via Transaction
        # 3. Verify all data readable

    def test_multiple_sequential_appends(self):
        """Multiple sequential appends"""
        # Loop 10 times append
        # Verify manifest version increments

    def test_append_with_different_batch_sizes(self):
        """Append with different batch sizes"""
        # Test batches of 100, 1000, 10000 rows

    def test_append_preserves_existing_data(self):
        """Append does not affect existing data"""

    def test_append_with_file_rolling(self):
        """Append triggers file rolling"""
```

#### 2.2 Add Field Tests (`transaction/test_add_field.py`)

**Goal**: Verify add field (schema evolution) via Transaction

```python
class TestAddField:

    def test_add_single_field(self):
        """Add a single new field"""
        # 1. Initial schema: {id, name}
        # 2. Add field via Transaction: {value}
        # 3. Verify new data can include value field

    def test_add_multiple_fields(self):
        """Add multiple fields"""

    def test_add_field_different_types(self):
        """Add fields of different data types"""
        # int32, int64, float32, float64, string, bool, binary

    def test_read_after_add_field(self):
        """Read old data after adding field"""
        # New field in old data should be null
```

#### 2.3 Mix Workflow Tests (`transaction/test_mix_workflow.py`)

**Goal**: Verify mixed workflow (append + add_field in same transaction)

```python
class TestMixWorkflow:

    def test_append_then_add_field(self):
        """Append data then add field in sequence"""

    def test_add_field_then_append(self):
        """Add field then append data in sequence"""

    def test_multiple_mixed_operations(self):
        """Multiple mixed operations in sequence"""
        # append -> add_field -> append -> add_field

    def test_mixed_workflow_data_integrity(self):
        """Data integrity after mixed workflow"""
```

#### 2.4 Concurrent Mix Workflow Tests (`transaction/test_concurrent_mix_workflow.py`)

**Goal**: Verify concurrent mixed workflows

```python
class TestConcurrentMixWorkflow:

    def test_concurrent_append(self):
        """Concurrent append operations"""
        # Multiple threads appending simultaneously

    def test_concurrent_add_field(self):
        """Concurrent add field operations"""

    def test_concurrent_mixed_operations(self):
        """Concurrent mixed append and add_field"""

    def test_conflict_resolution_merge(self):
        """Conflict resolution with MergeResolver"""

    def test_conflict_resolution_fail(self):
        """Conflict resolution with FailResolver"""

    def test_transaction_retry_on_conflict(self):
        """Transaction retry mechanism on conflict"""
```

---

### Category 3: manifest

Manifest operation tests.

#### 3.1 Version Upgrade Tests (`manifest/test_version_upgrade.py`)

```python
class TestVersionUpgrade:

    def test_read_older_version_data(self):
        """Read data written by older version"""

    def test_write_backward_compatible(self):
        """New version writes backward compatible data"""

    def test_manifest_version_migration(self):
        """Manifest version migration"""

    def test_property_compatibility(self):
        """Property compatibility across versions"""
```

---

### Category 4: schema_evolution

Schema evolution and column grouping tests.

#### 4.1 Schema Evolution Tests (`schema_evolution/test_schema_evolution.py`)

**Goal**: Verify dynamic field addition functionality

```python
class TestSchemaEvolution:

    def test_add_single_field(self):
        """Add a single new field"""
        # 1. Initial schema: {id, name}
        # 2. Add field: {value}
        # 3. Verify new data can include value field

    def test_add_multiple_fields(self):
        """Add multiple fields at once"""
        # Add {field1, field2, field3}

    def test_add_field_different_types(self):
        """Add fields of different data types"""
        # int32, int64, float32, float64, string, bool, binary, list, struct

    def test_add_field_duplicate_name_fails(self):
        """Adding duplicate field name should fail"""
        # Verify error handling

    def test_read_after_add_field(self):
        """Read old data after adding field"""
        # New field in old data should be null or default

    def test_multiple_schema_versions(self):
        """Multiple schema evolutions"""
        # v1: {id}
        # v2: {id, name}
        # v3: {id, name, value}
        # Verify all version data is readable

    def test_add_field_with_default_value(self):
        """Add field with default value"""
        # If supported

    def test_schema_evolution_concurrent(self):
        """Concurrent schema evolution"""
        # Multiple threads adding different fields simultaneously
```

#### 4.2 Column Group Policy Tests (`schema_evolution/test_column_group_policy.py`)

**Goal**: Verify different column grouping strategies

```python
class TestColumnGroupPolicy:

    def test_single_column_group_policy(self):
        """Single column group - all columns together"""
        # All columns in one file

    def test_schema_based_policy_basic(self):
        """Schema-based policy - pattern matching"""
        # Pattern: "id|name, value, vector"
        # Should create 3 column groups

    def test_schema_based_policy_complex_patterns(self):
        """Schema-based policy with complex patterns"""
        # Test various pattern combinations

    def test_size_based_policy_basic(self):
        """Size-based policy - auto grouping by size"""
        # max_avg_column_size, max_columns_in_group

    def test_size_based_policy_large_columns(self):
        """Size-based policy with large columns"""
        # Vector columns should be isolated

    def test_policy_with_nullable_columns(self):
        """Policy with nullable columns"""

    def test_policy_preserves_row_alignment(self):
        """Verify row alignment across column groups"""
        # All column groups should have same row count

    def test_mixed_column_types_grouping(self):
        """Grouping with mixed column types"""
        # int, string, binary, list, struct
```

---

### Category 5: external_table

External table import tests. Import existing Parquet/Vortex/Lance files into milvus-storage.

#### 5.1 Parquet Import Tests (`external_table/test_parquet_import.py`)

**Goal**: Import existing Parquet files into milvus-storage via Transaction

```python
class TestParquetImport:

    def test_explore_single_parquet_file(self):
        """Explore a single Parquet file"""
        # loon_exttable_explore -> get file list

    def test_get_parquet_file_info(self):
        """Get Parquet file row count and schema"""
        # loon_exttable_get_file_info -> row count

    def test_import_single_parquet_file(self):
        """Import single Parquet file to manifest"""
        # 1. loon_exttable_explore
        # 2. loon_exttable_get_file_info
        # 3. Build ColumnGroup struct
        # 4. Transaction.add_column_group
        # 5. Transaction.commit

    def test_import_multiple_parquet_files(self):
        """Import multiple Parquet files"""
        # Multiple files -> single ColumnGroup with file list

    def test_import_parquet_with_partitions(self):
        """Import partitioned Parquet dataset"""
        # Directory with partition structure

    def test_read_after_parquet_import(self):
        """Read data after Parquet import"""
        # Verify data accessible via Reader

    def test_append_after_parquet_import(self):
        """Append new data after importing Parquet"""
        # Import -> append -> verify all data
```

#### 5.2 Vortex Import Tests (`external_table/test_vortex_import.py`)

**Goal**: Import existing Vortex files into milvus-storage

```python
class TestVortexImport:

    def test_explore_single_vortex_file(self):
        """Explore a single Vortex file"""

    def test_get_vortex_file_info(self):
        """Get Vortex file row count and schema"""

    def test_import_single_vortex_file(self):
        """Import single Vortex file to manifest"""
        # Same flow as Parquet

    def test_import_multiple_vortex_files(self):
        """Import multiple Vortex files"""

    def test_read_after_vortex_import(self):
        """Read data after Vortex import"""

    def test_mixed_format_import(self):
        """Import both Parquet and Vortex files"""
        # Different column groups with different formats
```

#### 5.3 Lance Import Tests (`external_table/test_lance_import.py`)

**Goal**: Import existing Lance dataset (with fragments) into milvus-storage

```python
class TestLanceImport:

    def test_explore_lance_dataset(self):
        """Explore Lance dataset directory"""
        # loon_exttable_explore -> list of fragment files

    def test_explore_lance_fragments(self):
        """Get fragment IDs from Lance dataset"""
        # Same path, different fragment IDs

    def test_import_lance_single_fragment(self):
        """Import single Lance fragment"""

    def test_import_lance_all_fragments(self):
        """Import all fragments from Lance dataset"""
        # 1. Explore -> get all fragment files
        # 2. Build ColumnGroup per fragment (or merged)
        # 3. Transaction.add_column_group
        # 4. Commit

    def test_read_after_lance_import(self):
        """Read data after Lance import"""

    def test_lance_fragment_metadata(self):
        """Verify Lance fragment metadata preserved"""
```

---

### Category 6: recovery

Fault tolerance and recovery tests.

#### 6.1 Crash Recovery Tests (`recovery/test_crash_recovery.py`)

```python
class TestCrashRecovery:

    def test_recovery_after_writer_crash(self):
        """Recovery after writer process crash"""
        # Simulate crash during write
        # Verify data integrity after recovery

    def test_recovery_partial_write(self):
        """Recovery from partial write"""
        # Incomplete file should be handled

    def test_recovery_corrupted_file(self):
        """Handle corrupted data file"""
        # Skip or report corrupted files

    def test_manifest_recovery(self):
        """Manifest recovery after crash"""
        # Rollback to last valid manifest

    def test_transaction_rollback(self):
        """Transaction rollback on failure"""
```

#### 6.2 Data Validation Tests (`recovery/test_data_validation.py`)

```python
class TestDataValidation:

    def test_checksum_verification(self):
        """Data checksum verification"""

    def test_schema_validation(self):
        """Schema validation on read"""

    def test_row_count_verification(self):
        """Row count consistency check"""

    def test_column_group_alignment(self):
        """Column group row alignment verification"""

    def test_manifest_integrity_check(self):
        """Manifest integrity validation"""
```

---

### Cross-Category: Error Handling Tests (`test_error_handling.py`)

```python
class TestErrorHandling:

    def test_invalid_path(self):
        """Invalid path"""

    def test_corrupted_manifest(self):
        """Corrupted manifest"""

    def test_missing_data_file(self):
        """Missing data file"""

    def test_schema_mismatch(self):
        """Schema mismatch"""

    def test_permission_denied(self):
        """Permission denied"""

    def test_disk_full_simulation(self):
        """Disk full simulation"""

    def test_invalid_column_projection(self):
        """Invalid column names in projection"""

    def test_take_out_of_range(self):
        """Take with out-of-range indices"""

    def test_empty_write(self):
        """Write empty data"""

    def test_duplicate_column_names(self):
        """Duplicate column names in schema"""

    def test_invalid_properties(self):
        """Invalid property values"""
```

---

## Stress Test Design (`stress/`)

> **Scale Configuration**: Medium scale (10GB data volume / 100 million rows), max 100 concurrent threads

### Large Scale Write Tests (`test_large_scale_write.py`)

```python
@pytest.mark.stress
class TestLargeScaleWrite:

    @pytest.mark.parametrize("num_rows", [1_000_000, 10_000_000, 100_000_000])
    def test_large_row_count(self, num_rows):
        """Large row count write (max 100 million rows)"""
        # Write large number of rows, verify performance and correctness

    @pytest.mark.parametrize("num_columns", [10, 50, 100, 500])
    def test_wide_table(self, num_columns):
        """Wide table write"""
        # Multi-column scenarios

    def test_large_string_values(self):
        """Large string values"""
        # Each string 1KB-1MB

    def test_large_binary_values(self):
        """Large binary values (simulate vectors)"""
        # 1536-dim float32 vectors (6KB/row)
        # ~1.7 million rows = ~10GB vector data

    def test_continuous_write_10gb(self):
        """Continuous write 10GB data"""
        # Batch write, verify data integrity

    def test_memory_usage_stability(self):
        """Memory usage stability"""
        # Monitor memory while writing 10GB data
        # Use tracemalloc to detect leaks

    def test_throughput_benchmark(self):
        """Throughput benchmark"""
        # Record MB/s and rows/s
```

### High Concurrency Tests (`test_high_concurrency.py`)

```python
@pytest.mark.stress
class TestHighConcurrency:

    @pytest.mark.parametrize("thread_count", [10, 25, 50, 75, 100])
    def test_concurrent_writers(self, thread_count):
        """High concurrency write (max 100 threads)"""

    @pytest.mark.parametrize("thread_count", [10, 25, 50, 75, 100])
    def test_concurrent_readers(self, thread_count):
        """High concurrency read"""

    def test_mixed_read_write_load(self):
        """Mixed read/write load"""
        # 50 readers + 50 writers = 100 threads

    def test_transaction_contention(self):
        """Transaction contention"""
        # 100 concurrent transactions

    def test_resource_cleanup_under_load(self):
        """Resource cleanup under load"""
        # Verify no resource leaks (file handles, memory)

    def test_concurrent_schema_evolution(self):
        """Concurrent schema evolution"""
        # Multiple threads adding different fields simultaneously
```

### Long Running Tests (`test_long_running.py`)

```python
@pytest.mark.stress
@pytest.mark.slow
class TestLongRunning:

    def test_continuous_operation_1hour(self):
        """Continuous operation for 1 hour"""
        # Loop read/write, monitor stability

    def test_manifest_growth_over_time(self):
        """Manifest growth over time"""
        # 10000 appends
        # Verify manifest size and performance

    def test_file_handle_leak_detection(self):
        """File handle leak detection"""
        # Use psutil to monitor

    def test_gradual_schema_evolution(self):
        """Gradual schema evolution"""
        # Add one field every 100 appends
        # Final 100 fields

    def test_stress_with_gc_pressure(self):
        """Stability under GC pressure"""
        # Frequent object creation/destruction
```

---

## pytest Configuration

### `tests/pytest.ini` (Project Root)

```ini
[pytest]
testpaths = .
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    stress: marks tests as stress tests (deselect with '-m "not stress"')
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
```

### `conftest.py` Common Fixtures

```python
import pytest
import pyarrow as pa
from milvus_storage import Filesystem, Writer, Reader, Properties
from .config import TestConfig, get_config

@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Session-scoped test configuration."""
    return get_config()

@pytest.fixture
def temp_case_path(test_config, request):
    """Temporary storage path for each test case.

    Uses Filesystem API for both local and cloud backends.
    Path is relative to SubtreeFilesystem root (root_path for local, bucket_name for cloud).
    """
    test_name = request.node.name
    base = test_config.base_path
    path = f"{base}/{test_name}" if base else test_name

    fs = Filesystem.get(properties=test_config.to_fs_properties())

    # Cleanup before test
    try:
        files = fs.list_dir(path, recursive=True)
        for f in files:
            if not f.is_dir:
                fs.delete_file(f.path)
    except Exception:
        pass

    # Print path info for debugging
    if test_config.is_local:
        root = test_config.root_path
        print(f"\n[temp_case_path] root_path: {root}")
    else:
        root = test_config.bucket_name
        print(f"\n[temp_case_path] bucket_name: {root}")
    print(f"[temp_case_path] path: {path}")
    print(f"[temp_case_path] full_path: {root}/{path}")

    yield path

    # Cleanup after test
    try:
        files = fs.list_dir(path, recursive=True)
        for f in files:
            if not f.is_dir:
                fs.delete_file(f.path)
    except Exception:
        pass

@pytest.fixture
def simple_schema():
    """Standard test schema"""
    return pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("value", pa.float64()),
    ])

@pytest.fixture
def batch_generator(simple_schema):
    """Factory fixture to generate batches with specified size and offset."""
    def _generate(num_rows: int, offset: int = 0):
        return pa.RecordBatch.from_pydict({
            "id": list(range(offset, offset + num_rows)),
            "name": [f"name_{i}" for i in range(offset, offset + num_rows)],
            "value": [float(i) * 0.1 for i in range(offset, offset + num_rows)],
        }, schema=simple_schema)
    return _generate
```

### Environment Setup

Since tests are located in the project root `tests/` directory but depend on the Python FFI package in `python/`, the following setup is required:

```bash
# 1. Build the C++ FFI library
cd cpp && make python-lib && cd ..

# 2. Install Python FFI package in development mode
cd python && pip install -e ".[dev]" && cd ..

# 3. Run tests from project root
pytest tests/integration/ -v
```

**Alternative: Using PYTHONPATH**

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"
pytest tests/integration/ -v
```

---

## Implementation Phases

### Phase 1: Test Framework Setup ✅ Completed

1. ✅ Create directory structure under `tests/`
2. ✅ Write `conftest.py` common fixtures
3. ✅ Create `tests/pytest.ini` for test configuration
4. ✅ Implement `config.py` configuration loader
5. ✅ Add sample test (`test_file_rolling.py`) for validation

### Phase 2: Fault Injection Integration

1. Integrate [libfiu](https://blitiri.com.ar/p/libfiu/) as fault injection framework
2. Add fault injection points in C++ layer
3. Expose fault injection FFI interface to Python
4. Implement pytest fixtures for fault injection

### Phase 3: Integration Tests

1. `write_read/` - Core read/write tests
2. `transaction/` - Transaction workflow tests
3. `manifest/` - Manifest operation tests
4. `schema_evolution/` - Schema evolution tests
5. `external_table/` - External table import tests
6. `recovery/` - Recovery and fault tolerance tests
7. `test_error_handling.py` - Cross-category error handling

### Phase 4: Stress Tests

1. `stress/test_large_scale_write.py`
2. `stress/test_high_concurrency.py`
3. `stress/test_long_running.py`

---

## CI Configuration (`.github/workflows/python-integration-test.yml`)

```yaml
name: Python Integration Tests

on:
  push:
    paths:
      - 'cpp/**'
      - 'python/**'
      - 'tests/**'
      - '.github/workflows/python-integration-test.yml'
  pull_request:
    paths:
      - 'cpp/**'
      - 'python/**'
      - 'tests/**'

jobs:
  build-cpp:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - name: Setup C++ dependencies
        run: |
          pip install conan==1.61.0
          conan profile new default --detect
      - name: Build C++ library with Python bindings
        working-directory: cpp
        run: make python-lib
      - name: Upload library artifact
        uses: actions/upload-artifact@v4
        with:
          name: python-ffi-lib
          path: cpp/build/Release/libmilvus-storage.so

  integration-test:
    needs: build-cpp
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Download library to python package
        uses: actions/download-artifact@v4
        with:
          name: python-ffi-lib
          path: python/milvus_storage/lib/
      - name: Install Python FFI package
        working-directory: python
        run: pip install -e ".[dev]"
      - name: Run integration tests
        run: pytest tests/integration/ -v --tb=short

  stress-test:
    needs: build-cpp
    runs-on: ubuntu-22.04
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Download library to python package
        uses: actions/download-artifact@v4
        with:
          name: python-ffi-lib
          path: python/milvus_storage/lib/
      - name: Install Python FFI package
        working-directory: python
        run: pip install -e ".[dev]"
      - name: Run stress tests (exclude slow)
        run: pytest tests/stress/ -v -m "stress and not slow" --tb=short
        timeout-minutes: 60
```

**CI Notes:**
- **integration-test**: Runs on every PR and push for quick functional validation
- **stress-test**: Runs only on main branch push to avoid long PR times
- **slow tests**: Excluded via `-m "not slow"`, can be triggered manually or on schedule
- **Test execution**: Tests run from project root directory, Python FFI package installed via `pip install -e`

---

## Verification Commands

All commands should be run from the **project root directory** (`milvus-storage/`).

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific category
pytest tests/integration/write_read/ -v
pytest tests/integration/transaction/ -v
pytest tests/integration/manifest/ -v
pytest tests/integration/schema_evolution/ -v
pytest tests/integration/external_table/ -v
pytest tests/integration/recovery/ -v

# Run stress tests (exclude slow tests)
pytest tests/stress/ -v -m "stress and not slow"

# Run all tests (including slow tests, ~1-2 hours)
pytest tests/ -v

# Quick validation excluding slow tests
pytest tests/ -v -m "not slow"

# Generate coverage report
pytest tests/ --cov=milvus_storage --cov-report=html

# Run specific test module only
pytest tests/integration/transaction/test_append.py -v
```

---

## Key File Paths

| File | Description |
|------|-------------|
| `python/milvus_storage/_ffi.py` | Python FFI bindings |
| `python/tests/test_write_read.py` | Existing unit tests (in python package) |
| `tests/integration/` | Integration tests (project root) |
| `tests/stress/` | Stress tests (project root) |
| `tests/conftest.py` | Shared pytest fixtures |
| `cpp/include/milvus-storage/writer.h` | C++ Writer API |
| `cpp/include/milvus-storage/reader.h` | C++ Reader API |
| `cpp/include/milvus-storage/transaction/transaction.h` | C++ Transaction API |
| `cpp/include/milvus-storage/manifest.h` | C++ Manifest structure |
| `cpp/include/milvus-storage/column_groups.h` | C++ ColumnGroups structure |
| `cpp/include/milvus-storage/properties.h` | C++ Properties definitions |
| `cpp/include/milvus-storage/common/config.h` | C++ Config constants |
