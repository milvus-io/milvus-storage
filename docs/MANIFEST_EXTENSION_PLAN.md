# Manifest Data Structure Extension Plan

## Overview

This document outlines the plan to extend the manifest data structure to include **delta logs** and **stats** in addition to the existing **column groups**. The manifest format also includes **MAGIC number** and **version** fields for format validation and version tracking.

- **MAGIC Number**: 0x4D494C56 ("MILV" in ASCII) - validates that the file is a valid Manifest format
- **Version**: Currently version 1 - tracks format version for future compatibility
- **Delta Logs**: A list of delta log entries, where each entry is a tuple of `(path, type, num_entries)`. The type can be one of: `PRIMARY_KEY` (0), `POSITIONAL` (1), or `EQUALITY` (2).
- **Stats**: File lists keyed with strings (i.e., `std::map<std::string, std::vector<std::string>>`). Typical stats keys include: `"pk.delete"`, `"bloomfilter"`, and `"bm25"`.

Delta logs and stats are isolated as separate directories/attributes in the manifest structure. MAGIC and version are encoded through Avro encoder to maintain Avro binary format compatibility.

## Current State

### Current Manifest Structure ✅ IMPLEMENTED

- **Manifest** is now a concrete class (not a type alias):
  ```cpp
  class Manifest : public Serializable {
    uint32_t version_;  // Format version (currently 1)
    std::shared_ptr<ColumnGroups> column_groups_;
    std::vector<DeltaLog> delta_logs_;
    std::map<std::string, std::vector<std::string>> stats_;
  };
  ```

- **ColumnGroups** contains:
  - `column_groups_`: `std::vector<std::shared_ptr<ColumnGroup>>`
  - Column-to-group mapping for fast lookups
  - ✅ `metadata_` preserved for backward compatibility

- **ColumnGroup** contains:
  - `columns`: `std::vector<std::string>`
  - `format`: `std::string`
  - `files`: `std::vector<ColumnGroupFile>`

- **DeltaLog** struct:
  - `path`: `std::string` - relative path to delta log file
  - `type`: `DeltaLogType` enum (PRIMARY_KEY=0, POSITIONAL=1, EQUALITY=2)
  - `num_entries`: `int64_t`

- **Serialization**: Uses Avro binary format with MAGIC number and version validation
  - MAGIC: 0x4D494C56 ("MILV" in ASCII) - validates file format
  - Version: Currently 1 - tracks format version
  - All fields (including MAGIC and version) encoded through Avro encoder for compatibility

### Current Usage ✅ UPDATED

- ✅ `Transaction` is now a concrete class (not a template)
- ✅ `Transaction::Open()` factory method replaces constructor + `begin()`
- ✅ Fluent builder API: `AddColumnGroup()`, `AppendFiles()`, `AddDeltaLog()`, `UpdateStat()`
- ✅ Resolver system for conflict resolution
- ✅ Serialized to Avro format files: `manifest-{version}.avro`
- ✅ Used in C++, Rust, Java, and Python bindings
- ✅ Accessed via FFI (C API) and JNI (Java)
- ⚠️ FFI functions for Manifest delta logs and stats are **NOT YET IMPLEMENTED**

## Proposed Changes

### 1. New Manifest Class Structure ✅ IMPLEMENTED

Create a new `Manifest` class that contains:
- **Column Groups** (existing): `std::shared_ptr<ColumnGroups>` - wrapped from existing ColumnGroups (with metadata preserved)
- **Delta Logs** (new): `std::vector<DeltaLog>` - list of delta log entries
- **Stats** (new): `std::map<std::string, std::vector<std::string>>` - file lists keyed by stat name/type

#### DeltaLog Structure ✅ IMPLEMENTED

Each delta log entry is a tuple containing:
- `path`: `std::string` - relative path to the delta log file (renamed from `relative_path`)
- `type`: `DeltaLogType` (enum) - type of delta log: `PRIMARY_KEY` (0), `POSITIONAL` (1), or `EQUALITY` (2)
- `num_entries`: `int64_t` - number of entries in the delta log

```cpp
enum class DeltaLogType {
  PRIMARY_KEY = 0,    // Primary key delete (default)
  POSITIONAL = 1,     // Positional delete
  EQUALITY = 2,       // Equality delete
};

struct DeltaLog {
  std::string path;        // Relative path to delta log file
  DeltaLogType type;       // Type of delta log
  int64_t num_entries;    // Number of entries
};
```

### 2. Design Decisions ✅ IMPLEMENTED

#### Create New Manifest Class
- ✅ Created a new `Manifest` class that wraps `ColumnGroups`
- ✅ Maintains backward compatibility by delegating column group operations
- ✅ Clear separation of concerns
- ✅ Easier to extend in the future
- ✅ Manifest is now a real class instead of type alias

### 3. Delta Log Types ✅ IMPLEMENTED

Delta log types are defined as an enum:
- `PRIMARY_KEY` (0) - Primary key delete (default)
- `POSITIONAL` (1) - Positional delete
- `EQUALITY` (2) - Equality delete

### 4. Stats Keys ✅ IMPLEMENTED

The following are typical stats keys that are used:
- `"pk.delete"` - Primary key deletion markers
- `"bloomfilter"` - Bloom filter files for fast membership testing
- `"bm25"` - BM25 ranking/index files

Other custom keys may be added as needed.

### 5. Implementation Plan

#### Phase 1: Core C++ Implementation ✅ COMPLETED

1. **ColumnGroups** - **KEPT UNCHANGED FOR BACKWARD COMPATIBILITY**
   - ✅ Metadata functionality preserved in ColumnGroups
   - ✅ No changes to ColumnGroups serialization/deserialization
   - ✅ Added `resolve_paths()` method for path resolution

2. **Create New Manifest Class** (`cpp/include/milvus-storage/transaction/manifest.h`) ✅ COMPLETED
   - ✅ Define `DeltaLogType` enum with values: `PRIMARY_KEY` (0), `POSITIONAL` (1), `EQUALITY` (2)
   - ✅ Define `DeltaLog` struct with `path`, `type`, and `num_entries`
   - ✅ Define `Manifest` class inheriting from `Serializable`
   - ✅ Include:
     - `std::shared_ptr<ColumnGroups> column_groups_` - wrapped ColumnGroups
     - `std::vector<DeltaLog> delta_logs_` - list of delta log entries
     - `std::map<std::string, std::vector<std::string>> stats_` - file lists keyed by stat name
   - ✅ Implement delegation methods for column groups (for backward compatibility):
     - All ColumnGroups methods delegated to `column_groups_` member
   - ✅ Add methods for delta logs management:
     - `arrow::Status add_delta_log(const DeltaLog& delta_log)`
     - `arrow::Status add_delta_log(const std::string& path, DeltaLogType type, int64_t num_entries)`
     - `arrow::Result<std::vector<DeltaLog>> get_delta_logs() const`
     - `arrow::Result<std::vector<DeltaLog>> get_delta_logs_by_type(DeltaLogType type) const`
     - `[[nodiscard]] inline size_t delta_logs_size() const` - Get number of delta log entries
   - ✅ Add methods for stats management:
     - `arrow::Status add_stat(const std::string& key, const std::vector<std::string>& files)`
     - `arrow::Result<std::vector<std::string>> get_stat(const std::string& key) const`
     - `arrow::Result<std::map<std::string, std::vector<std::string>>> get_all_stats() const`
     - `[[nodiscard]] inline size_t stats_size() const` - Get number of stats entries

3. **Implement Serialization** (`cpp/src/transaction/manifest.cpp`) ✅ COMPLETED
   - ✅ Implemented Avro serialization for Manifest class
   - ✅ Serialization order (all encoded through Avro encoder to maintain Avro compatibility):
     1. **MAGIC number** (int32: 0x4D494C56 = "MILV" in ASCII) - NEW: validates file format
     2. **Version** (int32: currently 1) - NEW: tracks format version
     3. Column groups (serialized inline, same format as ColumnGroups including metadata)
     4. Delta logs array (array of DeltaLog records) - NEW (optional, only if non-empty)
     5. Stats map (string -> array of strings) - NEW (optional, only if non-empty)
   - ✅ DeltaLog serialization format:
     - `path` (string) - Note: renamed from `relative_path` to `path`
     - `type` (int/enum: 0=PRIMARY_KEY, 1=POSITIONAL, 2=EQUALITY)
     - `num_entries` (long/int64)
   - ✅ Format validation:
     - MAGIC number validated first to detect invalid/empty streams before Avro decoding
     - Version checked to ensure compatibility (currently only version 1 supported)
     - Prevents segfaults from malformed data by validating format early
   - ✅ Backward compatibility:
     - Old manifests (without MAGIC/version) will fail validation gracefully with clear error message
     - New manifests always include MAGIC and version for format validation
     - Delta logs and stats are optional and default to empty if absent

4. **Transaction Implementation** (`cpp/src/transaction/transaction.cpp`) ✅ COMPLETED
   - ✅ `Transaction::Open()` factory method implementation
   - ✅ `GetManifest()` returns manifest at read version
   - ✅ `Commit()` uses resolver to merge changes with seen manifest
   - ✅ Fluent builder methods implementation
   - ✅ Path resolution for delta log files and stat files
   - ✅ Maintains backward compatibility with old manifest format
   - ✅ `read_manifest()` helper method for reading manifests at specific versions

#### Phase 2: Backward Compatibility ✅ COMPLETED

1. **Version Handling** ✅ COMPLETED
   - ✅ **MAGIC number and version field** added to Manifest serialization for format validation
     - MAGIC: 0x4D494C56 ("MILV" in ASCII) - validates file is a valid Manifest
     - Version: Currently 1 - tracks format version for future compatibility
     - Both encoded through Avro encoder to maintain Avro binary format compatibility
   - ✅ Format validation:
     - MAGIC number checked first to detect invalid/empty/malformed streams
     - Version validated to ensure supported format version
     - Prevents segfaults by catching invalid data before Avro decoding
   - ✅ Deserialization handles both old and new formats:
     - **Old format (without MAGIC/version)**: Will fail validation with clear error message
     - **New format**: ColumnGroups + MAGIC + version + optional delta_logs/stats
   - ✅ Old manifests deserialize ColumnGroups first, then attempt optional delta_logs/stats
   - ✅ If delta_logs/stats are absent, they default to empty
   - ✅ Serialization conditionally writes delta_logs/stats only if non-empty

2. **Migration Path** ✅ COMPLETED
   - ✅ Old manifests (ColumnGroups format) continue to work
   - ✅ When reading old format, ColumnGroups wrapped in Manifest automatically
   - ✅ When committing, new Manifest format used automatically
   - ✅ No explicit migration utility needed - handled transparently

#### Phase 3: Transaction Updates ✅ COMPLETED

**Important Note**: Delta logs and stats are NOT part of the Writer/Reader workflow. They are metadata that external systems add to manifests to track their own files. The Transaction system handles merging delta logs and stats when committing manifests.

1. **Transaction API Refactoring** ✅ COMPLETED
   - ✅ `Transaction` is now a concrete class (not a template)
   - ✅ `Transaction::Open()` static factory method replaces constructor + `begin()`
   - ✅ `GetManifest()` method returns the manifest at read version
   - ✅ Fluent-style builder methods: `AddColumnGroup()`, `AppendFiles()`, `AddDeltaLog()`, `UpdateStat()`
   - ✅ `set_changes_from_manifest()` removed - use fluent builder methods instead
   - ✅ `read_version()` getter removed
   - ✅ `Commit()` returns `arrow::Result<int64_t>` (committed version)
   - ✅ Resolver functions are static inline functions (`MergeResolver`, `FailResolver`)
   - ✅ `TransactionCommitResult` struct removed from FFI
   - ✅ Version system simplified: `LATEST` constant (-1) for fetching latest version

2. **Resolver System** ✅ COMPLETED
   - ✅ `Resolver` function type: `std::function<arrow::Result<std::shared_ptr<Manifest>>(read_manifest, seen_manifest, updates)>`
   - ✅ `MergeResolver`: Merges changes with `seen_manifest` (latest manifest)
   - ✅ `FailResolver`: Fails if concurrent changes detected
   - ✅ Resolver passed to `Transaction::Open()` with default `FailResolver`

3. **Updates Tracking** ✅ COMPLETED
   - ✅ `Updates` struct tracks: `added_column_groups`, `appended_files`, `added_delta_logs`, `added_stats`
   - ✅ Fluent builder methods populate `Updates` struct
   - ✅ Resolver uses `Updates` to merge changes with `seen_manifest`
   - ✅ Delta logs and stats are merged automatically during commit resolution

#### Phase 4: FFI/C API Updates ✅ COMPLETED

**Current State**: 
- ✅ Transaction FFI updated to use new API (`transaction_open`, `transaction_get_manifest`, `transaction_commit`)
- ✅ `transaction_commit` simplified: removed `resolve_id` parameter, returns `int64_t* out_committed_version`
- ✅ `TransactionCommitResult` struct removed
- ✅ `CManifest` structure already includes all fields (delta_log_paths, delta_log_num_entries, num_delta_logs, stat_keys, stat_files, stat_file_counts, num_stats)
- ✅ `exttable_read_manifest()` already exports to `CManifest*` with all fields populated
- ✅ `transaction_get_manifest()` updated to return `CManifest*` instead of `ManifestHandle`
- ✅ `transaction_get_read_version()` added to get the read version of a transaction
- ✅ Transaction builder methods implemented:
  - ✅ `transaction_add_column_group()` - Add a new column group to transaction
  - ✅ `transaction_append_files()` - Append files to existing column groups
  - ✅ `transaction_add_delta_log()` - Add delta log (PRIMARY_KEY type hardcoded)
  - ✅ `transaction_update_stat()` - Update stat entry

**Simplification**: 
- For simplicity, FFI only supports `DELTA_LOG_TYPE_PRIMARY_KEY` (hardcoded). DeltaLogType enum and DeltaLog struct are NOT exported.
- Users cannot modify manifests directly, so `manifest_add_XXX` functions are not needed.
- **No handles needed**: Use `CManifest*` and `CColumnGroup*` directly instead of `ManifestHandle` and `ColumnGroupHandle`.
- **No getters needed**: All fields are directly accessible from `CManifest` structure:
  - `cmanifest->delta_log_paths` - array of delta log paths (PRIMARY_KEY only)
  - `cmanifest->delta_log_num_entries` - array of entry counts
  - `cmanifest->num_delta_logs` - number of delta logs
  - `cmanifest->stat_keys` - array of stat keys
  - `cmanifest->stat_files` - array of arrays of file paths
  - `cmanifest->stat_file_counts` - array of file counts per stat
  - `cmanifest->num_stats` - number of stats entries
  - `cmanifest->column_groups` - embedded `CColumnGroups` structure

1. **FFI Header** (`cpp/include/milvus-storage/ffi_c.h`) ✅ COMPLETED
   - ✅ **No enum or struct exports** - DeltaLogType and DeltaLog are internal only
   - ✅ Updated `transaction_get_manifest()` to return `CManifest*` instead of `ManifestHandle`:
     - `FFIResult transaction_get_manifest(TransactionHandle handle, CManifest* out_manifest)`
     - Caller allocates `CManifest` structure, function populates it
     - Caller must call `cmanifest->release(cmanifest)` when done (if release callback is set)
   - ✅ `manifest_destroy()` function removed (replaced by `cmanifest->release()` callback)
   - ✅ Added `transaction_get_read_version()`:
     - `FFIResult transaction_get_read_version(TransactionHandle handle, int64_t* out_read_version)`
   - ✅ Added transaction builder methods:
     - `FFIResult transaction_add_column_group(TransactionHandle handle, const CColumnGroup* column_group)`
     - `FFIResult transaction_append_files(TransactionHandle handle, const CColumnGroups* column_groups)`
     - `FFIResult transaction_add_delta_log(TransactionHandle handle, const char* path, int64_t num_entries)`
       - Takes path and num_entries only
       - Type is hardcoded to PRIMARY_KEY internally
     - `FFIResult transaction_update_stat(TransactionHandle handle, const char* key, const char* const* files, size_t files_len)`

2. **FFI Implementation** (`cpp/src/ffi/manifest_c.cpp`) ✅ COMPLETED
   - ✅ Updated `transaction_get_manifest()`:
     - Changed signature to accept `CManifest*` instead of `ManifestHandle*`
     - Uses `ManifestExporter::Export()` to populate `CManifest` structure
     - Sets release callback on `CManifest` to clean up memory
     - Handles empty manifest case (when `read_version_ == 0`) by initializing empty `CManifest`
   - ✅ `manifest_destroy()` implementation removed (replaced by release callback)
   - ✅ Implemented `transaction_get_read_version()`:
     - Returns the read version from the transaction
   - ✅ Implemented `transaction_add_column_group()`:
     - Imports `CColumnGroup*` to C++ `ColumnGroup`
     - Calls `transaction->AddColumnGroup()` with the column group
   - ✅ Implemented `transaction_append_files()`:
     - Imports `CColumnGroups*` to C++ `ColumnGroups`
     - Calls `transaction->AppendFiles()` with the column groups
   - ✅ Implemented `transaction_add_delta_log()`:
     - Creates DeltaLog with hardcoded PRIMARY_KEY type
     - Calls `transaction->AddDeltaLog()` with the DeltaLog
   - ✅ Implemented `transaction_update_stat()`:
     - Calls `transaction->UpdateStat()` with key and files
   - ✅ Note: No `manifest_add_XXX` functions needed since users cannot modify manifests directly
   - ✅ Note: `CManifest` structure already has all fields directly accessible, so no getter functions needed

#### Phase 5: JNI Updates - **TODO**

**Current State**:
- ✅ Transaction JNI updated to use new API (`transaction_open`, `transaction_get_column_groups`, `transaction_commit`)
- ✅ `transaction_commit` simplified: removed `resolve_id` parameter
- ⚠️ JNI methods for Manifest delta logs and stats are **NOT YET IMPLEMENTED**

1. **JNI Header** (`cpp/include/milvus-storage/ffi_jni.h`) - **TODO**
   - **No Java enum or class exports** - DeltaLogType and DeltaLog are internal only
   - Add JNI methods for reading delta logs from Manifest (read-only):
     - `jobjectArray manifestGetDeltaLogs(jlong manifest_handle)`
       - Returns array of delta log file paths (strings) only
       - Only returns PRIMARY_KEY type delta logs (hardcoded internally)
       - Note: `manifestGetDeltaLogsByType` is NOT needed since we only support PRIMARY_KEY
   - Add JNI methods for reading stats from Manifest (read-only):
     - `jobjectArray manifestGetStat(jlong manifest_handle, jstring key)`
     - `jobject manifestGetAllStats(jlong manifest_handle)`
   - Add JNI methods for Transaction builder:
     - `jlong transactionAddDeltaLog(jlong transaction_handle, jstring path, jlong num_entries)`
       - Takes path and num_entries only
       - Type is hardcoded to PRIMARY_KEY internally
     - `jlong transactionAddStat(jlong transaction_handle, jstring key, jobjectArray files)`

2. **JNI Implementation** (`cpp/src/jni/manifest_jni.cpp`) - **TODO**
   - Implement `manifestGetDeltaLogs()`:
     - Filter delta logs to only PRIMARY_KEY type
     - Return array of path strings only
   - Implement `manifestGetStat()` and `manifestGetAllStats()` for reading stats
   - Implement `transactionAddDeltaLog()`:
     - Create DeltaLog with hardcoded PRIMARY_KEY type
     - Call FFI `transaction_add_delta_log()` with path and num_entries
   - Implement `transactionAddStat()`:
     - Call FFI `transaction_add_stat()` with key and files
   - Note: No `manifestAddXXX` methods needed since users cannot modify manifests directly

#### Phase 6: Language Bindings

**Important Note**: Delta logs and stats are NOT part of Writer/Reader APIs. They are managed via Transaction fluent builder methods and accessed via Manifest getters.

1. **Python** (`python/milvus_storage/`) - **TODO**
   - Update `transaction.py` to add `add_delta_log(path, num_entries)` fluent builder method
     - Type is hardcoded to PRIMARY_KEY internally
   - Update `transaction.py` to add `update_stat(key, files)` fluent builder method
   - Update `manifest.py` to add `get_delta_logs()` method (returns list of paths)
     - Only returns PRIMARY_KEY type delta logs
     - Note: `get_delta_logs_by_type()` is NOT needed since we only support PRIMARY_KEY
   - Update `manifest.py` to add `get_stat(key)` and `get_all_stats()` methods
   - Update `_ffi.py` to expose new FFI functions for Manifest delta logs and stats
   - **No DeltaLogType enum or DeltaLog class** - these are internal only
   - **Note**: Writer and Reader classes do NOT need delta log/stat methods

2. **Java/Scala** (`java/src/main/scala/`) - **TODO**
   - Update `MilvusStorageTransaction.scala` to add `addDeltaLog(path, numEntries)` fluent builder method
     - Type is hardcoded to PRIMARY_KEY internally
   - Update `MilvusStorageTransaction.scala` to add `updateStat(key, files)` fluent builder method
   - Update `MilvusStorageManifest.scala` to add `getDeltaLogs()` method (returns list of paths)
     - Only returns PRIMARY_KEY type delta logs
     - Note: `getDeltaLogsByType()` is NOT needed since we only support PRIMARY_KEY
   - Update `MilvusStorageManifest.scala` to add `getStat(key)` and `getAllStats()` methods
   - **No DeltaLogType enum or DeltaLog class** - these are internal only
   - **Note**: Writer and Reader classes do NOT need delta log/stat methods

3. **Rust** (`rust/src/`) - **TODO**
   - Update FFI bindings (`rust/src/ffi.rs`) for Manifest delta logs and stats
   - Update Transaction bindings to include fluent builder methods:
     - `add_delta_log(path, num_entries)` - type hardcoded to PRIMARY_KEY
     - `update_stat(key, files)`
   - Update Manifest bindings to include getter methods:
     - `get_delta_logs()` - returns Vec<String> (paths only)
     - `get_stat(key)` and `get_all_stats()`
   - **No DeltaLogType enum or DeltaLog struct** - these are internal only
   - Update table provider if needed (`rust/src/table_provider.rs`)

#### Phase 7: Testing

1. **Unit Tests**
   - Test serialization/deserialization with delta logs and stats
   - Test backward compatibility (reading old manifests)
   - Test forward compatibility (new manifests)
   - Test empty delta logs and stats cases
   - Test delta log operations (PRIMARY_KEY type)
   - Test typical stats keys: `"pk.delete"`, `"bloomfilter"`, `"bm25"`
   - Note: FFI only supports PRIMARY_KEY type, so filtering by type is not needed in FFI tests

2. **Integration Tests**
   - Test transaction fluent builder methods for delta logs and stats
   - Test manifest getters for delta logs and stats
   - Test transaction commit with delta logs and stats (via resolver)
   - Test FFI/JNI bindings for Manifest delta logs and stats
   - Test path resolution for delta log files and stat files
   - **Note**: Writer and Reader do NOT have delta log/stat methods - these are external metadata

3. **Migration Tests**
   - Test reading old manifest format
   - Test writing new manifest format
   - Test mixed version scenarios

## File Structure Changes

### New Files
- ✅ `cpp/include/milvus-storage/transaction/manifest.h` - Manifest class definition
- ✅ `cpp/src/transaction/manifest.cpp` - Implementation of Manifest class

### Modified Files
- ✅ `cpp/include/milvus-storage/column_groups.h` - Metadata preserved for backward compatibility
- ✅ `cpp/src/column_groups.cpp` - Metadata serialization/deserialization preserved
- ✅ `cpp/include/milvus-storage/transaction/manifest.h` - New Manifest class definition with DeltaLog struct and enum
- ✅ `cpp/src/transaction/manifest.cpp` - Implement Manifest serialization with delta logs and stats
- ✅ `cpp/include/milvus-storage/transaction/transaction.h` - Transaction refactored to concrete class with fluent builder API
- ✅ `cpp/src/transaction/transaction.cpp` - Transaction implementation with resolver system
- ✅ `cpp/include/milvus-storage/ffi_c.h` - Transaction FFI updated, `CManifest` structure includes all fields
- ✅ `cpp/src/ffi/manifest_c.cpp` - Transaction FFI updated, `exttable_read_manifest()` exports to `CManifest*`
- ✅ `cpp/include/milvus-storage/ffi_c.h` - **COMPLETED**: `transaction_get_manifest()` returns `CManifest*`, transaction builder methods added
- ✅ `cpp/src/ffi/manifest_c.cpp` - **COMPLETED**: `transaction_get_manifest()` implementation updated, all transaction builder methods implemented
- ⚠️ `cpp/include/milvus-storage/ffi_jni.h` - **TODO**: Add JNI methods for delta logs and stats
- ⚠️ `cpp/src/jni/manifest_jni.cpp` - **TODO**: Update to work with Manifest delta logs and stats
- ⚠️ `python/milvus_storage/` - **TODO**: Add Python bindings for delta logs and stats
- ⚠️ `java/src/main/scala/io/milvus/storage/*.scala` - **TODO**: Update Java bindings for delta logs and stats
- ⚠️ `rust/src/ffi.rs` - **TODO**: Update Rust bindings for delta logs and stats

**Note**: Writer and Reader APIs do NOT include delta logs and stats - these are external metadata managed via Transaction.

## Serialization Format

### Avro Schema Structure (Conceptual)

```
{
  "type": "record",
  "name": "Manifest",
  "fields": [
    {
      "name": "magic",
      "type": "int",
      "default": 0x4D494C56
    },
    {
      "name": "version",
      "type": "int",
      "default": 1
    },
    {
      "name": "column_groups",
      "type": "ColumnGroups"
    },
    {
      "name": "delta_logs",
      "type": {
        "type": "array",
        "items": {
          "type": "record",
          "name": "DeltaLog",
          "fields": [
            {
              "name": "path",
              "type": "string"
            },
            {
              "name": "type",
              "type": "int"
            },
            {
              "name": "num_entries",
              "type": "long"
            }
          ]
        }
      },
      "default": []
    },
    {
      "name": "stats",
      "type": {
        "type": "map",
        "values": {
          "type": "array",
          "items": "string"
        }
      },
      "default": {}
    }
  ]
}
```

### Serialization Order (Binary Avro) ✅ IMPLEMENTED
1. **MAGIC number** (int32: 0x4D494C56 = "MILV") - NEW: Format validation
2. **Version** (int32: currently 1) - NEW: Format version tracking
3. Column groups (serialized ColumnGroups format - **with metadata for backward compatibility**)
4. Delta logs array (array of DeltaLog records) - NEW (optional, only if non-empty)
5. Stats map (string -> array of strings) - NEW (optional, only if non-empty)

**Note**: All fields are encoded through Avro encoder to maintain Avro binary format compatibility. MAGIC and version are validated first to prevent segfaults from malformed data.

### ColumnGroups Serialization ✅ PRESERVED
ColumnGroups serialization format:
- Column groups array (existing format)
- ✅ **Metadata map preserved** for backward compatibility

### DeltaLog Serialization ✅ IMPLEMENTED
Each DeltaLog entry is serialized as:
- `path` (string) - Note: renamed from `relative_path` to `path`
- `type` (int: 0=PRIMARY_KEY, 1=POSITIONAL, 2=EQUALITY)
- `num_entries` (long/int64)

### Stats Keys ✅ IMPLEMENTED
Typical stats keys that are stored:
- `"pk.delete"` - Primary key deletion markers
- `"bloomfilter"` - Bloom filter files
- `"bm25"` - BM25 ranking/index files

## Backward Compatibility Strategy ✅ IMPLEMENTED

1. ✅ **Format Validation**: MAGIC number (0x4D494C56) and version field validate manifest format before Avro decoding
   - Prevents segfaults from malformed/empty streams by validating format early
   - Old manifests without MAGIC/version will fail with clear error message
2. ✅ **Version Handling**: Version field (currently 1) tracks format version for future compatibility
   - Unsupported versions are rejected with clear error message
3. ✅ **Default Values**: Delta logs and stats default to empty if absent during deserialization
4. ✅ **Metadata Handling**: Old manifests with metadata deserialize ColumnGroups including metadata (preserved for backward compatibility)
5. ✅ **Path Handling**: Delta log and stat file paths follow same relative/absolute path rules as column group files
6. ✅ **API Compatibility**: Existing column group APIs remain unchanged via delegation (metadata methods preserved in ColumnGroups)
7. ✅ **Manifest Class**: `Manifest` is now a real class wrapping `ColumnGroups` instead of type alias
8. ✅ **Isolation**: Delta logs and stats are isolated as separate directories/attributes
9. ✅ **ColumnGroups**: ColumnGroups metadata preserved for backward compatibility
10. ✅ **Avro Compatibility**: All fields (including MAGIC and version) encoded through Avro encoder to maintain standard Avro binary format

## Migration Strategy ✅ IMPLEMENTED

1. ✅ **Read Path**: 
   - New format manifests include MAGIC and version for validation
   - Old format manifests (without MAGIC/version) will fail validation with clear error message
   - Format validation prevents segfaults from malformed data
2. ✅ **Write Path**: 
   - Always write new format (Manifest with MAGIC + version + ColumnGroups + optional delta_logs/stats)
   - MAGIC and version are always included for format validation
   - Delta logs and stats are conditionally written only if non-empty
   - All fields encoded through Avro encoder to maintain Avro binary format compatibility

## Example Usage

### C++ API ✅ UPDATED
```cpp
// Open transaction
ARROW_ASSIGN_OR_RAISE(auto fs, CreateArrowFileSystem(fs_config));
ARROW_ASSIGN_OR_RAISE(auto transaction, Transaction::Open(fs, base_path, LATEST));

// Get current manifest
ARROW_ASSIGN_OR_RAISE(auto manifest, transaction->GetManifest());

// Add delta logs using fluent builder methods (external system tracks their own files)
transaction->AddDeltaLog({"delta/positional_001.log", DeltaLogType::POSITIONAL, 1000});
transaction->AddDeltaLog({"delta/equality_001.log", DeltaLogType::EQUALITY, 500});
transaction->AddDeltaLog({"delta/pk_001.log", DeltaLogType::PRIMARY_KEY, 200});

// Add stats using fluent builder methods (external system tracks their own files)
transaction->UpdateStat("pk.delete", {"delete_file_1.parquet", "delete_file_2.parquet"});
transaction->UpdateStat("bloomfilter", {"bloom_filter_1.bf"});
transaction->UpdateStat("bm25", {"bm25_index_1.idx"});

// Append files to existing column groups
transaction->AppendFiles(new_column_groups);

// Commit transaction (delta logs and stats are merged automatically via resolver)
ARROW_ASSIGN_OR_RAISE(auto committed_version, transaction->Commit());

// Read manifest to access delta logs and stats
ARROW_ASSIGN_OR_RAISE(auto current_manifest, transaction->GetManifest());
ARROW_ASSIGN_OR_RAISE(auto all_delta_logs, current_manifest->get_delta_logs());
// Note: FFI only supports PRIMARY_KEY type, so get_delta_logs_by_type() is not needed in FFI
ARROW_ASSIGN_OR_RAISE(auto delete_files, current_manifest->get_stat("pk.delete"));
ARROW_ASSIGN_OR_RAISE(auto all_stats, current_manifest->get_all_stats());
```

### Python API ✅ UPDATED
```python
# Open transaction
transaction = Transaction.open(fs, base_path, version=LATEST)

# Get current manifest
manifest = transaction.get_manifest()

# Add delta logs using fluent builder methods (external system tracks their own files)
# Note: FFI only supports PRIMARY_KEY type (hardcoded)
transaction.add_delta_log("delta/pk_001.log", 200)
transaction.add_delta_log("delta/pk_002.log", 150)

# Add stats using fluent builder methods (external system tracks their own files)
transaction.update_stat("pk.delete", ["delete_file_1.parquet", "delete_file_2.parquet"])
transaction.update_stat("bloomfilter", ["bloom_filter_1.bf"])
transaction.update_stat("bm25", ["bm25_index_1.idx"])

# Append files to existing column groups
transaction.append_files(new_column_groups)

# Commit transaction (delta logs and stats are merged automatically via resolver)
committed_version = transaction.commit()

# Read manifest to access delta logs and stats
current_manifest = transaction.get_manifest()
all_delta_logs = current_manifest.get_delta_logs()  # Returns list of paths (PRIMARY_KEY only)
# Note: FFI only supports PRIMARY_KEY type, so get_delta_logs_by_type() is not needed in FFI
delete_files = current_manifest.get_stat("pk.delete")
all_stats = current_manifest.get_all_stats()
```

### C API (FFI) ⚠️ TODO
```c
// Open transaction
TransactionHandle transaction;
transaction_open(base_path, properties, &transaction);

// Get manifest (returns CManifest* directly, no handles needed)
CManifest cmanifest;
transaction_get_manifest(transaction, &cmanifest);

// Access delta logs directly from CManifest structure (no getters needed)
// Note: Only PRIMARY_KEY type delta logs are included
for (uint32_t i = 0; i < cmanifest.num_delta_logs; i++) {
    const char* path = cmanifest.delta_log_paths[i];
    int64_t num_entries = cmanifest.delta_log_num_entries[i];
    // Use path and num_entries...
}

// Access stats directly from CManifest structure (no getters needed)
for (uint32_t i = 0; i < cmanifest.num_stats; i++) {
    const char* key = cmanifest.stat_keys[i];
    uint32_t file_count = cmanifest.stat_file_counts[i];
    for (uint32_t j = 0; j < file_count; j++) {
        const char* file = cmanifest.stat_files[i][j];
        // Use key and file...
    }
}

// Access column groups from embedded CColumnGroups structure
CColumnGroups* cgs = &cmanifest.column_groups;
for (uint32_t i = 0; i < cgs->num_of_column_groups; i++) {
    CColumnGroup* cg = &cgs->column_group_array[i];
    // Use column group...
}

// Add delta logs using transaction builder (PRIMARY_KEY type hardcoded)
transaction_add_delta_log(transaction, "delta/pk_001.log", 200);
transaction_add_delta_log(transaction, "delta/pk_002.log", 150);

// Add stats using transaction builder
const char* delete_files[] = {"delete_file_1.parquet", "delete_file_2.parquet"};
transaction_add_stat(transaction, "pk.delete", delete_files, 2);

// Commit transaction
int64_t committed_version;
transaction_commit(transaction, &committed_version);

// Clean up: release CManifest memory (if release callback is set)
if (cmanifest.release) {
    cmanifest.release(&cmanifest);
}

// Destroy transaction
transaction_destroy(transaction);
```

## Testing Checklist

### Phase 1-3 (Core Implementation) ✅ COMPLETED
- [x] ColumnGroups metadata preserved for backward compatibility
- [x] Old manifests with metadata deserialize correctly
- [x] Manifest serialization with delta logs and stats
- [x] Manifest deserialization with delta logs and stats
- [x] Backward compatibility: reading old manifest format (ColumnGroups without delta logs/stats, with or without metadata)
- [x] Forward compatibility: new manifest format (Manifest with delta logs and stats)
- [x] Empty delta logs and stats handling
- [x] ColumnGroups delegation methods work correctly
- [x] Path resolution for delta log files and stat files
- [x] Transaction fluent builder API (AddColumnGroup, AppendFiles, AddDeltaLog, UpdateStat)
- [x] Transaction resolver system (MergeResolver, FailResolver)
- [x] Transaction commit with delta logs and stats
- [x] All delta log types: PRIMARY_KEY, POSITIONAL, EQUALITY
- [x] Delta log filtering by type
- [x] Typical stats keys: "pk.delete", "bloomfilter", "bm25"
- [x] Multiple delta log entries with different types
- [x] Multiple stats entries with different keys

### Phase 4 (FFI/C API) ✅ COMPLETED
- [x] Update `transaction_get_manifest()` to return `CManifest*` instead of `ManifestHandle`
- [x] Remove `manifest_destroy()` (replaced by `CManifest->release()` callback)
- [x] Add `transaction_get_read_version()` to get transaction read version
- [x] FFI transaction builder methods: `transaction_add_column_group(column_group)`
- [x] FFI transaction builder methods: `transaction_append_files(column_groups)`
- [x] FFI transaction builder methods: `transaction_add_delta_log(path, num_entries)` (PRIMARY_KEY hardcoded)
- [x] FFI transaction builder methods: `transaction_update_stat(key, files)`
- [x] Note: No getter functions needed - `CManifest` fields are directly accessible

### Phase 5-7 (JNI and Language Bindings) ⚠️ TODO
- [ ] JNI Java API functions for reading Manifest delta logs (path-only, PRIMARY_KEY only)
- [ ] JNI Java API functions for reading Manifest stats
- [ ] JNI transaction builder methods: `transactionAddDeltaLog(path, numEntries)` (PRIMARY_KEY hardcoded)
- [ ] JNI transaction builder methods: `transactionAddStat(key, files)`
- [ ] Python bindings for Transaction and Manifest delta logs/stats (PRIMARY_KEY only, path-only)
- [ ] Java/Scala bindings for Transaction and Manifest delta logs/stats (PRIMARY_KEY only, path-only)
- [ ] Rust bindings for Transaction and Manifest delta logs/stats (PRIMARY_KEY only, path-only)
- [ ] Integration tests for FFI/JNI
- [ ] Performance tests (serialization overhead)

## Risks and Mitigations

1. **Risk**: Breaking backward compatibility
   - **Mitigation**: ✅ Careful version handling and default values - IMPLEMENTED

2. **Risk**: Serialization format changes breaking existing readers
   - **Mitigation**: ✅ Version field and backward-compatible deserialization - IMPLEMENTED
   - **Mitigation**: ✅ MAGIC number validation prevents segfaults from malformed data - IMPLEMENTED

3. **Risk**: Performance impact of additional serialization data
   - **Mitigation**: Benchmark and optimize if needed

4. **Risk**: Inconsistent behavior across language bindings
   - **Mitigation**: Comprehensive testing across all bindings

## Timeline Estimate

- ✅ Phase 1 (Core C++): **COMPLETED** - Manifest class, serialization, backward compatibility
- ✅ Phase 2 (Backward Compatibility): **COMPLETED** - Version handling, migration path
- ✅ Phase 3 (Transaction Updates): **COMPLETED** - Transaction refactoring, resolver system, fluent builder API
- ✅ Phase 4 (FFI/C API): **COMPLETED** - Transaction FFI updated, all builder methods implemented
- ⏳ Phase 5 (JNI): **PENDING** - JNI methods for delta logs and stats
- ⏳ Phase 6 (Language Bindings): **PENDING** - Python, Java, Rust bindings
- ⏳ Phase 7 (Testing): **PENDING** - Comprehensive testing

**Remaining Work**:
- Phase 5: JNI updates (~1 day)
- Phase 6: Language bindings (~2-3 days)
- Phase 7: Testing (~2-3 days)

**Total Remaining Estimate**: 4-6 days

**Note**: ColumnGroups metadata was preserved (not removed) for backward compatibility, which simplified the implementation.

## Open Questions

1. ✅ **RESOLVED**: Delta logs support (path, type, num_entries) - sufficient for current needs
2. Should there be validation for delta log and stat file paths?
3. Should delta logs or stats support the same private_data field as ColumnGroupFile?
4. Should there be limits on the number of delta log entries or stats entries?
5. ✅ **RESOLVED**: Typical stats keys include "pk.delete", "bloomfilter", "bm25" - others can be added as needed
6. Should delta logs be ordered (e.g., by timestamp or sequence number)?
7. ✅ **RESOLVED**: Delta log entries are appended during commit resolution - no consolidation needed currently
8. ✅ **RESOLVED**: ColumnGroups metadata preserved for backward compatibility - no migration needed
9. ✅ **RESOLVED**: Format validation implemented with MAGIC number (0x4D494C56) and version field to prevent segfaults from malformed data

## Recent Bug Fixes ✅ COMPLETED

1. ✅ **Format Validation**: Added MAGIC number (0x4D494C56) and version field to prevent segfaults from malformed/empty streams
   - MAGIC and version are validated before Avro decoding
   - Invalid manifests fail gracefully with clear error messages
   - All fields encoded through Avro encoder to maintain Avro binary format compatibility

2. ✅ **Bounds Checking**: Added proper bounds checking for column group index access
   - `Manifest::getColumnGroup(size_t index)` now checks bounds and returns nullptr if out of range
   - `Reader::get_chunk_reader()` validates index before accessing column groups
   - Prevents undefined behavior from out-of-bounds access

3. ✅ **Column Ordering**: Fixed `Reader::take()` to merge columns in schema order
   - Columns are now reordered to match `needed_columns_` order instead of column group order
   - Ensures output table columns match expected schema order
   - Handles missing columns by filling with null arrays

## Next Steps

1. ✅ **COMPLETED**: Phases 1-3 implementation (Core C++, Backward Compatibility, Transaction Updates)
2. ✅ **COMPLETED**: Format validation with MAGIC and version fields
3. ✅ **COMPLETED**: Bug fixes for bounds checking and column ordering
4. ✅ **COMPLETED**: Phase 4 - FFI functions for Manifest delta logs and stats, all transaction builder methods
5. ⏳ **PENDING**: Phase 5 - JNI updates for delta logs and stats
6. ⏳ **PENDING**: Phase 6 - Language bindings (Python, Java, Rust)
7. ⏳ **PENDING**: Phase 7 - Comprehensive testing

## Implementation Status Summary

- ✅ **Phase 1**: Core C++ Implementation - COMPLETED
  - Manifest class with delta logs and stats
  - Serialization/deserialization with backward compatibility
  - Path resolution for all file types
  - MAGIC number (0x4D494C56) and version field for format validation

- ✅ **Phase 2**: Backward Compatibility - COMPLETED
  - Version handling with format validation
  - MAGIC number validation prevents segfaults from malformed data
  - Old manifest format detection and error handling

- ✅ **Phase 3**: Transaction Updates - COMPLETED
  - Transaction refactored to concrete class
  - Fluent builder API (AddColumnGroup, AppendFiles, AddDeltaLog, UpdateStat)
  - Resolver system (MergeResolver, FailResolver)
  - Updates tracking system

- ✅ **Bug Fixes**: Format Validation and Safety - COMPLETED
  - MAGIC number and version field added for format validation
  - Bounds checking for column group index access
  - Column ordering fix in `Reader::take()` to match schema order

- ✅ **Phase 4**: FFI/C API Updates - COMPLETED
  - ✅ Transaction FFI updated
  - ✅ `CManifest` structure includes all fields (delta logs and stats)
  - ✅ `exttable_read_manifest()` already exports to `CManifest*`
  - ✅ `transaction_get_manifest()` updated to return `CManifest*` instead of `ManifestHandle`
  - ✅ `transaction_get_read_version()` added
  - ✅ Transaction builder methods implemented:
    - ✅ `transaction_add_column_group()` - Add new column group
    - ✅ `transaction_append_files()` - Append files to existing column groups
    - ✅ `transaction_add_delta_log()` - Add delta log (PRIMARY_KEY hardcoded)
    - ✅ `transaction_update_stat()` - Update stat entry

- ⏳ **Phase 5**: JNI Updates - PENDING

- ⏳ **Phase 6**: Language Bindings - PENDING

- ⏳ **Phase 7**: Testing - PENDING
