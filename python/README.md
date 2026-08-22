# milvus-storage Python Bindings

Python bindings for [milvus-storage](https://github.com/milvus-io/milvus-storage), a high-performance storage engine using Apache Arrow Parquet as the underlying format, optimized for analytical workloads.

## Features

- **High Performance**: Built on Apache Arrow and Parquet for efficient columnar storage
- **Packed Storage**: Groups narrow columns together to reduce file count and control memory usage
- **Cloud Native**: Support for AWS S3, Azure Blob Storage, Google Cloud Storage, and more
- **Zero-Copy**: Efficient data transfer between Python and C++ using Arrow C Data Interface
- **Pythonic API**: Clean, intuitive interface following Python best practices
- **Type Safe**: Full PyArrow integration with schema validation

## Installation

### Prerequisites

- Python 3.8 or later
- C++ compiler (for building from source)
- Conan (for C++ dependencies)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/milvus-io/milvus-storage.git
cd milvus-storage

# Build the C++ library
cd cpp
make python-lib
cd ..

# Install Python package
cd python
pip install -e .
```

## API Reference

### Writer

Create a writer to store data in milvus-storage format.

```python
Writer(path: str, schema: pa.Schema, properties: Optional[Dict[str, str]] = None)
```

**Methods:**
- `write(batch: pa.RecordBatch)` - Write a record batch
- `flush()` - Flush buffered data to storage
- `close() -> str` - Close writer and return manifest JSON

### Reader

Read data from milvus-storage datasets.

```python
Reader(
    manifest: str,
    schema: pa.Schema,
    columns: Optional[List[str]] = None,
    properties: Optional[Dict[str, str]] = None
)
```

**Methods:**
- `scan(predicate: Optional[str] = None) -> pa.RecordBatchReader` - Full table scan
- `take(indices: Union[List[int], np.ndarray], parallelism: int = 1) -> pa.RecordBatch` - Random access *(not yet implemented)*
- `get_chunk_reader(column_group_id: int) -> ChunkReader` - Get chunk reader for column group

### Properties

Configuration properties for milvus-storage.

**Common Properties:**

| Property | Description | Default |
|----------|-------------|---------|
| `fs.storage_type` | Storage type (local, s3, azure, etc.) | - |
| `fs.root_path` | Root path for local storage | - |
| `storage.memory.limit` | Memory limit in bytes | - |
| `storage.row_group.max_size` | Max row group size | - |
| `storage.batch.size` | Batch size for reading | 8192 |
| `storage.s3.access_key_id` | AWS access key | - |
| `storage.s3.secret_access_key` | AWS secret key | - |
| `storage.s3.region` | AWS region | - |
| `storage.azure.account_name` | Azure account name | - |
| `storage.azure.account_key` | Azure account key | - |

### Error handling

Every failure that crosses the native boundary carries its verdict as
attributes, and the exception type is chosen from the category — so you branch
on a type or a field, never on the message text.

```python
from milvus_storage.exceptions import (
    MilvusStorageError,   # base: catches everything this library raises
    FFIError,             # any native failure that is not your input
    RetryableError,       # a transient cause was identified
    ConflictError,        # concurrent modification; coordinate, do not replay
    DataFormatError,      # persisted bytes do not decode
    InvalidArgumentError, # your input was invalid
)

try:
    writer.write(batch)
except RetryableError as e:
    ...            # e.retryable is True; see the writer caveat below
except InvalidArgumentError:
    raise          # the caller's problem — hand it back
except FFIError as e:
    log.error("storage failed: code=%s category=%s", e.err_code, e.category)
```

Attributes on every exception:

| Attribute | Meaning |
|---|---|
| `err_code` | the fine-grained native code, or `None` if the failure never crossed the C ABI |
| `category` | an `ErrorCategory` — `USER`, `RETRYABLE`, `CONFLICT`, `DATA_FORMAT`, `SYSTEM`, `UNKNOWN` |
| `retryable` | `category == RETRYABLE` |

`InvalidArgumentError` is a **sibling** of `FFIError`, not a subclass: the same
exception is raised by this package's own argument checks and by the native
`USER` category, and you should not have to care which side caught your mistake.
To catch everything, catch `MilvusStorageError`.

**`retryable` is about the cause, not about the object.** A failed `Writer` is
terminal — every later call returns the same failure. Retrying means a new
`Writer` on a new path. Call `close()` on the failed one anyway (it re-raises,
that is expected): closing is what releases what it still holds in the store.
Better, use it as a context manager and let `__exit__` do it.

A code from a newer library that this binding has never seen degrades to
`ErrorCategory.UNKNOWN` and `FFIError` rather than raising during error
handling, so an older binding keeps working against a newer library.

See [`docs/error-codes.md`](../docs/error-codes.md) for the full code table.

## Testing

Run tests with pytest:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=milvus_storage --cov-report=html
```

## Development

### Building from Source
```bash
# Build C++ library
cd cpp
make python-lib

# Install in development mode
cd ../python
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Requirements

- Python >= 3.8
- pyarrow >= 10.0.0
- numpy >= 1.20.0

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Contributing

Contributions are welcome! Please see the main repository for contribution guidelines.

## Support

- GitHub Issues: [milvus-storage issues](https://github.com/milvus-io/milvus-storage/issues)
- Documentation: [GitHub Repository](https://github.com/milvus-io/milvus-storage)