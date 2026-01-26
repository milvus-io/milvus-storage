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