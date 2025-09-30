# Milvus Storage DataFusion Integration

This library provides a DataFusion `TableProvider` implementation for Milvus Storage, allowing you to query Milvus datasets using SQL through Apache DataFusion.

## Features

- **SQL Interface**: Query Milvus storage datasets using standard SQL
- **Arrow Integration**: Seamless integration with Apache Arrow data format
- **Filter Pushdown**: Efficient query execution with predicate pushdown to Milvus
- **Streaming**: Support for streaming large datasets
- **Type Safety**: Rust's type system ensures memory safety when interfacing with C++

## Architecture

The integration consists of several key components:

1. **FFI Bindings** (`ffi.rs`): Safe Rust wrappers around the milvus-storage C API
2. **TableProvider** (`table_provider.rs`): DataFusion integration implementing the `TableProvider` trait
3. **ExecutionPlan** (`execution_plan.rs`): Physical execution plan for query execution
4. **RecordBatchStream** (`record_batch_stream.rs`): Streaming interface for reading data

## Usage

### Basic Example

```rust
use std::sync::Arc;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use milvus_storage_datafusion::MilvusTableProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create DataFusion context
    let ctx = SessionContext::new();

    // Define your schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("vector", DataType::FixedSizeBinary(128), false),
        Field::new("metadata", DataType::Utf8, true),
    ]));

    // Create filesystem handle and manifest (see examples for details)
    let fs_handle = create_filesystem_handle()?;
    let manifest = load_manifest("path/to/manifest.json")?;

    // Create table provider
    let table_provider = MilvusTableProvider::try_new(
        fs_handle,
        &manifest,
        schema,
        "my_table".to_string(),
        None,
        None,
    )?;

    // Register table
    ctx.register_table("vectors", Arc::new(table_provider))?;

    // Query with SQL
    let df = ctx.sql("SELECT id, metadata FROM vectors WHERE id > 100 LIMIT 10").await?;
    df.show().await?;

    Ok(())
}
```

### Advanced Usage

#### Column Projection
```rust
// Only read specific columns
let columns = Some(&["id", "metadata"][..]);
let table_provider = MilvusTableProvider::new(
    fs_handle,
    &manifest,
    schema,
    "my_table".to_string(),
    columns,
    None,
)?;
```

#### Read Properties
```rust
// Configure read behavior
let properties = vec![
    ("batch_size".to_string(), "8192".to_string()),
    ("parallel_reads".to_string(), "4".to_string()),
];

let table_provider = MilvusTableProvider::new(
    fs_handle,
    &manifest,
    schema,
    "my_table".to_string(),
    None,
    Some(properties),
)?;
```

#### Complex Queries
```rust
// Aggregations
let df = ctx.sql("SELECT COUNT(*), AVG(score) FROM vectors WHERE category = 'active'").await?;

// Joins (if you have multiple Milvus tables)
let df = ctx.sql(r#"
    SELECT v.id, v.vector, m.description 
    FROM vectors v 
    JOIN metadata m ON v.id = m.vector_id
    WHERE v.score > 0.8
"#).await?;

// Window functions
let df = ctx.sql(r#"
    SELECT id, score, 
           ROW_NUMBER() OVER (ORDER BY score DESC) as rank
    FROM vectors 
    WHERE category = 'premium'
"#).await?;
```

## Schema Requirements

Your Milvus storage schema should be compatible with Apache Arrow. The library supports:

- **Scalar Types**: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Boolean, Utf8
- **Binary Types**: Binary, FixedSizeBinary (for vectors)
- **Temporal Types**: Date32, Date64, Timestamp
- **Nullable Types**: All types can be nullable

### Vector Columns

Vector columns should use `FixedSizeBinary` type:

```rust
Field::new("embedding", DataType::FixedSizeBinary(768), false) // 768-dimensional vector
```

## Building

### Prerequisites

- Rust 1.70+
- C++ compiler (g++ or clang++)
- milvus-storage C++ library built and installed
- pkg-config

### Build Steps

```bash
# Build the milvus-storage C++ library first
cd cpp
mkdir build && cd build
cmake ..
make -j$(nproc)

# Build the Rust integration
cd ../rust
cargo build --release
```

### Dependencies

The library depends on:
- `datafusion`: DataFusion query engine
- `arrow`: Apache Arrow implementation
- `tokio`: Async runtime
- `bindgen`: C++ binding generation

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.rs`: Simple table scanning and filtering
- `advanced_queries.rs`: Complex SQL operations
- `filesystem_setup.rs`: Setting up different filesystem backends

## Testing

```bash
cargo test
```

Note: Tests require a running milvus-storage setup with test data.

## Error Handling

The library uses a comprehensive error type that covers:

- FFI errors from the C interface
- Arrow conversion errors  
- DataFusion execution errors
- I/O errors

```rust
use milvus_storage_datafusion::{MilvusError, Result};

match table_provider.scan(...).await {
    Ok(plan) => { /* handle success */ },
    Err(MilvusError::Ffi(msg)) => { /* handle FFI error */ },
    Err(MilvusError::Arrow(err)) => { /* handle Arrow error */ },
    Err(e) => { /* handle other errors */ },
}
```

## Performance Considerations

- **Batch Size**: Tune `batch_size` based on your memory constraints and query patterns
- **Parallelism**: Use `parallel_reads` to control I/O parallelism
- **Projection**: Only select columns you need to reduce I/O
- **Filters**: Push filters down to Milvus for better performance

## Limitations

- Single partition execution plans (no intra-query parallelism yet)
- Limited filter pushdown translation
- No support for custom vector similarity functions (yet)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Licensed under the Apache License, Version 2.0.
