#!/usr/bin/env python
"""
Basic example of writing data to milvus-storage.
"""

import pandas as pd
import pyarrow as pa

from milvus_storage import Writer


def main():
    """Write sample data to milvus-storage."""
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=True, metadata={"PARQUET:field_id": "1"}),
            pa.field("vector", pa.binary(5), nullable=True, metadata={"PARQUET:field_id": "2"}),
            pa.field("text", pa.string(), nullable=True, metadata={"PARQUET:field_id": "3"}),
            pa.field("score", pa.float64(), nullable=True, metadata={"PARQUET:field_id": "4"}),
        ]
    )

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "vector": ["12345", "54321", "12345", "54321", "abcde"],
            "text": ["apple", "banana", "cherry", "date", "elderberry"],
            "score": [0.9, 0.8, 0.95, 0.7, 0.85],
        }
    )
    batch = pa.RecordBatch.from_pandas(df, schema)

    # Write to storage
    storage_path = "/tmp/milvus_storage_example"
    properties = {
        "fs.storage_type": "local",
        "fs.root_path": "/tmp/",
    }

    with Writer(storage_path, schema, properties) as writer:
        writer.write(batch)
        print(f"Wrote {len(batch)} rows")

        # Can write multiple batches
        writer.write(batch)
        print(f"Wrote another {len(batch)} rows")

        # Flush to ensure data is written
        writer.flush()
        print("Flushed data to storage")

        # Close and get column groups
        column_groups = writer.close()

    # Save column groups for reading later
    with open("/tmp/milvus_storage_manifest.json", "w") as f:
        f.write(column_groups)

    print("\nColumn groups saved to /tmp/milvus_storage_manifest.json")
    print(f"Total rows written: {len(batch) * 2}")


if __name__ == "__main__":
    print(f"pyarrow version: {pa.__version__}")
    main()
