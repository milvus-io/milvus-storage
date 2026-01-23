#!/usr/bin/env python
"""
Basic example of reading data from milvus-storage.
"""

import pyarrow as pa

from milvus_storage import Reader


def main():
    """Read data from milvus-storage."""

    # Load column groups metadata
    column_groups_file = "/tmp/milvus_storage_manifest.json"
    try:
        with open(column_groups_file, "r") as f:
            column_groups = f.read()
    except FileNotFoundError:
        print(f"Column groups file not found at {column_groups_file}")
        print("Please run basic_write.py first")
        return

    properties = {
        "fs.storage_type": "local",
        "fs.root_path": "/tmp/",
    }
    # Define schema (must match the schema used for writing)
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=True, metadata={"PARQUET:field_id": "1"}),
            pa.field("vector", pa.binary(5), nullable=True, metadata={"PARQUET:field_id": "2"}),
            pa.field("text", pa.string(), nullable=True, metadata={"PARQUET:field_id": "3"}),
            pa.field("score", pa.float64(), nullable=True, metadata={"PARQUET:field_id": "4"}),
        ]
    )

    print("Reading data from milvus-storage")

    # Create reader
    with Reader(column_groups, schema, properties=properties) as reader:
        print(f"Schema: {reader.schema}")

        # Example 1: Full table scan
        print("\n=== Example 1: Full table scan ===")
        batch_count = 0
        total_rows = 0

        for batch in reader.scan():
            batch_count += 1
            total_rows += len(batch)
            print(f"Batch {batch_count}: {len(batch)} rows")

            # Show first few rows
            if batch_count == 1:
                print("\nFirst batch data:")
                print(batch.to_pandas().head())

        print(f"\nTotal batches: {batch_count}")
        print(f"Total rows: {total_rows}")

        # Example 2: Filtered scan
        print("\n=== Example 2: Filtered scan ===")
        # Note: Filter syntax depends on the backend implementation
        # This is just an example
        try:
            for batch in reader.scan(predicate="id > 2"):
                print(f"Filtered batch: {len(batch)} rows")
                print(batch.to_pandas())
        except Exception as e:
            print(f"Filtering not supported or failed: {e}")

        # Example 3: Random access (take specific rows)
        # TODO: enable this when take() is implemented in the C++ backend
        # NOTE: take() is not yet implemented in the C++ backend
        # Uncomment when implementation is available
        # print("\n=== Example 3: Random access ===")
        # indices = [0, 2, 4]  # Read rows at indices 0, 2, 4
        # batch = reader.take(indices)
        # print(f"Took {len(batch)} rows at indices {indices}")
        # print(batch.to_pandas())

        # Example 4: Read specific columns only
        print("\n=== Example 4: Read specific columns ===")

    # Create a new reader with column projection
    with Reader(column_groups, schema, columns=["id", "text"], properties=properties) as reader:
        for batch in reader.scan():
            print(f"Projected batch: {batch.column_names}")
            print(batch.to_pandas())
            break  # Just show first batch


if __name__ == "__main__":
    main()
