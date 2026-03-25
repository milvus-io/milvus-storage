# loon — CLI tool for manifest-based storage

`loon` is a command-line tool for exploring external data sources
(Parquet, Vortex, Lance, Iceberg) and reading data through the
manifest-based loon storage system.

## Build

The tool is built automatically when `WITH_UT=ON`:

```bash
export LDFLAGS="-L/opt/homebrew/opt/llvm@16/lib/c++ -lc++abi"
CMAKE_POLICY_VERSION_MINIMUM=3.5 make build
```

The binary is located at `cpp/build/Release/tools/loon`.

## Commands

### demo-table

Create a demo table for testing.

```
loon demo-table --type <type> --path <dir> \
                [--rows N] [--deletes pos1,pos2,...]
```

Supported types: `iceberg`

Creates a table with schema `(id int64, name string, value float64)`
and data: `id=0..N-1, name="row_0".."row_{N-1}", value=id*1.5`.

### create

Explore an external data source and commit a manifest.

```
loon create --format <format> --source <uri> \
            --target <base_path> --columns col1,col2,... \
            [--prop key=value ...]
```

Supported formats: `parquet`, `vortex`, `lance-table`, `iceberg-table`

Format-specific notes:

| Format          | `--source`                  | Extra props                          |
|-----------------|-----------------------------|--------------------------------------|
| parquet         | directory of .parquet files | —                                    |
| vortex          | directory of .vortex files  | —                                    |
| lance-table     | lance dataset directory     | —                                    |
| iceberg-table   | metadata.json path          | `--prop iceberg.snapshot_id=N`       |

### describe

Dump a manifest file as formatted JSON.

```
loon describe <manifest_path> [--prop key=value ...]
```

Outputs the full manifest structure including column groups, files,
delta logs, stats, and indexes.

### read

Read data from a committed manifest.

```
loon read <manifest_path> --columns col1,col2,... \
          [--take pos1,pos2,...] [--prop key=value ...]
```

- Without `--take`: sequential read (all row groups printed).
- With `--take`: random access at specified row positions.

## Examples

### End-to-end Iceberg demo

```bash
# 1. Create a demo Iceberg table (100 rows, delete rows 3,7,15)
loon demo-table --type iceberg --path /tmp/demo \
  --rows 100 --deletes 3,7,15

# 2. Ingest into a manifest
loon create --format iceberg-table \
  --source /tmp/demo/metadata/v1.metadata.json \
  --target /tmp/my_table \
  --columns id,name,value \
  --prop iceberg.snapshot_id=1

# 3. Inspect the manifest
loon describe /tmp/my_table/_metadata/manifest-1.avro

# 4. Read all data (deleted rows filtered out)
loon read /tmp/my_table/_metadata/manifest-1.avro \
  --columns id,name,value

# 5. Random access
loon read /tmp/my_table/_metadata/manifest-1.avro \
  --columns id,name --take 0,5,10,15
```

### Other formats

```bash
# Parquet directory
loon create --format parquet \
  --source /data/parquets/ \
  --target /tmp/my_table \
  --columns id,name,value
```

## Properties

Use `--prop key=value` to configure filesystem and format options.
Common properties:

| Property            | Description                        |
|---------------------|------------------------------------|
| `fs.root.path`      | Filesystem root path (default `/`) |
| `fs.address`        | Cloud storage endpoint             |
| `fs.bucket.name`    | Cloud storage bucket               |
| `fs.access.key`     | Cloud storage access key           |
| `fs.secret.key`     | Cloud storage secret key           |
| `iceberg.snapshot_id` | Iceberg snapshot ID (required for iceberg-table) |
