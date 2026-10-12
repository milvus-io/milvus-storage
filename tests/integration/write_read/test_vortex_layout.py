"""Storage Writer Vortex layouts through the Python API."""

import pyarrow as pa
import pytest
from milvus_storage import ChunkMetadataType, Reader, Transaction, Writer
from milvus_storage.manifest import ColumnGroups


@pytest.mark.parametrize("layout_version", [1, 2])
@pytest.mark.parametrize("stats_enabled", ["true", "false"])
@pytest.mark.e2e_smoke
def test_vortex_layout_preserves_rows_across_flushed_batches(
    temp_case_path, default_properties, layout_version, stats_enabled
):
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("payload", pa.string())])
    properties = dict(default_properties)
    properties.update(
        {
            "writer.format": "vortex",
            "writer.vortex.format_version": str(layout_version),
            "writer.vortex_v2.row_group_max_size": str(128 * 1024),
            "writer.vortex.enable_statistics": stats_enabled,
        }
    )
    expected_ids = list(range(2400))
    expected_payloads = [f"{i:05d}-" + "z" * 128 for i in expected_ids]
    with Writer(temp_case_path, schema, properties) as writer:
        for part in range(3):
            start = part * 800
            stop = start + 800
            writer.write(
                pa.RecordBatch.from_pydict(
                    {
                        "id": expected_ids[start:stop],
                        "payload": expected_payloads[start:stop],
                    },
                    schema=schema,
                )
            )
            if part == 1:
                writer.flush()
        written = writer.close()
    assert all(group.format == "vortex" for group in written.to_list())

    with Transaction(temp_case_path, properties) as txn:
        txn.append_files(written)
        txn.commit()
    written.destroy()
    with Transaction(temp_case_path, properties) as txn:
        manifest = txn.get_manifest()

    with ColumnGroups.from_list(manifest.column_groups) as groups:
        with Reader(groups, schema, properties=properties) as reader:
            scanned = pa.Table.from_batches(list(reader.scan()))
            assert scanned.column("id").to_pylist() == expected_ids
            assert scanned.column("payload").to_pylist() == expected_payloads

            selected = [0, 799, 800, 1599, 2399]
            taken = pa.Table.from_batches(reader.take(selected))
            assert taken.column("id").to_pylist() == selected
            assert taken.column("payload").to_pylist() == [
                expected_payloads[i] for i in selected
            ]

            with reader.get_chunk_reader(0) as chunk_reader:
                count = chunk_reader.get_number_of_chunks()
                if layout_version == 2:
                    assert count >= 2
                ids_from_chunks = [
                    value
                    for index in range(count)
                    for value in chunk_reader.get_chunk(index).column("id").to_pylist()
                ]
                assert ids_from_chunks == expected_ids

        if layout_version == 2:
            for split_setting in ("auto", "true", "false"):
                read_properties = dict(properties)
                read_properties["reader.logical_chunk_rows"] = "25"
                read_properties["reader.vortex.split_row_indices"] = split_setting
                with Reader(groups, schema, properties=read_properties) as reader:
                    with reader.get_chunk_reader(0) as chunk_reader:
                        rows = chunk_reader.get_chunk_metadatas(
                            ChunkMetadataType.NUM_OF_ROWS
                        )
                        row_counts = [count for item in rows for count in item.data]
                    assert sum(row_counts) == len(expected_ids)
                    assert all(0 < count <= 25 for count in row_counts)
                    selected = [0, 24, 25, 2399]
                    taken = pa.Table.from_batches(
                        reader.take(selected, columns=["payload"])
                    )
                    assert taken.column("payload").to_pylist() == [
                        expected_payloads[i] for i in selected
                    ]
