from milvus_storage import Filesystem


def test_metrics_snapshot_deltas_and_continue_counting(temp_case_path, default_properties):
    path = f"{temp_case_path}/metrics.bin"
    with Filesystem.get(properties=default_properties) as fs:
        fs.create_dir(temp_case_path)
        # Cached filesystems may already have metrics, so compare snapshot deltas.
        before = {entry.source: entry.metrics for entry in fs.get_metrics_sources()}
        fs.write_file(path, b"abcdefgh")
        assert fs.read_file(path, 2, 3) == b"cde"
        metrics = {entry.source: entry.metrics for entry in fs.get_metrics_sources()}
        assert metrics["origin"].write_count - before["origin"].write_count >= 1
        assert metrics["origin"].read_count - before["origin"].read_count >= 1
        assert fs.read_file(path, 0, 2) == b"ab"
        continued = {entry.source: entry.metrics for entry in fs.get_metrics_sources()}
        assert continued["origin"].read_count - metrics["origin"].read_count >= 1
