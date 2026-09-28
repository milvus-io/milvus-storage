"""Remote read errors must keep their server diagnostics in Python."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from milvus_storage import Filesystem
from milvus_storage.exceptions import FFIError


@pytest.mark.cloud
def test_large_s3_error_body_keeps_trailing_server_reason():
    marker = "e2e-tail-reason-98765"
    response = (
        "<Error><Code>AccessDenied</Code><Message>"
        + "x" * 64_000
        + marker
        + "</Message><RequestId>e2e-request</RequestId></Error>"
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", "16")
            self.send_header("ETag", '"e2e-etag"')
            self.end_headers()

        def do_GET(self):
            self.send_response(403)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}"
    properties = {
        "fs.storage_type": "remote",
        "fs.cloud_provider": "aws",
        "fs.address": address,
        "fs.bucket_name": "e2e",
        "fs.access_key_id": "e2e-user",
        "fs.access_key_value": "e2e-key",
        "fs.region": "us-east-1",
        "fs.request_timeout_ms": "1000",
    }
    try:
        with Filesystem.get(properties=properties) as fs:
            with fs.open_reader("object") as reader:
                with pytest.raises(FFIError) as failure:
                    reader.read_at(0, 1)
        message = str(failure.value)
        assert marker in message
        assert "AccessDenied" in message
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
