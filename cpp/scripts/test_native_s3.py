#!/usr/bin/env python3
"""Run native S3 tests against an isolated in-memory HTTP fixture (no AWS account).

Run only inside wt-build. Delayed responses expose synchronous request waits;
this fixture does not establish real-service authentication/TLS compatibility.
"""
import http.server
import os
import resource
import subprocess
import sys
import threading
import time
import urllib.parse
from xml.sax.saxutils import escape


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    objects = {"root/hello #?+% 中文": b"abcdef", "root/slow": b"abcdef",
               "root/slow-list/file": b"data", **{"root/pages/" + str(i): b"data" for i in range(5)}}

    def log_message(self, *_):
        pass

    def reply(self, code, body=b"", headers=None, length=None):
        self.close_connection = True
        self.send_response(code)
        self.send_header("Connection", "close")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body) if length is None else length))
        self.send_header("Content-Type", "application/octet-stream" if code == 206 else "application/xml")
        self.send_header("Last-Modified", "Wed, 01 Jan 2025 00:00:00 GMT")
        self.send_header("ETag", '"8aa99b1f439ff71293e95357bac6fd94"')
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def handle_request(self):
        if os.environ.get("STORAGE_NATIVE_S3_DEBUG"):
            print(self.command, self.path, self.headers.get("Range"), flush=True)
        uri = urllib.parse.urlsplit(self.path)
        path = urllib.parse.unquote(uri.path).lstrip("/")
        bucket, _, key = path.partition("/")
        query = urllib.parse.parse_qs(uri.query, keep_blank_values=True)
        if key == "root/slow" or query.get("prefix") == ["root/slow-list/"]:
            time.sleep(0.3)
        if not bucket and self.command == "GET":
            if "continuation-token" not in query:
                return self.reply(200, b"<ListAllMyBucketsResult><ContinuationToken>next</ContinuationToken></ListAllMyBucketsResult>")
            return self.reply(200, b"<ListAllMyBucketsResult><Buckets><Bucket><Name>bucket</Name></Bucket></Buckets></ListAllMyBucketsResult>")
        if bucket != "bucket":
            return self.reply(404)
        if key == "root/denied":
            return self.reply(403, b"<Error><Code>AccessDenied</Code></Error>")
        if self.command == "HEAD":
            if not key:
                return self.reply(200)
            if key not in self.objects:
                return self.reply(404)
            return self.reply(200, length=len(self.objects[key]))
        if query.get("list-type") == ["2"]:
            prefix = query.get("prefix", [""])[0]
            delimiter = query.get("delimiter", [""])[0]
            if prefix == "root/bad-token/":
                return self.reply(200, b"<ListBucketResult><IsTruncated>true</IsTruncated></ListBucketResult>")
            all_keys = sorted(k for k in self.objects if k.startswith(prefix))
            files, dirs = [], set()
            for k in all_keys:
                if delimiter and delimiter in k[len(prefix):]:
                    dirs.add(prefix + k[len(prefix):].split(delimiter)[0] + delimiter)
                else:
                    files.append(k)
            entries = [(k, False) for k in files] + [(k, True) for k in sorted(dirs)]
            start = int(query.get("continuation-token", ["0"])[0])
            count = min(2, int(query.get("max-keys", ["1000"])[0]))
            batch = entries[start:start + count]
            more = start + count < len(entries)
            xml = "<ListBucketResult><IsTruncated>" + str(more).lower() + "</IsTruncated>"
            if more:
                xml += "<NextContinuationToken>" + str(start + count) + "</NextContinuationToken>"
            for k, directory in batch:
                if directory:
                    xml += "<CommonPrefixes><Prefix>" + escape(k) + "</Prefix></CommonPrefixes>"
                else:
                    xml += ("<Contents><Key>" + escape(k) + "</Key><Size>" + str(len(self.objects[k])) +
                            "</Size><LastModified>2026-09-18T00:00:00Z</LastModified></Contents>")
            return self.reply(200, (xml + "</ListBucketResult>").encode())
        if key not in self.objects:
            return self.reply(404, b"<Error><Code>NoSuchKey</Code></Error>")
        data = self.objects[key]
        if "Range" in self.headers:
            first, last = map(int, self.headers["Range"][6:].split("-"))
            if first >= len(data):
                return self.reply(416, b"<Error><Code>InvalidRange</Code></Error>")
            last = min(last, len(data) - 1)
            return self.reply(206, data[first:last + 1], {"Content-Range": f"bytes {first}-{last}/{len(data)}"})
        return self.reply(200, data)

    do_HEAD = handle_request
    do_GET = handle_request


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = dict(os.environ, NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost", AWS_EC2_METADATA_DISABLED="true", STORAGE_NATIVE_S3_ENDPOINT=f"127.0.0.1:{server.server_port}")
    try:
        result = subprocess.run(sys.argv[1:] or ["cpp/build/Release/test/milvus_test", "--gtest_filter=NativeS3*"],
                                env=env, timeout=180)
        return result.returncode
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    sys.exit(main())
