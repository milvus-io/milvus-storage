#!/usr/bin/env python3
"""Run native S3 tests against an isolated in-memory HTTP fixture (no AWS account).

Run only inside wt-build. Delayed responses expose synchronous request waits;
this fixture does not establish real-service authentication/TLS compatibility.
"""
import http.server
import uuid
import xml.etree.ElementTree as ET
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

    metadata = {}
    uploads = {}

    def log_message(self, *_):
        pass

    def reply(self, code, body=b"", headers=None, length=None):
        self.close_connection = True
        self.send_response(code)
        self.send_header("Connection", "close")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body) if length is None else length))
        if not any(k.lower() == "content-type" for k in (headers or {})):
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
        if key in ("root/slow", "root/slow-write") or query.get("prefix") == ["root/slow-list/"]:
            time.sleep(0.3)
        if not bucket and self.command == "GET":
            if "continuation-token" not in query:
                return self.reply(200, b"<ListAllMyBucketsResult><ContinuationToken>next</ContinuationToken></ListAllMyBucketsResult>")
            return self.reply(200, b"<ListAllMyBucketsResult><Buckets><Bucket><Name>bucket</Name></Bucket></Buckets></ListAllMyBucketsResult>")
        if bucket != "bucket":
            return self.reply(404)
        if key == "root/denied":
            return self.reply(403, b"<Error><Code>AccessDenied</Code></Error>")
        if self.command in ("PUT", "POST", "DELETE"):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.command == "POST" and "delete" in query:
                xml = ET.fromstring(body)
                keys = [item.text for item in xml.iter() if item.tag.rsplit("}", 1)[-1] == "Key"]
                for object_key in keys:
                    self.objects.pop(object_key, None)
                    self.metadata.pop(object_key, None)
                deleted = "".join("<Deleted><Key>" + escape(k) + "</Key></Deleted>" for k in keys)
                return self.reply(200, ("<DeleteResult>" + deleted + "</DeleteResult>").encode())
            if self.command == "DELETE" and "uploadId" in query:
                self.uploads.pop(query["uploadId"][0], None)
                return self.reply(204)
            if self.command == "DELETE":
                if "If-Match" in self.headers and self.headers["If-Match"] != '"8aa99b1f439ff71293e95357bac6fd94"':
                    return self.reply(412)
                self.objects.pop(key, None)
                self.metadata.pop(key, None)
                return self.reply(204)
            if "x-amz-copy-source" in self.headers:
                src = urllib.parse.unquote(self.headers["x-amz-copy-source"]).lstrip("/").partition("/")[2]
                if src not in self.objects:
                    return self.reply(404)
                if key == "root/copy-error":
                    return self.reply(200, b"<Error><Code>InternalError</Code></Error>")
                self.objects[key] = self.objects[src]
                self.metadata[key] = dict(self.metadata.get(src, {}))
                return self.reply(200, b'<CopyObjectResult><ETag>"8aa99b1f439ff71293e95357bac6fd94"</ETag></CopyObjectResult>')
            if self.command == "POST" and "uploads" in query:
                upload_id = str(uuid.uuid4())
                self.uploads[upload_id] = {"parts": {}, "key": key}
                return self.reply(200, ("<InitiateMultipartUploadResult><UploadId>" + upload_id +
                                       "</UploadId></InitiateMultipartUploadResult>").encode())
            if "uploadId" in query:
                upload = self.uploads.get(query["uploadId"][0])
                if not upload or upload["key"] != key:
                    return self.reply(404)
                if self.command == "PUT":
                    upload["parts"][int(query["partNumber"][0])] = body
                    return self.reply(200)
                if key == "root/error-complete":
                    return self.reply(200, b"<Error><Code>InternalError</Code></Error>")
                xml = ET.fromstring(body)
                numbers = [int(item.text) for item in xml.iter() if item.tag.endswith("PartNumber")]
                body = b"".join(upload["parts"][n] for n in numbers)
            if self.headers.get("If-None-Match") == "*" and key in self.objects:
                return self.reply(412)
            if "If-Match" in self.headers and self.headers["If-Match"] != '"8aa99b1f439ff71293e95357bac6fd94"':
                return self.reply(412)
            self.objects[key] = body
            self.metadata[key] = {k: v for k, v in self.headers.items()
                                  if k.lower().startswith("x-amz-meta-") or k.lower() == "content-type"}
            if self.command == "POST":
                self.uploads.pop(query["uploadId"][0], None)
                return self.reply(200, b'<CompleteMultipartUploadResult><ETag>"8aa99b1f439ff71293e95357bac6fd94"</ETag></CompleteMultipartUploadResult>')
            return self.reply(200)
        if self.command == "HEAD":
            if not key:
                return self.reply(200)
            if key not in self.objects:
                return self.reply(404)
            return self.reply(200, headers=self.metadata.get(key), length=len(self.objects[key]))
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
            xml = ("<ListBucketResult><KeyCount>" + str(len(batch)) + "</KeyCount><IsTruncated>" +
                   str(more).lower() + "</IsTruncated>")
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
    do_PUT = handle_request
    do_POST = handle_request
    do_DELETE = handle_request


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
