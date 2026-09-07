#!/usr/bin/env python3
"""Seed a MinIO/S3 object for the Talon integration test using only the Python
standard library. The milvus builder image ships no s3cmd/mc/aws/boto3, so
scripts/setup_talon.sh calls this instead to guarantee the test object exists.

Path-style requests, AWS Signature V4. Idempotent: it HEADs the object and
uploads --size-mb MiB of deterministic bytes only when the object is missing,
creating the bucket first if needed. This is a test fixture, not a general S3
client (single unsigned-streaming PUT, no multipart, minimal error handling).

Usage:
  s3_seed.py --endpoint http://minio:9000 --bucket B --key path/to/obj \
             --access KEY --secret SECRET [--size-mb 64] [--region us-east-1]
Exit 0 when the object is present (already there or freshly uploaded).
"""
import argparse
import datetime
import hashlib
import hmac
import sys
import urllib.error
import urllib.request

_SERVICE = "s3"
_ALGO = "AWS4-HMAC-SHA256"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret: str, datestamp: str, region: str) -> bytes:
    k = _hmac(("AWS4" + secret).encode("utf-8"), datestamp)
    k = _hmac(k, region)
    k = _hmac(k, _SERVICE)
    return _hmac(k, "aws4_request")


def _signed_request(method, endpoint, canonical_uri, access, secret, region, body):
    """Build a SigV4-signed urllib Request for a path-style S3 call."""
    host = endpoint.split("://", 1)[1]
    payload_hash = _sha256_hex(body)
    now = datetime.datetime.utcnow()
    amzdate = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")

    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amzdate}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = (
        f"{method}\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )
    scope = f"{datestamp}/{region}/{_SERVICE}/aws4_request"
    string_to_sign = f"{_ALGO}\n{amzdate}\n{scope}\n{_sha256_hex(canonical_request.encode('utf-8'))}"
    signature = hmac.new(
        _signing_key(secret, datestamp, region), string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    authorization = (
        f"{_ALGO} Credential={access}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    req = urllib.request.Request(endpoint + canonical_uri, data=body, method=method)
    req.add_header("Host", host)
    req.add_header("x-amz-content-sha256", payload_hash)
    req.add_header("x-amz-date", amzdate)
    req.add_header("Authorization", authorization)
    return req


def _call(method, endpoint, canonical_uri, access, secret, region, body=b""):
    req = _signed_request(method, endpoint, canonical_uri, access, secret, region, body)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--access", required=True)
    ap.add_argument("--secret", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--size-mb", type=int, default=64)
    args = ap.parse_args()

    obj_uri = f"/{args.bucket}/{args.key}"

    # Already present? HEAD returns 200 when the object exists.
    status, _ = _call("HEAD", args.endpoint, obj_uri, args.access, args.secret, args.region)
    if status == 200:
        print(f"Test object s3://{args.bucket}/{args.key} already present")
        return 0

    # Ensure the bucket exists (200 fresh, 409 already-owned are both fine).
    status, body = _call("PUT", args.endpoint, f"/{args.bucket}/", args.access, args.secret, args.region)
    if status not in (200, 409):
        sys.stderr.write(f"bucket create failed: HTTP {status}: {body[:200]!r}\n")
        return 1

    # Upload deterministic bytes so repeated seeds are byte-identical.
    nbytes = args.size_mb * 1024 * 1024
    data = (hashlib.sha256(b"talon-seed").digest() * ((nbytes // 32) + 1))[:nbytes]
    print(f"Seeding s3://{args.bucket}/{args.key} ({args.size_mb} MiB)")
    status, body = _call("PUT", args.endpoint, obj_uri, args.access, args.secret, args.region, data)
    if status not in (200, 201):
        sys.stderr.write(f"object PUT failed: HTTP {status}: {body[:200]!r}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
