#!/usr/bin/env bash

set -e

# Check if Minio is already running locally (possibly outside Docker)
if curl -s http://localhost:9000/minio/health/live &> /dev/null; then
    echo "Minio is already running locally and healthy."
else
    # Check if Minio container exists in docker
    if [ "$(docker ps -aq -f name=minio)" ]; then
        if [ ! "$(docker ps -q -f name=minio)" ]; then
            echo "Starting existing minio container..."
            docker start minio
        else
            echo "Minio container is already running."
        fi
    else
        echo "Creating and starting new minio container..."
        # milvusdb/minio:RELEASE.2024-12-18T13-15-44Z requires Content-MD5 for
        # DeleteObjects, but our AWS SDK sends CRC checksums, breaking test cleanup.
        # Pin Chainguard's MinIO RELEASE.2026-09-22T19-25-18Z for amd64 and arm64.
        # /data matches the server data path below; MinIO does not require this name.
        # The image pre-creates .minio.sys there. A volume avoids overlayfs EXDEV
        # errors when MinIO renames those directories during startup.
        docker run -d -p 9000:9000 -p 9001:9001 --name minio \
          -v /data \
          -e "MINIO_ROOT_USER=minioadmin" \
          -e "MINIO_ROOT_PASSWORD=minioadmin" \
          cgr.dev/chainguard/minio:latest@sha256:71674988a1c7ddd5724928633199152b11e4ddefd6c6ce2d60772ff4a8f22ca9 server /data --console-address ":9001"
    fi
fi

# Install s3cmd if not present
if ! command -v s3cmd &> /dev/null; then
    echo "Installing s3cmd..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y s3cmd
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install s3cmd
        else
            echo "Homebrew not found. Please install s3cmd manually."
            exit 1
        fi
    else
        echo "Unsupported OS for automatic s3cmd installation. Please install s3cmd manually."
        exit 1
    fi
fi

# Configure s3cmd for minio
cat <<EOF > ~/.s3cfg
[default]
access_key = minioadmin
secret_key = minioadmin
host_base = localhost:9000
host_bucket = localhost:9000
use_https = False
EOF

# Wait for minio to be ready
echo "Waiting for Minio to be ready..."
max_retries=10
count=0
while [ $count -lt $max_retries ]; do
  if curl -s http://localhost:9000/minio/health/live; then
    echo "Minio is ready"
    break
  fi
  echo "Still waiting for Minio... ($((count + 1))/$max_retries)"
  sleep 2
  count=$((count + 1))
done

if [ $count -eq $max_retries ]; then
    echo "Minio failed to start in time"
    exit 1
fi

# Create bucket for test if not exists
if s3cmd ls s3://test-bucket &> /dev/null; then
    echo "Bucket test-bucket already exists"
else
    echo "Creating bucket test-bucket..."
    s3cmd mb s3://test-bucket
fi
