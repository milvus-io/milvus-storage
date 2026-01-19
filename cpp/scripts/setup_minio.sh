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
        docker run -d -p 9000:9000 -p 9001:9001 --name minio \
          -e "MINIO_ACCESS_KEY=minioadmin" \
          -e "MINIO_SECRET_KEY=minioadmin" \
          quay.io/minio/minio server /data --console-address ":9001"
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
