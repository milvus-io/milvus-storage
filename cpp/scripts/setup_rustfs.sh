#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/rustfs_env.sh"

if docker container inspect rustfs &> /dev/null; then
    echo "Starting existing RustFS container..."
    docker start rustfs
else
    echo "Creating and starting new RustFS container..."
    # Keep MinIO's ports free so both backends can run in the same test job.
    docker run -d -p 127.0.0.1:9002:9000 --name rustfs \
      -e "RUSTFS_ACCESS_KEY=${TEST_ENV_ACCESS_KEY}" \
      -e "RUSTFS_SECRET_KEY=${TEST_ENV_SECRET_KEY}" \
      -e "RUSTFS_CONSOLE_ENABLE=false" \
      rustfs/rustfs:1.0.0 /data
fi

echo "Waiting for RustFS to be ready..."
max_retries=30
count=0
while [ "$count" -lt "$max_retries" ]; do
    if curl -fsS --max-time 5 "${TEST_ENV_ADDRESS}/health/ready" > /dev/null; then
        echo "RustFS is ready"
        break
    fi
    echo "Still waiting for RustFS... ($((count + 1))/$max_retries)"
    sleep 2
    count=$((count + 1))
done

if [ "$count" -eq "$max_retries" ]; then
    echo "RustFS failed to start in time" >&2
    docker logs rustfs >&2 || true
    exit 1
fi

if ! command -v s3cmd &> /dev/null; then
    echo "Installing s3cmd..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y s3cmd
    elif [[ "$OSTYPE" == "darwin"* ]] && command -v brew &> /dev/null; then
        brew install s3cmd
    else
        echo "Please install s3cmd manually." >&2
        exit 1
    fi
fi

# Use a temporary config so MinIO and local s3cmd settings remain intact.
s3_config=$(mktemp)
trap 'rm -f "$s3_config"' EXIT
cat > "$s3_config" <<EOF
[default]
access_key = ${TEST_ENV_ACCESS_KEY}
secret_key = ${TEST_ENV_SECRET_KEY}
host_base = localhost:9002
host_bucket = localhost:9002
bucket_location = ${TEST_ENV_REGION}
use_https = False
EOF

if s3cmd -c "$s3_config" info "s3://${TEST_ENV_BUCKET_NAME}" &> /dev/null; then
    echo "Bucket ${TEST_ENV_BUCKET_NAME} already exists"
else
    echo "Creating bucket ${TEST_ENV_BUCKET_NAME}..."
    s3cmd -c "$s3_config" mb "s3://${TEST_ENV_BUCKET_NAME}"
fi
