#!/usr/bin/env bash

set -e

# Azurite well-known test credentials (not real secrets):
# https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite#well-known-storage-account-and-key
AZURITE_ACCOUNT="devstoreaccount1"
AZURITE_KEY="Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
AZURITE_BLOB_PORT=10000
AZURITE_BLOB_HOST="127.0.0.1"
AZURITE_CONTAINER="test-container"

# Check if Azurite is already running locally
if curl -s "http://${AZURITE_BLOB_HOST}:${AZURITE_BLOB_PORT}/${AZURITE_ACCOUNT}?comp=list" &> /dev/null; then
    echo "Azurite is already running locally."
else
    # Check if Azurite container exists in docker
    if [ "$(docker ps -aq -f name=azurite)" ]; then
        if [ ! "$(docker ps -q -f name=azurite)" ]; then
            echo "Starting existing azurite container..."
            docker start azurite
        else
            echo "Azurite container is already running."
        fi
    else
        echo "Creating and starting new azurite container..."
        docker run -d -p ${AZURITE_BLOB_PORT}:${AZURITE_BLOB_PORT} --name azurite \
          mcr.microsoft.com/azure-storage/azurite \
          azurite-blob --blobHost 0.0.0.0 --blobPort ${AZURITE_BLOB_PORT} --skipApiVersionCheck
    fi
fi

# Wait for Azurite to be ready
echo "Waiting for Azurite to be ready..."
max_retries=10
count=0
while [ $count -lt $max_retries ]; do
  if curl -s "http://${AZURITE_BLOB_HOST}:${AZURITE_BLOB_PORT}/${AZURITE_ACCOUNT}?comp=list" &> /dev/null; then
    echo "Azurite is ready"
    break
  fi
  echo "Still waiting for Azurite... ($((count + 1))/$max_retries)"
  sleep 2
  count=$((count + 1))
done

if [ $count -eq $max_retries ]; then
    echo "Azurite failed to start in time"
    exit 1
fi

# Install az CLI if not present
if ! command -v az &> /dev/null; then
    echo "Installing Azure CLI..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install azure-cli
        else
            echo "Homebrew not found. Please install Azure CLI manually: https://aka.ms/InstallAzureCli"
            exit 1
        fi
    else
        echo "Unsupported OS. Please install Azure CLI manually: https://aka.ms/InstallAzureCli"
        exit 1
    fi
fi

# Create test container using az CLI
AZURITE_CONN_STR="DefaultEndpointsProtocol=http;AccountName=${AZURITE_ACCOUNT};AccountKey=${AZURITE_KEY};BlobEndpoint=http://${AZURITE_BLOB_HOST}:${AZURITE_BLOB_PORT}/${AZURITE_ACCOUNT};"

if az storage container show -n "${AZURITE_CONTAINER}" --connection-string "${AZURITE_CONN_STR}" &> /dev/null; then
    echo "Container ${AZURITE_CONTAINER} already exists"
else
    echo "Creating container ${AZURITE_CONTAINER}..."
    az storage container create -n "${AZURITE_CONTAINER}" --connection-string "${AZURITE_CONN_STR}" --output none
    echo "Container created"
fi

echo ""
echo "Azurite is ready. Use these env vars for tests:"
echo "  export TEST_ENV_ACCESS_KEY=${AZURITE_ACCOUNT}"
echo "  export TEST_ENV_SECRET_KEY=${AZURITE_KEY}"
echo "  export TEST_ENV_ADDRESS=${AZURITE_BLOB_HOST}:${AZURITE_BLOB_PORT}"
echo "  export TEST_ENV_CLOUD_PROVIDER=azure"
echo "  export TEST_ENV_BUCKET_NAME=${AZURITE_CONTAINER}"
echo "  export TEST_ENV_STORAGE_TYPE=remote"
