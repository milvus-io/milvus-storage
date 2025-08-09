#!/bin/bash

# Cloud Provider Configuration and Test Runner
# This file contains configurations for 5 cloud providers that can be executed in a loop

# AWS Configuration
aws() {
    export AWS_STS_REGIONAL_ENDPOINTS=regional
    export AWS_REGION=
    export AWS_ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME
    export AWS_WEB_IDENTITY_TOKEN_FILE=/path/to/aws_kc
    export BUCKET_NAME=your-aws-bucket-name
    export CLOUD_PROVIDER=aws
    export ADDRESS=s3.us-west-2.amazonaws.com
}

# GCP Configuration
gcp() {
    export REGION=gcp-us-west1
    export BUCKET_NAME=your-gcp-bucket-name
    export CLOUD_PROVIDER=gcp
    export ADDRESS=storage.googleapis.com
}

# Azure Configuration
azure() {
    export AZURE_AUTHORITY_HOST=https://login.microsoftonline.com/
    export AZURE_CLIENT_ID=fa90c202-afe4-4a00-a20b-b31248062a5b
    export AZURE_TENANT_ID=7b1e450c-99dd-4e36-9f88-5fbfd0d8cc5f
    export AZURE_FEDERATED_TOKEN_FILE=/home/shaoting/workspace/milvus-storage/azure_kc
    export BUCKET_NAME=zilliz-az-westus3-1-zyebjqwhru4
    export REGION=az-westus3
    export CLOUD_PROVIDER=azure
    export ADDRESS=blob.core.windows.net
}

# Alibaba Cloud Configuration
aliyun() {
    export ALIBABA_CLOUD_ROLE_ARN=acs:ram::ACCOUNT_ID:role/ROLE_NAME
    export ALIBABA_CLOUD_OIDC_PROVIDER_ARN=acs:ram::ACCOUNT_ID:oidc-provider/PROVIDER_NAME
    export ALIBABA_CLOUD_OIDC_TOKEN_FILE=/path/to/aliyun_kc
    export BUCKET_NAME=your-aliyun-bucket-name
    export REGION=cn-hangzhou
    export CLOUD_PROVIDER=aliyun
    export ADDRESS=oss-cn-hangzhou.aliyuncs.com
}

# Tencent Cloud Configuration
tencent() {
    export TKE_REGION=ap-nanjing
    export TKE_PROVIDER_ID=your-provider-id
    export TKE_ROLE_ARN=qcs::cam::uin/ACCOUNT_ID:roleName/ROLE_NAME
    export TKE_WEB_IDENTITY_TOKEN_FILE=/path/to/tencent_kc
    export BUCKET_NAME=your-tencent-bucket-name
    export REGION=ap-nanjing
    export CLOUD_PROVIDER=tencent
    export ADDRESS=cos.ap-nanjing.myqcloud.com
}

# List of all cloud providers
CLOUD_PROVIDERS=("aws" "gcp" "azure" "aliyun" "tencent")

# Function to run tests for a specific cloud provider
run_cloud_test() {
    local provider=$1
    echo "=== Running tests for $provider ==="
    
    # Source the configuration for the specific provider
    $provider
    
    # Display current configuration
    echo "Current configuration:"
    echo "  CLOUD_PROVIDER: $CLOUD_PROVIDER"
    echo "  REGION: $REGION"
    echo "  BUCKET_NAME: $BUCKET_NAME"
    echo "  ADDRESS: $ADDRESS"

    build/Release/test/milvus_test --gtest_filter="*TestOneFile*"

    echo "=== Completed tests for $provider ==="
    echo
}

# Function to run tests for all cloud providers
run_all_cloud_tests() {
    echo "Starting tests for all cloud providers..."
    echo
    
    for provider in "${CLOUD_PROVIDERS[@]}"; do
        run_cloud_test "$provider"
    done
    
    echo "All cloud provider tests completed!"
}

# Main script execution
echo "Cloud Provider Test Runner"
echo "=========================="
echo

# Check if a specific provider was requested
if [ $# -eq 1 ]; then
    provider=$1
    if [[ " ${CLOUD_PROVIDERS[@]} " =~ " ${provider} " ]]; then
        echo "Running tests for specific provider: $provider"
        run_cloud_test "$provider"
    else
        echo "Error: Unknown provider '$provider'"
        echo "Available providers: ${CLOUD_PROVIDERS[*]}"
        exit 1
    fi
else
    echo "Running tests for all cloud providers..."
    run_all_cloud_tests
fi

echo "Test execution completed!" 