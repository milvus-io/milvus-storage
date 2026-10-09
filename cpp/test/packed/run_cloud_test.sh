#!/bin/bash

# Cloud Provider Configuration and Test Runner
# This file contains configurations for 5 cloud providers that can be executed in a loop

# AWS Configuration
aws() {
    export AWS_STS_REGIONAL_ENDPOINTS=regional
    export AWS_ROLE_ARN=your-role-arn
    export AWS_WEB_IDENTITY_TOKEN_FILE=/path/to/aws_kc
    export TEST_ENV_ADDRESS=s3.us-west-2.amazonaws.com
    export TEST_ENV_BUCKET_NAME=your-bucket-name
    export TEST_ENV_CLOUD_PROVIDER=aws
    export TEST_ENV_REGION=us-west-2
    export TEST_ENV_USE_IAM=true
    export TEST_ENV_USE_SSL=true
}

# GCP Configuration
gcp() {
    export TEST_ENV_ADDRESS=storage.googleapis.com
    export TEST_ENV_BUCKET_NAME=your-gcp-bucket-name
    export TEST_ENV_CLOUD_PROVIDER=gcp
    export TEST_ENV_REGION=gcp-us-west1
    export TEST_ENV_USE_IAM=true
    export TEST_ENV_USE_SSL=true
}

# Azure Configuration
azure() {
    export AZURE_AUTHORITY_HOST=https://login.microsoftonline.com/
    export AZURE_CLIENT_ID=your-client-id
    export AZURE_TENANT_ID=your-tenant-id
    export AZURE_FEDERATED_TOKEN_FILE=/path/to/azure_kc
    export TEST_ENV_ACCESS_KEY=your-access-key
    export TEST_ENV_ADDRESS=core.windows.net
    export TEST_ENV_BUCKET_NAME=your-bucket-name
    export TEST_ENV_CLOUD_PROVIDER=azure
    export TEST_ENV_REGION=your-region
    export TEST_ENV_USE_IAM=true
    export TEST_ENV_USE_SSL=true
}

# Alibaba Cloud Configuration
aliyun() {
    export ALIBABA_CLOUD_ROLE_ARN=acs:ram::ACCOUNT_ID:role/ROLE_NAME
    export ALIBABA_CLOUD_OIDC_PROVIDER_ARN=acs:ram::ACCOUNT_ID:oidc-provider/PROVIDER_NAME
    export ALIBABA_CLOUD_OIDC_TOKEN_FILE=/path/to/aliyun_kc
    export TEST_ENV_ADDRESS=oss-cn-hangzhou.aliyuncs.com
    export TEST_ENV_BUCKET_NAME=your-aliyun-bucket-name
    export TEST_ENV_CLOUD_PROVIDER=aliyun
    export TEST_ENV_REGION=cn-hangzhou
    export TEST_ENV_USE_IAM=true
    export TEST_ENV_USE_SSL=true
}

# Tencent Cloud Configuration
tencent() {
    export TKE_REGION=ap-nanjing
    export TKE_PROVIDER_ID=your-provider-id
    export TKE_ROLE_ARN=qcs::cam::uin/ACCOUNT_ID:roleName/ROLE_NAME
    export TKE_WEB_IDENTITY_TOKEN_FILE=/path/to/tencent_kc
    export TEST_ENV_ADDRESS=cos.ap-nanjing.myqcloud.com
    export TEST_ENV_BUCKET_NAME=your-tencent-bucket-name
    export TEST_ENV_CLOUD_PROVIDER=tencent
    export TEST_ENV_REGION=ap-nanjing
    export TEST_ENV_USE_IAM=true
    export TEST_ENV_USE_SSL=true
}

huawei() {
    export HUAWEICLOUD_SDK_REGION=cn-east-3
    export HUAWEICLOUD_SDK_PROJECT_ID=4930abf6e99348b79d8c8dab69683157
    export HUAWEICLOUD_SDK_ID_TOKEN_FILE=/var/run/secrets/tokens/oidc-token
    export HUAWEICLOUD_SDK_IDP_ID=k8s-1
    export TEST_ENV_ADDRESS=obs.your-region.myhuaweicloud.com
    export TEST_ENV_BUCKET_NAME=milvus-poc-bucket
    export TEST_ENV_CLOUD_PROVIDER=huawei
    export TEST_ENV_REGION=your-region
    export TEST_ENV_USE_IAM=true
    export TEST_ENV_USE_SSL=true
}

# List of all cloud providers
CLOUD_PROVIDERS=("aws" "gcp" "azure" "aliyun" "tencent" "huawei")

# Function to run tests for a specific cloud provider
run_cloud_test() {
    local provider=$1
    echo "=== Running tests for $provider ==="

    # Source the configuration for the specific provider
    $provider
    export TEST_ENV_STORAGE_TYPE=remote

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