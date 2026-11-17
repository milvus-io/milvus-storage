if (NOT TARGET OpenSSL::Crypto)
  find_package(OpenSSL REQUIRED)
endif()

add_executable(aliyun_credentials_test
  aliyun_credentials_test.cpp
  native_sts_credentials_test.cpp
  ../../src/filesystem/s3/provider/AliyunCredentialsProvider.cpp
  ../../src/filesystem/s3/provider/AliyunSTSClient.cpp
  ../../src/filesystem/s3/provider/AliyunRAMCredentialsProvider.cpp
  ../../src/filesystem/s3/provider/AliyunRAMSTSClient.cpp
  ../../src/filesystem/s3/provider/AliyunOIDCAssumeRoleChainProvider.cpp
  ../../src/filesystem/s3/provider/TencentCloudCredentialsProvider.cpp
  ../../src/filesystem/s3/provider/TencentCloudSTSClient.cpp
  ../../src/filesystem/s3/provider/TencentCloudOIDCAssumeRoleChainProvider.cpp
  ../../src/filesystem/s3/provider/HuaweiCloudCredentialsProvider.cpp
  ../../src/filesystem/s3/provider/HuaweiCloudSTSClient.cpp
  ../../src/filesystem/s3/provider/VolcengineCredentialsProvider.cpp
  ../../src/filesystem/s3/provider/VolcengineSTSClient.cpp
  ../../src/filesystem/s3/provider/VolcengineOIDCAssumeRoleChainProvider.cpp
  ../../src/common/log.cpp)
target_compile_features(aliyun_credentials_test PRIVATE cxx_std_17)
target_include_directories(aliyun_credentials_test PRIVATE ../../include)
target_link_libraries(aliyun_credentials_test PRIVATE
  AWS::aws-sdk-cpp-core OpenSSL::Crypto arrow::arrow fmt::fmt glog::glog GTest::gtest_main)
if (NOT APPLE)
  target_link_options(aliyun_credentials_test PRIVATE "LINKER:--disable-new-dtags")
endif()
add_test(NAME aliyun_credentials_test COMMAND aliyun_credentials_test)
