# Copyright 2026 Zilliz
# Licensed under the Apache License, Version 2.0.

# Header-only SDK extensions for 1.11.921. Expose the borrowed native handle
# and per-client interceptor registration without changing layout or vtables.
# Keep the dependency cache immutable: the overlay belongs to this build tree.
# Refresh discovery when the SDK changes in an existing build directory.
unset(STORAGE_AWS_SDK_INCLUDE_DIR CACHE)
find_path(STORAGE_AWS_SDK_INCLUDE_DIR aws/core/client/AWSClient.h
  HINTS ${AWSSDK_INCLUDE_DIRS} NO_DEFAULT_PATH REQUIRED)
set(sdk_overlay "${CMAKE_CURRENT_BINARY_DIR}/sdk-extension")
if(WITH_CRT)
  set(crt_header "${STORAGE_AWS_SDK_INCLUDE_DIR}/aws/s3-crt/S3CrtClient.h")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${crt_header}")
  file(READ "${crt_header}" sdk_header)
  set(native_member "  struct aws_s3_client* m_s3CrtClient = {};")
  string(FIND "${sdk_header}" "${native_member}" member_offset)
  if(member_offset EQUAL -1)
    message(FATAL_ERROR "Unsupported S3CrtClient header: review the native handle extension for this SDK version")
  endif()
  string(REPLACE "${native_member}"
    "public:
  // Borrowed handle. The S3CrtClient must outlive every submitted request.
  struct aws_s3_client* GetUnderlyingS3Client() const noexcept { return m_s3CrtClient; }
  using Aws::Client::AWSClient::GetCredentialsProvider;
private:
${native_member}"
    sdk_header "${sdk_header}")
  file(MAKE_DIRECTORY "${sdk_overlay}/aws/s3-crt")
  file(CONFIGURE OUTPUT "${sdk_overlay}/aws/s3-crt/S3CrtClient.h"
    CONTENT "${sdk_header}" @ONLY)
endif()
# The pinned SDK supports per-client transmit interceptors but exposes no
# registration method. Add a construction-time accessor; keep the SDK ABI.
set(core_header "${STORAGE_AWS_SDK_INCLUDE_DIR}/aws/core/client/AWSClient.h")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${core_header}")
file(READ "${core_header}" sdk_header)
set(interceptor_list "Aws::Vector<std::shared_ptr<smithy::interceptor::Interceptor>> m_interceptors;")
string(FIND "${sdk_header}" "${interceptor_list}" member_offset)
if(member_offset EQUAL -1)
  message(FATAL_ERROR "Unsupported AWSClient header: review the request interceptor extension")
endif()
string(REPLACE "${interceptor_list}" "${interceptor_list}
        public:
            void AddRequestInterceptor(std::shared_ptr<smithy::interceptor::Interceptor> interceptor) {
                m_interceptors.emplace_back(std::move(interceptor));
            }
        private:" sdk_header "${sdk_header}")
file(MAKE_DIRECTORY "${sdk_overlay}/aws/core/client")
file(CONFIGURE OUTPUT "${sdk_overlay}/aws/core/client/AWSClient.h" CONTENT "${sdk_header}" @ONLY)
# All targets in this build see the same extended class declaration.
include_directories(BEFORE "${sdk_overlay}")
