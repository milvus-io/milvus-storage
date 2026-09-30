# Copyright 2026 Zilliz
# Licensed under the Apache License, Version 2.0.

# Header-only SDK extension for the pinned 1.11.842 dependency. Add a borrowed
# native handle accessor without changing layout, virtual methods, or ownership.
# Keep the dependency cache immutable: the overlay belongs to this build tree.
find_path(STORAGE_S3_CRT_INCLUDE_DIR aws/s3-crt/S3CrtClient.h
  HINTS ${AWSSDK_INCLUDE_DIRS} REQUIRED)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${STORAGE_S3_CRT_INCLUDE_DIR}/aws/s3-crt/S3CrtClient.h")
file(READ "${STORAGE_S3_CRT_INCLUDE_DIR}/aws/s3-crt/S3CrtClient.h" sdk_header)
set(native_member "  struct aws_s3_client* m_s3CrtClient = {};")
string(FIND "${sdk_header}" "${native_member}" member_offset)
if(member_offset EQUAL -1)
  message(FATAL_ERROR "Unsupported S3CrtClient header: review the native handle extension for this SDK version")
endif()
string(REPLACE "${native_member}"
  "public:
  // Borrowed handle. The S3CrtClient must outlive every submitted request.
  struct aws_s3_client* GetUnderlyingS3Client() const noexcept { return m_s3CrtClient; }
private:
${native_member}"
  sdk_header "${sdk_header}")
set(sdk_overlay "${CMAKE_CURRENT_BINARY_DIR}/sdk-extension")
file(MAKE_DIRECTORY "${sdk_overlay}/aws/s3-crt")
file(CONFIGURE OUTPUT "${sdk_overlay}/aws/s3-crt/S3CrtClient.h"
  CONTENT "${sdk_header}" @ONLY)
# All targets in this build see the same extended class declaration.
include_directories(BEFORE "${sdk_overlay}")
