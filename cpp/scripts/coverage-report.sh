#!/bin/bash

# Configuration
BUILD_TYPE=${BUILD_TYPE:-Release}

# Detect LCOV version
LCOV_VERSION=$(lcov --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
LCOV_MAJOR=$(echo $LCOV_VERSION | cut -d. -f1)

# LCOV 2.0+ requires --ignore-errors for many common Clang/LLVM inconsistencies
if [ "$LCOV_MAJOR" -ge 2 ]; then
    LCOV_IGNORE_ERRORS="--ignore-errors mismatch,inconsistent,gcov,unused,deprecated,unsupported,format,count,category,empty,source"
else
    LCOV_IGNORE_ERRORS=""
fi

COVERAGE_BASE_DIR="coverage"
COVERAGE_RAW_INFO="${COVERAGE_BASE_DIR}/coverage.info"
COVERAGE_EXTRACTED_INFO="${COVERAGE_BASE_DIR}/coverage_extracted.info"
COVERAGE_FILTERED_INFO="${COVERAGE_BASE_DIR}/coverage_filtered.info"

echo "Generating coverage report..."
rm -rf ${COVERAGE_BASE_DIR}
mkdir -p ${COVERAGE_BASE_DIR}

# Capture coverage data (without branch coverage to avoid LLVM issues)
lcov --capture \
    --directory build/"${BUILD_TYPE}" \
    --output-file ${COVERAGE_RAW_INFO} \
    ${LCOV_IGNORE_ERRORS}

# Filter coverage data
# 1. Extract production code areas (broader patterns to handle different CI layouts)
lcov --extract ${COVERAGE_RAW_INFO} \
    '*/milvus-storage/cpp/src/*' \
    '*/milvus-storage/cpp/include/*' \
    --output-file ${COVERAGE_EXTRACTED_INFO} \
    ${LCOV_IGNORE_ERRORS}

# 2. Explicitly remove unwanted directories (tests, benchmarks, build artifacts, system/vcpkg headers)
lcov --remove ${COVERAGE_EXTRACTED_INFO} \
    "/usr/*" \
    "*/llvm/*" \
    "*/src/pb/*" \
    "*/src/core/bench/*" \
    "*/unittest/*" \
    "*/thirdparty/*" \
    "*/3rdparty_download/*" \
    "*/.conan/data/*" \
    --output-file ${COVERAGE_FILTERED_INFO} \
    ${LCOV_IGNORE_ERRORS}

# Generate HTML report (without branch coverage for LLVM compatibility)
genhtml ${COVERAGE_FILTERED_INFO} \
    --output-directory ${COVERAGE_BASE_DIR}/html \
    --title "Milvus Storage Coverage Report" \
    --legend \
    --highlight \
    ${LCOV_IGNORE_ERRORS}

echo ""
echo "====================================="
echo "Coverage report generated at: ${COVERAGE_BASE_DIR}/html/index.html"
echo "====================================="

# Print summary
lcov --summary ${COVERAGE_FILTERED_INFO} ${LCOV_IGNORE_ERRORS}