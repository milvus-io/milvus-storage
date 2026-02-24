find_package(benchmark REQUIRED)

# Core benchmark sources (always included)
set(BENCHMARK_SOURCES
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_main.cpp
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_wr.cpp
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_footer_size.cpp
)

# Format Layer benchmarks (format availability checked at runtime)
list(APPEND BENCHMARK_SOURCES
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_format_write.cpp
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_format_read.cpp
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_v2_v3.cpp
)

# Storage Layer benchmark
list(APPEND BENCHMARK_SOURCES
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_storage_layer.cpp
)

# Data loader for configurable data sources (synthetic or Milvus segment)
list(APPEND BENCHMARK_SOURCES
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_data_loader.cpp
)

# Include test helper implementation used by benchmarks
list(APPEND BENCHMARK_SOURCES "${PROJECT_SOURCE_DIR}/test/test_env.cpp")

add_executable(
  benchmark
  ${BENCHMARK_SOURCES}
)

target_link_libraries(
  benchmark milvus-storage benchmark::benchmark
)

# Include directories for benchmark headers and test helpers
target_include_directories(benchmark PUBLIC
  ${PROJECT_SOURCE_DIR}/benchmark
  ${PROJECT_SOURCE_DIR}/test/include
)
