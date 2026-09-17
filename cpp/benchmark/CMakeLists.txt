find_package(benchmark REQUIRED)

# Core benchmark sources (always included)
set(BENCHMARK_SOURCES
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_main.cpp
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_wr.cpp
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_footer_size.cpp
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_compression_levels.cpp
)

# Format Layer benchmarks (format availability checked at runtime)
list(APPEND BENCHMARK_SOURCES
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_format_write.cpp
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_format_read.cpp
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_predicate.cpp
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_v2_v3.cpp
  ${PROJECT_SOURCE_DIR}/benchmark/benchmark_crt_reader.cpp
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

if (WITH_TALON)
  list(APPEND BENCHMARK_SOURCES
    ${PROJECT_SOURCE_DIR}/benchmark/benchmark_talon_reader.cpp
  )
endif()

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

# Small independent target so tracing overhead can be measured without building
# unrelated format benchmarks. The SDK/exporter is linked by the host only.
add_executable(tracing_benchmark benchmark_main.cpp benchmark_tracing.cpp)
target_link_libraries(tracing_benchmark PRIVATE milvus-storage benchmark::benchmark
  opentelemetry-cpp::opentelemetry_trace)
if (WITH_ASAN)
  target_compile_options(tracing_benchmark PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(tracing_benchmark PRIVATE -fsanitize=address)
endif()
if (NOT APPLE)
  target_link_options(tracing_benchmark PRIVATE "LINKER:--disable-new-dtags")
endif()

# Run the same end-to-end harness against pre-tracing and current libraries.
add_executable(telemetry_read_benchmark benchmark_telemetry_read.cpp)
target_link_libraries(telemetry_read_benchmark PRIVATE milvus-storage benchmark::benchmark
  opentelemetry-cpp::opentelemetry_trace)
target_compile_definitions(telemetry_read_benchmark PRIVATE STORAGE_TELEMETRY_BENCHMARK)
if (WITH_ASAN)
  target_compile_options(telemetry_read_benchmark PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(telemetry_read_benchmark PRIVATE -fsanitize=address)
endif()
if (NOT APPLE)
  target_link_options(telemetry_read_benchmark PRIVATE "LINKER:--disable-new-dtags")
endif()
