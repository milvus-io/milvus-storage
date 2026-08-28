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

# Talon read-through cache benchmark (only when built with the Talon C SDK).
if (WITH_TALON)
  list(APPEND BENCHMARK_SOURCES
    ${PROJECT_SOURCE_DIR}/benchmark/benchmark_talon_reader.cpp
  )
endif()

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

# Keep every transitive dependency (notably libmilvus-common.so) in the
# benchmark's DT_NEEDED. The benchmark's own objects only reference
# libmilvus-storage.so symbols, so under the default --as-needed the loader
# reaches libmilvus-common.so only transitively and initializes it in an order
# that crashes its prometheus metric registration during static init, before
# main. milvus_test avoids this because its objects reference milvus-common
# symbols directly, keeping it a direct dependency. --no-as-needed reproduces
# that working DSO initialization order. (Adding the Talon C SDK to
# libmilvus-storage.so's DT_NEEDED is what shifted the order enough to expose
# the latent milvus-common static-init dependency.)
if (NOT APPLE)
  target_link_options(benchmark PRIVATE "LINKER:--no-as-needed")
endif()

# Include directories for benchmark headers and test helpers
target_include_directories(benchmark PUBLIC
  ${PROJECT_SOURCE_DIR}/benchmark
  ${PROJECT_SOURCE_DIR}/test/include
)
