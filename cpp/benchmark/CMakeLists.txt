
find_package(benchmark REQUIRED)

file(GLOB_RECURSE BUSTUB_BENCH_SOURCES "${PROJECT_SOURCE_DIR}/benchmark/*.cpp")

add_executable(
  benchmark
  ${BUSTUB_BENCH_SOURCES}
)

target_link_libraries(
  benchmark milvus-storage benchmark::benchmark
)