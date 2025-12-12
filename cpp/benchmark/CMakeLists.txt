
find_package(benchmark REQUIRED)

file(GLOB_RECURSE BUSTUB_BENCH_SOURCES "${PROJECT_SOURCE_DIR}/benchmark/*.cpp")

# Also include test helper implementation used by some benchmarks
list(APPEND BUSTUB_BENCH_SOURCES "${PROJECT_SOURCE_DIR}/test/test_env.cpp")

add_executable(
  benchmark
  ${BUSTUB_BENCH_SOURCES}
)

target_link_libraries(
  benchmark milvus-storage benchmark::benchmark
)

# for test_env.h 
target_include_directories(benchmark PUBLIC ${PROJECT_SOURCE_DIR}/test/include)
