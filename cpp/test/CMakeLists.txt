find_package(GTest REQUIRED)

file(GLOB_RECURSE BUSTUB_TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*.cpp")

add_executable(
  milvus_test
  ${BUSTUB_TEST_SOURCES}
)

target_link_libraries(
  milvus_test GTest::gtest_main milvus-storage
)

include(GoogleTest)
gtest_discover_tests(milvus_test)
