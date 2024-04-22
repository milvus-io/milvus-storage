enable_testing()

file(GLOB_RECURSE BUSTUB_TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*.cpp")

add_executable(
  milvus_test
  ${BUSTUB_TEST_SOURCES}
)

target_link_libraries(
  milvus_test milvus-storage GTest::gtest_main GTest::gmock_main 
)

include(GoogleTest)
gtest_discover_tests(milvus_test)
