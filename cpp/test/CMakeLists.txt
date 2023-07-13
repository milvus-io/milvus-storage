find_package(GTest REQUIRED)

add_executable(
  milvus_test
  options_test.cpp
  schema_test.cpp
  manifest_test.cpp
)

target_link_libraries(
  milvus_test storage GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(milvus_test)
