find_package(GTest REQUIRED)

add_executable(
  milvus_test
  options_test.cpp
  schema_test.cpp
)

target_link_libraries(
  milvus_test storage GTest::gtest_main
)
