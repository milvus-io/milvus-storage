enable_testing()

add_executable(
  space_test
  space_test.cpp
)

target_link_libraries(
  space_test
  storage
  arrow
  parquet
  format
  GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(space_test)
