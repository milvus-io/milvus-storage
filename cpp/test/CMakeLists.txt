enable_testing()

add_executable(
  space_test
  space_test.cpp
)

target_link_libraries(
  space_test
  arrow
  parquet
  reader
  filter
  storage
  format 
  common
  GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(space_test)
