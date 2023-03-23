
find_package(GTest REQUIRED)

add_executable(
  space_test
  space_test.cpp
)

target_link_libraries(
  space_test
  arrow
  parquet
  storage
  GTest::gtest_main
)
