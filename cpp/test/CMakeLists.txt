find_package(GTest REQUIRED)

if (WITH_ASAN STREQUAL "True")
  set(CMAKE_CXX_FLAGS
    "-fno-stack-protector -fno-omit-frame-pointer -fno-var-tracking -g -fsanitize=address ${CMAKE_CXX_FLAGS}"
  )
endif()

file(GLOB_RECURSE BUSTUB_TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*.cpp")

add_executable(milvus_test ${BUSTUB_TEST_SOURCES})

target_include_directories(milvus_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(milvus_test PROPERTIES
  CXX_STANDARD 17
)

target_link_libraries(
  milvus_test PRIVATE milvus-storage GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(milvus_test)
