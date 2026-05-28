
find_package(GTest REQUIRED)

if (WITH_ASAN STREQUAL "True")
  set(CMAKE_CXX_FLAGS
    "-fno-stack-protector -fno-omit-frame-pointer -fno-var-tracking -g -fsanitize=address ${CMAKE_CXX_FLAGS}"
  )
  set(CMAKE_C_FLAGS
    "-fno-stack-protector -fno-omit-frame-pointer -fno-var-tracking -g -fsanitize=address ${CMAKE_C_FLAGS}"
  )
endif()

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall")

file(GLOB_RECURSE BUSTUB_TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*.cpp")

add_executable(milvus_test ${BUSTUB_TEST_SOURCES})

target_include_directories(milvus_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(milvus_test PROPERTIES
  CXX_STANDARD 17
)
# Force the linker to emit RPATH instead of RUNPATH (Linux only).
# RUNPATH only applies to direct dependencies, so when milvus_test loads
# libmilvus-storage.so -> libfolly.so -> libgflags_nothreads.so, the search
# paths in milvus_test's RUNPATH are NOT used to locate libgflags.
# RPATH propagates to the entire transitive dependency chain, which is
# required for gtest_discover_tests to run the binary at build time
# without setting LD_LIBRARY_PATH.
# This became necessary after folly/gflags switched to shared libraries
# in commit bd46999 ("adjust conan dependencies for folly shared").
if (NOT APPLE)
  target_link_options(milvus_test PRIVATE "LINKER:--disable-new-dtags")
endif()


target_link_libraries(
  milvus_test PRIVATE milvus-storage GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(milvus_test DISCOVERY_TIMEOUT 120)

#CTest for the FFI
enable_testing()
file(GLOB_RECURSE FFI_TEST_FILES ffi/*.c)
add_executable(Test_FFI ${FFI_TEST_FILES})

# no need link arrow
target_link_libraries(Test_FFI PRIVATE milvus-storage)
target_include_directories(Test_FFI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(Test_FFI PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
)
