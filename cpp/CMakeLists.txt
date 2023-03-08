cmake_minimum_required(VERSION 3.0.0)
project(milvus-storage VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CTest)
enable_testing()

add_subdirectory(thirdparty)
add_library(milvus-storage milvus-storage.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
message(STATUS "PROJECT_NAME: ${PROJECT_TARGET_DIR}")
include(CPack)

target_link_libraries(milvus-storage libarrow_dataset.a)