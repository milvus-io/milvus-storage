cmake_minimum_required(VERSION 3.20.0)

project(milvus-storage VERSION 0.1.0)

option(WITH_UT "Build the testing tree." ON)
option(WITH_ASAN "Build with address sanitizer." OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost REQUIRED)
find_package(arrow REQUIRED)
find_package(protobuf REQUIRED)

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cc)
message(STATUS "SRC_FILES: ${SRC_FILES}")
add_library(storage ${SRC_FILES})
target_include_directories(storage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(storage PUBLIC arrow::libarrow arrow::libparquet Boost::boost protobuf::protobuf)

if (WITH_UT)
  add_subdirectory(test)
endif()
