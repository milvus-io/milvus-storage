cmake_minimum_required(VERSION 3.20.0)

project(milvus-storage VERSION 0.1.0)

option(WITH_UT "Build the testing tree." ON)
option(WITH_ASAN "Build with address sanitizer." OFF)
option(WITH_OPENDAL "Build with opendal." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (WITH_OPENDAL)
  add_compile_definitions(MILVUS_OPENDAL)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(libopendal)
endif()

add_subdirectory(thirdparty)

get_target_property(arrow_INCLUDE_DIR libarrow INCLUDE_DIRECTORIES)
get_target_property(parquet_INCLUDE_DIR libparquet INCLUDE_DIRECTORIES)
get_target_property(glog_INCLUDE_DIR libglog INCLUDE_DIRECTORIES)
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cc)
add_library(milvus-storage ${SRC_FILES})
add_dependencies(milvus-storage arrow_ep)
add_dependencies(milvus-storage glog_ep)
target_include_directories(milvus-storage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/milvus-storage ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/test/include ${arrow_INCLUDE_DIR} ${parquet_INCLUDE_DIR} ${glog_INCLUDE_DIR})
set(LINK_LIBS 
    libarrow
    libparquet
    Boost::uuid
    Boost::algorithm
    protobuf
    Boost::filesystem
    libglog
    )

if (WITH_OPENDAL)
  list(APPEND LINK_LIBS opendal)
endif()

target_link_libraries(milvus-storage PUBLIC ${LINK_LIBS})

if (WITH_UT)
  add_subdirectory(test)
endif()
