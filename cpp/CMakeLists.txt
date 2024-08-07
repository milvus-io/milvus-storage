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

find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(Arrow REQUIRED)
find_package(protobuf REQUIRED)
find_package(glog REQUIRED)
find_package(Parquet REQUIRED)

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cc)
add_library(milvus-storage ${SRC_FILES})
target_include_directories(milvus-storage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/milvus-storage ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/test/include)
set(LINK_LIBS 
    Arrow::arrow_shared
    Parquet::parquet_shared
    protobuf::libprotobuf
    glog::glog)

if (WITH_OPENDAL)
  list(APPEND LINK_LIBS opendal)
endif()

target_link_libraries(milvus-storage PUBLIC 
  ${LINK_LIBS}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
)

if (WITH_UT)
  enable_testing()
  add_subdirectory(test)
endif()
