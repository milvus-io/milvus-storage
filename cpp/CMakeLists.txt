cmake_minimum_required(VERSION 3.20.0)

project(milvus-storage VERSION 0.1.0)

option(WITH_UT "Build the testing tree." ON)
option(WITH_ASAN "Build with address sanitizer." OFF)
option(WITH_OPENDAL "Build with opendal." OFF)
option(WITH_BENCHMARK "Build with micro benchmark." ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
include(GNUInstallDirs)


if (WITH_OPENDAL)
  add_compile_definitions(MILVUS_OPENDAL)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(libopendal)
endif()

find_package(Azure REQUIRED)
find_package(Boost REQUIRED)
find_package(arrow REQUIRED)
find_package(protobuf REQUIRED)
find_package(glog REQUIRED)
find_package(AWSSDK REQUIRED)

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cc)
add_library(milvus-storage ${SRC_FILES})
target_include_directories(milvus-storage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/milvus-storage ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/test/include)
set(LINK_LIBS 
    arrow::arrow
    Boost::boost
    protobuf::protobuf
    AWS::aws-sdk-cpp-identity-management
    glog::glog
    Azure::azure-core
    Azure::azure-storage-blobs
)

if (WITH_OPENDAL)
  list(APPEND LINK_LIBS opendal)
endif()

target_link_libraries(milvus-storage PUBLIC ${LINK_LIBS})

if (WITH_UT)
  enable_testing()
  add_subdirectory(test)
endif()

if (WITH_BENCHMARK)
  add_subdirectory(benchmark)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/milvus-storage.pc.in "${CMAKE_CURRENT_BINARY_DIR}/milvus-storage.pc" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/milvus-storage.pc" DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")