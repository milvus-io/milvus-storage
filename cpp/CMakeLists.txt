cmake_minimum_required(VERSION 3.20.0)

project(milvus-storage VERSION 0.1.0)

option(WITH_UT "Build the testing tree." ON)
option(WITH_ASAN "Build with address sanitizer." OFF)
option(WITH_OPENDAL "Build with opendal." OFF)
option(WITH_BENCHMARK "Build with micro benchmark." OFF)
option(WITH_AZURE_FS "Build with azure file system." ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
include(GNUInstallDirs)

if (WITH_OPENDAL)
  add_compile_definitions(MILVUS_OPENDAL)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(libopendal)
endif()

find_package(Boost REQUIRED)
find_package(Arrow REQUIRED)
find_package(Protobuf REQUIRED)
find_package(glog REQUIRED)

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cc)

add_library(milvus-storage SHARED ${SRC_FILES})
target_include_directories(milvus-storage PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/src 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/include
)

set(LINK_LIBS 
    arrow::arrow
    Boost::boost
    protobuf::protobuf
    AWS::aws-sdk-cpp-identity-management
    glog::glog
)

if (WITH_OPENDAL)
  list(APPEND LINK_LIBS opendal)
endif()

if (WITH_AZURE_FS)
  add_compile_definitions(MILVUS_AZURE_FS)
  list(APPEND LINK_LIBS Azure::azure-core)
  list(APPEND LINK_LIBS Azure::azure-storage-blobs)
endif()

target_link_libraries(milvus-storage PUBLIC ${LINK_LIBS})

set_target_properties(milvus-storage PROPERTIES 
    INSTALL_RPATH "$ORIGIN/../lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

if (WITH_UT)
  enable_testing()
  add_subdirectory(test)
endif()

if (WITH_BENCHMARK)
  add_subdirectory(benchmark)
endif()

function(add_pkg_config module)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/${module}.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/${module}.pc
    @ONLY
  )
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${module}.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/"
  )
endfunction()


add_pkg_config(libstorage)

install(TARGETS milvus-storage
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/milvus-storage"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_INCLUDE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/include)