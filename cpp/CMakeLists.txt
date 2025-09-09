cmake_minimum_required(VERSION 3.20.0)

project(milvus-storage VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WITH_UT "Build the testing tree." OFF)
option(WITH_ASAN "Build with address sanitizer." OFF)
option(WITH_OPENDAL "Build with opendal." OFF)
option(WITH_BENCHMARK "Build with micro benchmark." OFF)
option(WITH_AZURE_FS "Build with azure file system." ON)
option(WITH_VORTEX "Build with vortex" OFF)

if(WITH_VORTEX)
  include(ExternalProject)
  include(FetchContent)

  set(VORTEX_BUILD_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-vortex.sh")
  set(VORTEX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vortex-src)

  FetchContent_Declare(nanoarrow
    URL "https://www.apache.org/dyn/closer.lua?action=download&filename=arrow/apache-arrow-nanoarrow-0.7.0/apache-arrow-nanoarrow-0.7.0.tar.gz")
  FetchContent_MakeAvailable(nanoarrow)

  ExternalProject_Add(
    vortex_storage
    GIT_REPOSITORY "https://github.com/jiaqizho/vortex.git"
    GIT_TAG "develop"
    SOURCE_DIR ${VORTEX_SOURCE_DIR}
    GIT_SHALLOW 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${VORTEX_BUILD_SCRIPT} "${VORTEX_SOURCE_DIR}"
    BUILD_IN_SOURCE 0
    INSTALL_COMMAND ""
  )
endif()

# Disable Azure FS on macOS since the Arrow library is built without Azure support
if(APPLE)
  set(WITH_AZURE_FS OFF)
  message(STATUS "Azure filesystem disabled on macOS (Arrow built without Azure support)")
endif()
option(ARROW_WITH_JEMALLOC "Build with jemalloc." OFF)

include(GNUInstallDirs)

add_compile_definitions(GLOG_USE_GLOG_EXPORT) # fix glog build error

if (WITH_OPENDAL)
  add_compile_definitions(MILVUS_OPENDAL)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(libopendal)
endif()

if (WITH_AZURE_FS)
  add_compile_definitions(MILVUS_AZURE_FS)
endif()

find_package(Boost REQUIRED)

find_package(Arrow REQUIRED)
include_directories(${Arrow_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
find_package(glog REQUIRED)
find_package(google-cloud-cpp REQUIRED)

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cc)

add_library(milvus-storage SHARED ${SRC_FILES})

list(APPEND LINK_LIBS arrow::arrow)
list(APPEND LINK_LIBS Boost::boost)
list(APPEND LINK_LIBS protobuf::protobuf)
list(APPEND LINK_LIBS AWS::aws-sdk-cpp-identity-management)
list(APPEND LINK_LIBS glog::glog)
list(APPEND LINK_LIBS google-cloud-cpp::storage)

if (WITH_OPENDAL)
  list(APPEND LINK_LIBS opendal)
endif()

list(APPEND INLCUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include)
list(APPEND INLCUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (WITH_VORTEX) 
  message(${VORTEX_SOURCE_DIR})
  add_dependencies(milvus-storage vortex_storage)
  set(VORTEX_CXX_BUILD_PATH "${VORTEX_SOURCE_DIR}/vortex-cxx/build")

  list(APPEND LINK_LIBS "${VORTEX_CXX_BUILD_PATH}/libvortex.a" "${VORTEX_CXX_BUILD_PATH}/libvortex_cxx_bridge.a" "${VORTEX_CXX_BUILD_PATH}/libvortex_cxx.a")
  # for vortex header, vortex already link the nanoarrow_static, maybe direct include nanoarrow header?
  list(APPEND LINK_LIBS "nanoarrow_static")
  
  list(APPEND INLCUDE_PATHS ${VORTEX_SOURCE_DIR}/vortex-cxx/cpp/include)
  # only for debug...
  list(APPEND INLCUDE_PATHS ${VORTEX_CXX_BUILD_PATH}/corrosion_generated/cxxbridge/vortex_cxx_bridge/include/)
endif()

target_link_libraries(milvus-storage PUBLIC ${LINK_LIBS})
target_include_directories(milvus-storage PUBLIC ${INLCUDE_PATHS})

if (WITH_UT)
  add_subdirectory(test)
endif()

if (WITH_BENCHMARK)
  add_subdirectory(benchmark)
endif()

if (WITH_ASAN STREQUAL "True")
  set(CMAKE_CXX_FLAGS
    "-fno-stack-protector -fno-omit-frame-pointer -fno-var-tracking -g -fsanitize=address ${CMAKE_CXX_FLAGS}"
  )
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