cmake_minimum_required(VERSION 3.20.0)

project(milvus-storage VERSION 0.1.0)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WITH_UT "Build the testing tree." OFF)
option(WITH_ASAN "Build with address sanitizer." OFF)
option(WITH_OPENDAL "Build with opendal." OFF)
option(WITH_BENCHMARK "Build with micro benchmark." OFF)
option(WITH_AZURE_FS "Build with azure file system." ON)
option(WITH_VORTEX "Build with vortex." OFF)
option(WITH_JNI "Build with JNI library." OFF)

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
find_package(nlohmann_json REQUIRED)

if (WITH_VORTEX) 
  add_definitions(-DBUILD_VORTEX_BRIDGE)
  set(VORTEX_BRIDGE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/format/vortex/vx-bridge/)

  set(VXBRIDGE_LIB_NAME libvx_bridge.so)
  if(APPLE)
    set(VXBRIDGE_LIB_NAME libvx_bridge.dylib)
  endif()

  add_subdirectory(${VORTEX_BRIDGE_PATH})

  # todo: change the debug path
  set_target_properties(vx_bridge PROPERTIES
    IMPORTED_LOCATION "${VORTEX_BRIDGE_PATH}/target/debug/${VXBRIDGE_LIB_NAME}"
  )

  list(APPEND LINK_LIBS vx_bridge pvxbridge) 
  list(APPEND INLCUDE_PATHS ${VORTEX_BRIDGE_PATH}/src/include/)
endif()  # WITH_VORTEX

file(GLOB_RECURSE ALL_SRC_FILES 
    src/*.cpp 
    src/*.cc
)
list(FILTER ALL_SRC_FILES EXCLUDE REGEX ".*/src/format/vortex/vx-bridge/.*\\.(cpp|cc)$")
list(FILTER ALL_SRC_FILES EXCLUDE REGEX ".*jni.*")

add_library(milvus-storage SHARED ${ALL_SRC_FILES})

if (WITH_OPENDAL)
  list(APPEND LINK_LIBS opendal)
endif()

list(APPEND LINK_LIBS arrow::arrow Boost::boost protobuf::protobuf AWS::aws-sdk-cpp-identity-management glog::glog google-cloud-cpp::storage nlohmann_json::nlohmann_json)
list(APPEND INLCUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(milvus-storage PUBLIC ${LINK_LIBS})
target_include_directories(milvus-storage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(milvus-storage PRIVATE -fPIC)
target_link_options(milvus-storage PRIVATE -Wl,-z,lazy)

if (WITH_UT)
  add_subdirectory(test)
endif()

if (WITH_BENCHMARK)
  add_subdirectory(benchmark)
endif()

if (WITH_JNI)
  if(NOT DEFINED ENV{JAVA_HOME})
    message(FATAL_ERROR "JNI enabled but JAVA_HOME not set")
  endif()

  set(JAVA_HOME $ENV{JAVA_HOME})
  set(JAVA_INCLUDE_PATH "${JAVA_HOME}/include")
  set(JAVA_INCLUDE_PATH2 "${JAVA_HOME}/include/linux")

  message(STATUS "JAVA_HOME: ${JAVA_HOME}")
  message(STATUS "JAVA_INCLUDE_PATH: ${JAVA_INCLUDE_PATH}")
  message(STATUS "JAVA_INCLUDE_PATH2: ${JAVA_INCLUDE_PATH2}")

  add_library(milvus-storage-jni SHARED
    src/ffi/reader_writer_jni.cpp
  )

  target_include_directories(milvus-storage-jni PRIVATE
    ${JAVA_INCLUDE_PATH}
    ${JAVA_INCLUDE_PATH2}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

  target_link_libraries(milvus-storage-jni PRIVATE
    milvus-storage
  )

  set_target_properties(milvus-storage-jni PROPERTIES
    CXX_STANDARD 17
    OUTPUT_NAME milvus_storage_jni
    POSITION_INDEPENDENT_CODE ON
  )

  target_link_options(milvus-storage-jni PRIVATE
    -Wl,--no-as-needed
    -Wl,-z,lazy
  )

  add_dependencies(milvus-storage-jni milvus-storage)
  set(SCALA_TEST_DIR ${CMAKE_SOURCE_DIR}/../scala-test/)
  set(TOP_LEVEL_OUTPUT_DIR ${SCALA_TEST_DIR})

  add_custom_command(TARGET milvus-storage-jni POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:milvus-storage-jni> ${TOP_LEVEL_OUTPUT_DIR}
    COMMENT "Copying generated .so to top level directory..."
  )
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