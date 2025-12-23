cmake_minimum_required(VERSION 3.20.0)

project(milvus-storage VERSION 0.1.0)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WITH_UT "Build the testing tree." OFF)
option(WITH_ASAN "Build with address sanitizer." OFF)
option(WITH_BENCHMARK "Build with micro benchmark." ON)
option(WITH_AZURE_FS "Build with azure file system." ON)
option(WITH_VORTEX "Build with vortex." OFF)
option(WITH_JNI "Build with JNI library." OFF)
option(WITH_PYTHON_BINDING "Build for Python bindings with hidden symbols." OFF)

# Disable Azure FS on macOS since the Arrow library is built without Azure support
if(APPLE)
  set(WITH_AZURE_FS OFF)
  message(STATUS "Azure filesystem disabled on macOS (Arrow built without Azure support)")
endif()
option(ARROW_WITH_JEMALLOC "Build with jemalloc." OFF)

include(GNUInstallDirs)

if (WITH_AZURE_FS)
  add_compile_definitions(MILVUS_AZURE_FS)
endif()


find_package(Boost REQUIRED)
find_package(fmt REQUIRED)
find_package(folly REQUIRED)
find_package(Arrow REQUIRED)
include_directories(${Arrow_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
find_package(google-cloud-cpp REQUIRED)
find_package(libavrocpp REQUIRED)

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
list(FILTER ALL_SRC_FILES EXCLUDE REGEX ".*/src/jni/.*\\.cpp$")

add_library(milvus-storage SHARED ${ALL_SRC_FILES})

list(APPEND LINK_LIBS arrow::arrow Boost::boost protobuf::protobuf AWS::aws-sdk-cpp-identity-management google-cloud-cpp::storage libavrocpp::libavrocpp Folly::folly fmt::fmt-header-only)
list(APPEND INLCUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(milvus-storage PUBLIC ${LINK_LIBS})
target_include_directories(milvus-storage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(milvus-storage PRIVATE -fPIC)

# Python bindings: hide all symbols except FFI symbols
if (WITH_PYTHON_BINDING)
  target_compile_options(milvus-storage PRIVATE -fvisibility=hidden)
  # Use version script on Linux to export only FFI symbols
  target_link_options(milvus-storage PRIVATE
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/ffi_exports.map
  )
endif()

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
  if (APPLE)
    set(JAVA_INCLUDE_PATH2 "${JAVA_HOME}/include/darwin")
  else()
    set(JAVA_INCLUDE_PATH2 "${JAVA_HOME}/include/linux")
  endif()

  file(GLOB JNI_SRC_FILES src/jni/*.cpp)
  add_library(milvus-storage-jni SHARED ${JNI_SRC_FILES})

  target_include_directories(milvus-storage-jni PRIVATE
    ${JAVA_INCLUDE_PATH}
    ${JAVA_INCLUDE_PATH2}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

  target_link_libraries(milvus-storage-jni PRIVATE
    milvus-storage
    ${LINK_LIBS}
  )

  if(NOT APPLE)
    target_link_options(milvus-storage-jni PRIVATE
      -Wl,--no-as-needed
      -Wl,-z,lazy
    )
  endif()

  add_dependencies(milvus-storage-jni milvus-storage)
  set(JAVA_DIR ${CMAKE_SOURCE_DIR}/../java/)
  set(TOP_LEVEL_OUTPUT_DIR ${JAVA_DIR})

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