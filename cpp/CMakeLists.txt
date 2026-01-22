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
option(WITH_LANCE "Build with lance." OFF)
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

set(BUILD_RUST_BRIDGE OFF)
if(WITH_VORTEX OR WITH_LANCE)
    set(BUILD_RUST_BRIDGE ON)
endif()

if (BUILD_RUST_BRIDGE)
  if (WITH_VORTEX)
    add_definitions(-DBUILD_VORTEX_BRIDGE)
  endif() # WITH_VORTEX

  if (WITH_LANCE)
    add_definitions(-DBUILD_LANCE_BRIDGE)
  endif() # WITH_LANCE

  set(RUST_BRIDGE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/format/bridge/rust)
  add_subdirectory(${RUST_BRIDGE_PATH})

  list(APPEND LINK_LIBS prsbridge)
  list(APPEND INCLUDE_PATHS ${RUST_BRIDGE_PATH}/include/)
endif()  # BUILD_RUST_BRIDGE

file(GLOB_RECURSE ALL_SRC_FILES
    src/*.cpp
    src/*.cc
)
list(FILTER ALL_SRC_FILES EXCLUDE REGEX ".*/src/format/bridge/.*\\.(cpp|cc)$")
list(FILTER ALL_SRC_FILES EXCLUDE REGEX ".*/src/jni/.*\\.cpp$")

add_library(milvus-storage SHARED ${ALL_SRC_FILES})

list(APPEND LINK_LIBS arrow::arrow Boost::boost protobuf::protobuf AWS::aws-sdk-cpp-identity-management google-cloud-cpp::storage libavrocpp::libavrocpp Folly::folly fmt::fmt-header-only)
list(APPEND INCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(milvus-storage PUBLIC ${LINK_LIBS})
target_include_directories(milvus-storage PUBLIC ${INCLUDE_PATHS})
target_compile_options(milvus-storage PRIVATE -fPIC)

# Set RPATH to look for dependencies in the same directory ($ORIGIN)
if (NOT APPLE)
  set_target_properties(milvus-storage PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
endif()

# JNI bindings: use lazy binding to avoid TLS allocation issues
if (WITH_JNI AND NOT APPLE)
  target_link_options(milvus-storage PRIVATE
    -Wl,-z,lazy
  )
endif()

# Python bindings: hide all symbols except FFI symbols
if (WITH_PYTHON_BINDING)
  target_compile_options(milvus-storage PRIVATE -fvisibility=hidden)
  # Use version script on Linux to export only FFI symbols
  target_link_options(milvus-storage PRIVATE
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/ffi_exports.map
  )
endif()

if (WITH_UT)
  # Enable code coverage instrumentation when ut is enabled
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")

  add_definitions(-DBUILD_GTEST)
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

  add_dependencies(milvus-storage-jni milvus-storage)

  # Set RPATH to look for dependencies in the same directory ($ORIGIN)
  if(NOT APPLE)
    set_target_properties(milvus-storage-jni PROPERTIES
      BUILD_RPATH "$ORIGIN"
      INSTALL_RPATH "$ORIGIN"
    )
  endif()

  # link options and copy libraries
  if(NOT APPLE)
    target_link_options(milvus-storage-jni PRIVATE
      -Wl,--no-as-needed
      -Wl,-z,lazy
    )

    set(JAVA_DIR ${CMAKE_SOURCE_DIR}/../java/)
    add_custom_command(TARGET milvus-storage-jni POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:milvus-storage-jni> ${JAVA_DIR}
      COMMAND ${CMAKE_COMMAND}
        -DTARGET_FILE=$<TARGET_FILE:milvus-storage>
        -DJNI_FILE=$<TARGET_FILE:milvus-storage-jni>
        -DOUTPUT_DIR=${CMAKE_BINARY_DIR}/lib
        -DCONAN_LIB_DIRS="${CMAKE_PREFIX_PATH}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CopyDependencies.cmake
      COMMENT "Copying libraries and dependencies to ${CMAKE_BINARY_DIR}/lib"
    )
  endif()
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