set( MILVUS_COMMON_VERSION a88c21f3d7faff36ca4af119c76b75a17afa3c86 )
set( GIT_REPOSITORY "https://github.com/zilliztech/milvus-common.git" )

message(STATUS "milvus-common repo: ${GIT_REPOSITORY}")
message(STATUS "milvus-common version: ${MILVUS_COMMON_VERSION}")
message(STATUS "Building milvus-common-${MILVUS_COMMON_VERSION} from source")

include( FetchContent )

FetchContent_Declare(
        milvus-common
        GIT_REPOSITORY  ${GIT_REPOSITORY}
        GIT_TAG         ${MILVUS_COMMON_VERSION}
        SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/milvus-common-src
        BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/milvus-common-build )

FetchContent_MakeAvailable( milvus-common )

# milvus-common's own CMakeLists never declares nlohmann_json even though
# src/common/Tracer.cpp #includes it unconditionally; without this declaration
# the source build resolves nlohmann/json.hpp from whatever system header is
# present instead of the pinned nlohmann_json in the Conan graph. Link it here
# so the embedded build compiles against the pinned package.
target_link_libraries(milvus-common PUBLIC nlohmann_json::nlohmann_json)

# libatomic is a separate library only on Linux (needed for 64-bit atomics on
# some architectures with GCC). On macOS, atomic operations are part of the
# standard library.
if(NOT APPLE)
    target_link_libraries(milvus-common PUBLIC atomic)
endif()

set( MILVUS_COMMON_INCLUDE_DIR ${milvus-common_SOURCE_DIR}/include CACHE INTERNAL "Path to milvus-common include directory" )

# The source-built target does not carry interface include directories (its own
# CMakeLists uses directory-level include_directories), but every consumer of
# the target needs them to resolve milvus-common's public headers. Dependency
# header paths (fmt, glog, ...) keep flowing through the target's PUBLIC
# linkage, matching what the Conan config exposed via transitive_headers.
target_include_directories(milvus-common INTERFACE ${MILVUS_COMMON_INCLUDE_DIR})

# milvus-common is no longer a Conan dependency, so the recipe's generate() can
# no longer copy its shared library into build/<type>/libs. The unittest job
# (LD_LIBRARY_PATH), the fat-jar packaging and the JNI RUNPATH check all source
# that directory, so keep a copy there. The milvus-common target is created in
# the FetchContent subdirectory, so a directory-scoped POST_BUILD command would
# fail here; a stamped custom command depending on the target does the copy.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/libs")
set( MILVUS_COMMON_LIBS_COPY_STAMP "${CMAKE_BINARY_DIR}/milvus-common-libs-copy.stamp" )
add_custom_command(
    OUTPUT "${MILVUS_COMMON_LIBS_COPY_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:milvus-common> "${CMAKE_BINARY_DIR}/libs/"
    COMMAND ${CMAKE_COMMAND} -E touch "${MILVUS_COMMON_LIBS_COPY_STAMP}"
    DEPENDS milvus-common
    VERBATIM )
add_custom_target(milvus-common-libs-copy ALL
    DEPENDS "${MILVUS_COMMON_LIBS_COPY_STAMP}" )

include_directories(SYSTEM ${MILVUS_COMMON_INCLUDE_DIR})
