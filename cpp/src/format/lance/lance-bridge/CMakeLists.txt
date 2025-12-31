cmake_policy(SET CMP0135 NEW)

include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5
)
FetchContent_MakeAvailable(Corrosion)

set(RUST_SOURCE_FILE lib.rs)

corrosion_import_crate(
    MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
    FEATURES ${CORROSION_FEATURES}
    CRATES lance-bridge
)

corrosion_add_cxxbridge(lance-bridge CRATE lance_bridge FILES ${RUST_SOURCE_FILE})

target_include_directories(lance-bridge PUBLIC ${Arrow_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/lance-bridge/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

file(GLOB_RECURSE CPP_SOURCE_FILE CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

add_library(plancebridge STATIC ${CPP_SOURCE_FILE})

message(${CMAKE_CURRENT_SOURCE_DIR}/src/include)
message(STATUS "INCLUDED PATH: ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/lance-bridge/include")
message(STATUS "Arrow_INCLUDE_DIRS: ${Arrow_INCLUDE_DIRS}")

target_include_directories(plancebridge PUBLIC ${Arrow_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/lance-bridge/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

target_link_libraries(plancebridge PUBLIC lance-bridge)