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
    CRATES vx-bridge
)

corrosion_add_cxxbridge(vx-bridge CRATE vx_bridge FILES ${RUST_SOURCE_FILE})

target_include_directories(vx-bridge PUBLIC ${Arrow_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vx-bridge/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

file(GLOB_RECURSE CPP_SOURCE_FILE CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

add_library(pvxbridge STATIC ${CPP_SOURCE_FILE})

message(${CMAKE_CURRENT_SOURCE_DIR}/src/include)
message(STATUS "INCLUDED PATH: ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vx-bridge/include")
message(STATUS "Arrow_INCLUDE_DIRS: ${Arrow_INCLUDE_DIRS}")

target_include_directories(pvxbridge PUBLIC ${Arrow_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/vx-bridge/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

target_link_libraries(pvxbridge PUBLIC vx-bridge)