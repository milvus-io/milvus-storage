cmake_policy(SET CMP0135 NEW)

include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5
)
FetchContent_MakeAvailable(Corrosion)


corrosion_import_crate(
    MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
    FEATURES ${CORROSION_FEATURES}
    CRATES rust-bridge
)

# corrosion_add_cxxbridge can only import a single static library; 
# otherwise, `rlib` will conflict.
corrosion_add_cxxbridge(rust-bridge
    CRATE rust_bridge
    FILES lib.rs)

target_include_directories(rust-bridge PUBLIC ${arrow_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/rust-bridge/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB_RECURSE CPP_SOURCE_FILE CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(prsbridge STATIC ${CPP_SOURCE_FILE})

target_include_directories(prsbridge PUBLIC ${arrow_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/rust-bridge/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(prsbridge PUBLIC rust-bridge)
