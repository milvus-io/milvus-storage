include(ExternalProject)

ExternalProject_Add(
    arrow_ep
    GIT_REPOSITORY https://github.com/apache/arrow.git
    GIT_TAG 69118b2d26f2fbbbe65ad30dbe167d74b70fe791
    CMAKE_ARGS 
        -DARROW_PARQUET=ON
        -DARROW_DATASET=ON 
        -DARROW_FILESYSTEM=ON
    SOURCE_SUBDIR cpp
    )
    
message(STATUS ${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-prefix/src/arrow_ep-build/release/libparquet.so)
add_library(arrow SHARED IMPORTED)
set_target_properties( arrow
    PROPERTIES
    IMPORTED_LOCATION   ${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-prefix/src/arrow_ep-build/release/libarrow.so)

add_library(parquet SHARED IMPORTED)
set_target_properties( parquet
    PROPERTIES
    IMPORTED_LOCATION   ${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-prefix/src/arrow_ep-build/release/libparquet.so)


add_dependencies(arrow arrow_ep)
add_dependencies(parquet arrow_ep)