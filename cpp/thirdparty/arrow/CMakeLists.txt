include(ExternalProject)

ExternalProject_Add(
    arrow
    GIT_REPOSITORY https://github.com/apache/arrow.git
    GIT_TAG 50fb5b0df84a3ccaa665965e36a6e757d1af98f1
    CMAKE_ARGS 
        -DARROW_PARQUET=ON
        -DARROW_DATASET=ON 
        -DARROW_FILESYSTEM=ON
    SOURCE_SUBDIR cpp
    )