function(build_arrow)
    include(ExternalProject)
    set(ARROW_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/arrow-ep)

    file(MAKE_DIRECTORY 
        ${ARROW_PREFIX}
        "${ARROW_PREFIX}/include"
        "${ARROW_PREFIX}/lib"
    )
    ExternalProject_Add(
        arrow_ep
        GIT_REPOSITORY https://github.com/apache/arrow.git
        GIT_TAG 740889f
        CMAKE_ARGS 
            -DARROW_PARQUET=ON
            -DARROW_FILESYSTEM=ON
            -DARROW_S3=ON
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        SOURCE_SUBDIR cpp
        INSTALL_DIR ${ARROW_PREFIX}
        )

    ExternalProject_Get_Property(arrow_ep install_dir)

    message(STATUS ${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-prefix/src/arrow_ep-build/release/libparquet.so)
    add_library(libarrow SHARED IMPORTED)
    set_target_properties(libarrow
        PROPERTIES
        IMPORTED_GLOBAL TRUE
        IMPORTED_LOCATION   ${ARROW_PREFIX}/lib/libarrow.so
        INTERFACE_INCLUDE_DIRECTORIES ${install_dir}/include/
        )

    add_library(libparquet SHARED IMPORTED)
    set_target_properties(libparquet
        PROPERTIES 
        IMPORTED_GLOBAL TRUE
        IMPORTED_LOCATION   ${ARROW_PREFIX}/lib/libparquet.so
        INTERFACE_INCLUDE_DIRECTORIES ${install_dir}/include/
        )

    add_dependencies(libarrow arrow_ep)
    add_dependencies(libparquet arrow_ep)
endfunction()

build_arrow()
