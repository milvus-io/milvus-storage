function(build_glog)
    include(ExternalProject)
    set(GLOG_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/glog-ep)

    file(MAKE_DIRECTORY 
        ${GLOG_PREFIX}
        "${GLOG_PREFIX}/include"
        "${GLOG_PREFIX}/lib"
    )
    ExternalProject_Add(
        glog_ep
        GIT_REPOSITORY git@github.com:google/glog.git
        GIT_TAG v0.6.0
        CMAKE_ARGS 
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        INSTALL_DIR ${GLOG_PREFIX}
        )

    ExternalProject_Get_Property(glog_ep install_dir)

    add_library(libglog SHARED IMPORTED)
    set_target_properties(libglog
        PROPERTIES
        IMPORTED_GLOBAL TRUE
        IMPORTED_LOCATION   ${GLOG_PREFIX}/lib/libglog.so
        INTERFACE_INCLUDE_DIRECTORIES ${install_dir}/include/
        )

    add_dependencies(libglog glog_ep)
endfunction()

build_glog()