function(build_opendal)
    include(ExternalProject)
    set(OPENDAL_NAME "libopendal_c${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(OPENDAL_PREFIX ${CMAKE_BINARY_DIR}/thirdparty/opendal_ep)

    file(MAKE_DIRECTORY
        "${OPENDAL_PREFIX}"
        "${OPENDAL_PREFIX}/lib"
        "${OPENDAL_PREFIX}/include"
    )

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(OPENDAL_BUILD_TYPE "debug")
    else()
        set(OPENDAL_BUILD_TYPE "release")
        set(OPENDAL_BUILD_OPTS "--release")
    endif()

    ExternalProject_Add(
        opendal_ep
        GIT_REPOSITORY      https://github.com/apache/incubator-opendal.git
        GIT_TAG             main
        PREFIX              ${OPENDAL_PREFIX}
        SOURCE_SUBDIR       bindings/c
        CONFIGURE_COMMAND   echo "configure for opendal_ep"
        BUILD_COMMAND       cargo build ${OPENDAL_BUILD_OPTS}
        BUILD_IN_SOURCE     1
        INSTALL_COMMAND     bash -c "cp ${OPENDAL_PREFIX}/src/opendal_ep/bindings/c/target/${OPENDAL_BUILD_TYPE}/${OPENDAL_NAME} ${OPENDAL_PREFIX}/lib/ && cp ${OPENDAL_PREFIX}/src/opendal_ep/bindings/c/include/opendal.h ${OPENDAL_PREFIX}/include/")


    add_library(opendal STATIC IMPORTED)
    set_target_properties(opendal
        PROPERTIES
        IMPORTED_GLOBAL TRUE
        IMPORTED_LOCATION "${OPENDAL_PREFIX}/lib/${OPENDAL_NAME}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENDAL_PREFIX}/include")
    add_dependencies(opendal opendal_ep)
    if(APPLE)
        target_link_libraries(opendal INTERFACE "-framework CoreFoundation")
        target_link_libraries(opendal INTERFACE "-framework Security")
        target_link_libraries(opendal INTERFACE "-framework SystemConfiguration")
    endif()

    get_target_property(OPENDAL_IMPORTED_LOCATION opendal IMPORTED_LOCATION)
    get_target_property(OPENDAL_INTERFACE_INCLUDE_DIRECTORIES opendal INTERFACE_INCLUDE_DIRECTORIES)
    message("OPENDAL_IMPORTED_LOCATION: ${OPENDAL_IMPORTED_LOCATION}")
    message("OPENDAL_INTERFACE_INCLUDE_DIRECTORIES: ${OPENDAL_INTERFACE_INCLUDE_DIRECTORIES}")
endfunction()

if (opendal_FOUND)
    message("Found opendal: ${opendal_INCLUDE_DIRS}")
else()
    build_opendal()
endif()


