#!/bin/bash
VORTEX_SOURCE_DIR=$1
VORTEX_BUILD_DIR="${VORTEX_SOURCE_DIR}/vortex-cxx/build"
LIB_FILE="${VORTEX_BUILD_DIR}/libvortex.a" # main static lib

if [ -f "$LIB_FILE" ]; then
    echo "libvortex.a exists, running make only."
    cd "${VORTEX_BUILD_DIR}" && make -j24
else
    echo "libvortex.a does not exist, running full build process."
    cd "${VORTEX_SOURCE_DIR}/vortex-cxx" && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug && \
    make -j24
fi