#!/bin/bash

set +e

unameOut="$(uname -s)"

ROOT_DIR="$( cd -P "$( dirname "$SOURCE" )/.." && pwd )"

PKG_CONFIG_DIR="$ROOT_DIR/go/internal/core/output/lib/pkgconfig"

conan install milvus-storage/0.1.0@milvus/testing --install-folder $PKG_CONFIG_DIR -g pkg_config --build=missing

# Update PKG_CONFIG_PATH
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:$PKG_CONFIG_DIR"