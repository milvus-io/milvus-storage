#!/bin/bash

set +e

unameOut="$(uname -s)"

ROOT_DIR="$( cd -P "$( dirname "$SOURCE" )/.." && pwd )"

# Update PKG_CONFIG_PATH
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${ROOT_DIR}/cpp/build/Release/"