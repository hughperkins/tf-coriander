#!/bin/bash

set -e
set -x

git submodule update --init --recursive

pushd third_party/coriander
if [[ ! -d build ]]; then {
    mkdir build
} fi
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j 8

if [[ $(uname) == Darwin ]]; then {
    make install
} else {
    sudo make install
} fi

popd
echo Installed coriander
