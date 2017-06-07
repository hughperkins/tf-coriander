#!/bin/bash

set -e
set -x

if [[ x$1 == x ]]; then {
   echo please give version number
   exit 1
} fi

version=$1
echo version: $version

git fetch
git stash -u
git checkout $version
git submodule update --init --recursive
pushd third_party/coriander/build
rm -Rf *
cmake -DCMAKE_BUILD_TYPE=Debug ..
nice make -j 8
make install
popd

if [[ ! -d ~/env3 ]]; then {
    python3 -m venv ~/env3
} fi

source ~/env3/bin/activate
cat <<EOF | ./configure






EOF

util/build_mac.sh
util/build_wheel.sh

echo 'done :-)'
