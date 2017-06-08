#!/bin/bash

# assumes:
# - you've ready run install_deps.sh
# - you've already acitvated your python3 virtual environment

# it will build coriander, tensorflow, wheel
#
# wont install the wheel. wont install the pre-requisites
#
# the wheel will be created in soft/tensorflowpkg subidrecotry

set -e
set -x

. env3/bin/activate
CLANG_HOME=$PWD/soft/llvm-4.0 util/build_coriander.sh
util/run_configure.sh
util/build_tf.sh
util/build_wheel.sh
