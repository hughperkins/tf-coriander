#!/bin/bash

set -e
set -x

util/build_coriander.sh
CLANG_HOME=$PWD/soft/llvm-4.0 util/run_configure.sh
util/build_tf.sh
util/build_wheel.sh
