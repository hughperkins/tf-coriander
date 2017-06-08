#!/bin/bash

set -e
set -x

bazel-bin/tensorflow/tools/pip_package/build_pip_package $PWD/soft/tensorflowpkg
