#!/bin/bash

# builds python wheel, into soft/tensorflowpkg directory
#
# assumes you already ran installdeps.sh, and build.sh, and that you are currently in the root directory of the cloned
# tf-coriander repository

set -e
set -x

bazel-bin/tensorflow/tools/pip_package/build_pip_package $PWD/soft/tensorflowpkg
