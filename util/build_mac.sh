#!/bin/bash

set -e
set -x

# this means you can carry on doing other stuff whilst your Mac keeps your knees warm :-) :
LOCALRESOURCES=" --local_resources 8000,3,1.0"

nice bazel build ${LOCALRESOURCES} @grpc//:grpc_cpp_plugin
mkdir -p bazel-out/host/bin/external/grpc
ln -sf $PWD/bazel-bin/external/grpc/grpc_cpp_plugin bazel-out/host/bin/external/grpc/grpc_cpp_plugin
nice bazel build ${LOCALRESOURCES} --verbose_failures --logging 6 //tensorflow/tools/pip_package:build_pip_package
