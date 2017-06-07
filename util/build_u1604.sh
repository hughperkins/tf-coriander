#!/bin/bash

set -e
set -x

BUILDTYPE=
echo BUILDTYPE ${BUILDTYPE}
bazel build ${BUILDTYPE}  @grpc//:grpc_cpp_plugin
bazel build ${BUILDTYPE} @protobuf//:protoc
mkdir -p bazel-out/host/bin/external/grpc
mkdir -p bazel-out/host/bin/external/protobuf
ln -sf  $PWD/bazel-bin/external/protobuf/protoc bazel-out/host/bin/external/protobuf/protoc
ln -sf $PWD/bazel-bin/external/grpc/grpc_cpp_plugin bazel-out/host/bin/external/grpc/grpc_cpp_plugin
bazel build  ${BUILDTYPE} --verbose_failures --logging 6 //tensorflow/tools/pip_package:build_pip_package
