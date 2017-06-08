#!/bin/bash

set -e
set -x

sudo apt-get update && apt-get install -y --no-install-recommends \
    cmake cmake-curses-gui git gcc g++ libc6-dev zlib1g-dev \
    libtinfo-dev \
    curl ca-certificates build-essential wget xz-utils \
    bash-completion \
    python3 python3-virtualenv swig python \
    openjdk-8-jdk python3-dev \
    ocl-icd-opencl-dev

pushd soft
wget http://releases.llvm.org/4.0.0/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
tar -xf clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
mv clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04 llvm-4.0
wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel_0.4.5-linux-x86_64.deb
sudo dpkg -i bazel_0.4.5-linux-x86_64.deb
popd soft

python3 -m virtualenv -p python3 env3

. env3/bin/activate
pip install numpy
