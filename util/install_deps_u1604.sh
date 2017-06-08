#!/bin/bash

set -e
set -x

sudo apt-get update && apt-get install -y --no-install-recommends \
    cmake cmake-curses-gui git gcc g++ build-essential \
    libc6-dev zlib1g-dev libtinfo-dev \
    curl wget xz-utils unzip zip rsync \
    bash-completion \
    python3 python3-virtualenv swig python python3-dev \
    ca-certificates openjdk-8-jdk \
    ocl-icd-opencl-dev clinfo opencl-headers

pushd soft
wget http://releases.llvm.org/4.0.0/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
tar -xf clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
mv clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04 llvm-4.0
wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel_0.4.5-linux-x86_64.deb
sudo dpkg -i bazel_0.4.5-linux-x86_64.deb
popd soft

python3 -m virtualenv -p python3 env3

. env3/bin/activate
pip install -r util/requirements.txt
