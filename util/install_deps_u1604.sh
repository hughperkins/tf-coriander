#!/bin/bash

set -e
set -x

SUDO=sudo
if [[ ! $(cat /proc/1/sched | head -n 1 | grep '[init|systemd]') ]]; then {
    # running in docker
    echo running in docker
    SUDO=
} fi

${SUDO} apt-get update && ${SUDO} apt-get install -y --no-install-recommends \
    cmake cmake-curses-gui git gcc g++ build-essential \
    libc6-dev zlib1g-dev libtinfo-dev \
    curl wget xz-utils unzip zip rsync \
    bash-completion \
    python3 python3-virtualenv swig python python3-dev \
    ca-certificates openjdk-8-jdk \
    ocl-icd-opencl-dev clinfo opencl-headers

if [[ ! -d soft ]]; then {
    mkdir soft
} fi
pushd soft

if [[ ! -d llvm-4.0 ]]; then {
    wget http://releases.llvm.org/4.0.0/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz -O clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
    if [[ -d clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04 ]]; then {
        rm -R clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04
    } fi
    tar -xf clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
    mv clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04 llvm-4.0
} fi

set +e
bazelversion=none
bazelversion=$(bazel --batch version | grep 'Build label')
set -e

if [[ ${bazelversion} != 'Build label: 0.4.5' ]]; then {
    wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel_0.4.5-linux-x86_64.deb -O bazel_0.4.5-linux-x86_64.deb
    ${SUDO} dpkg -i bazel_0.4.5-linux-x86_64.deb
} fi

popd

if [[ ! -d env3 ]]; then {
    python3 -m virtualenv -p python3 env3
} fi

. env3/bin/activate
pip install -r util/requirements.txt
