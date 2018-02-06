#!/bin/bash

# Assumptions:
# - already have brew installed
# - already have xcode and command-line tools installed
# - running from root of the already cloned tf-coriander repo


if [ -x '/opt/local/bin/port' ]; then
  echo "Getting dependencies with MacPorts."
  sudo port install autoconf automake libtool gflags python36
  PYTHON=python3.6
elif [ -x '/usr/local/bin/brew' ]; then
  echo "Getting dependencies with Brew."
  brew install autoconf automake libtool shtool gflags python3
  PYTHON=python3
else
  echo "Could not find MacPorts or Brew in standard locations, aborting."
fi


set -e
set -x

if [[ ! -d env3 ]]; then {
    $PYTHON -m venv env3
} fi

source env3/bin/activate
pip install -r util/requirements.txt

if [[ ! -d soft ]]; then {
    mkdir soft
} fi

set +e
bazelversion=none
bazelversion=$(bazel --batch version | grep 'Build label')
set -e

if [[ ${bazelversion} != 'Build label: 0.4.5' ]]; then {
    pushd soft
    if [[ ! -f bazel-0.4.5-installer-darwin-x86_64.sh ]]; then {
        wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-installer-darwin-x86_64.sh -O tmp-bazel-0.4.5-installer-darwin-x86_64.sh
        mv tmp-bazel-0.4.5-installer-darwin-x86_64.sh bazel-0.4.5-installer-darwin-x86_64.sh
    } fi
    sh ./bazel-0.4.5-installer-darwin-x86_64.sh --user
    popd
} fi

pushd soft
if [[ ! -d llvm-4.0 ]]; then {
    if [[ ! -f clang+llvm-4.0.0-x86_64-apple-darwin.tar.xz ]]; then {
        wget http://llvm.org/releases/4.0.0/clang+llvm-4.0.0-x86_64-apple-darwin.tar.xz -O tmp-clang+llvm-4.0.0-x86_64-apple-darwin.tar.xz
        mv tmp-clang+llvm-4.0.0-x86_64-apple-darwin.tar.xz clang+llvm-4.0.0-x86_64-apple-darwin.tar.xz
    } fi
    rm -Rf clang+llvm-4.0.0-x86_64-apple-darwin
    tar -xf clang+llvm-4.0.0-x86_64-apple-darwin.tar.xz
    mv clang+llvm-4.0.0-x86_64-apple-darwin llvm-4.0
} fi
popd
