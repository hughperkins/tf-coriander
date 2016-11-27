#!/bin/bash

set -x

cd ${BASEDIR}

cd ${BASEDIR}/soft
java -version
wget https://github.com/bazelbuild/bazel/releases/download/0.3.2/bazel-0.3.2-installer-darwin-x86_64.sh
chmod +x bazel-0.3.2-installer-darwin-x86_64.sh
./bazel-0.3.2-installer-darwin-x86_64.sh --user
export PATH=$HOME/bin:$PATH
bazel --batch version
