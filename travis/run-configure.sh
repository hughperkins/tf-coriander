#!/bin/bash

set -x

if [[ ! -f ${BASEDIR}/tools/bazel.rc ]]; then {
    cd ${BASEDIR}
    export TF_NEED_GCP=0
    export TF_NEED_HDFS=0
    export TF_NEED_CUDA=0
    export PYTHON_BIN_PATH=/usr/local/bin/python3
    echo /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages | util/python/python_config.sh --setup /usr/local/bin/python3
    GEN_GIT_SOURCE=tensorflow/tools/git/gen_git_source.py
    chmod a+x ${GEN_GIT_SOURCE}
    SOURCE_BASE_DIR=${BASEDIR}
    echo SOURCE_BASE_DIR ${SOURCE_BASE_DIR}
    ${PYTHON_BIN_PATH} ${GEN_GIT_SOURCE} --configure ${SOURCE_BASE_DIR}
    ls tools
    perl -pi -e "s,SO_SUFFIX = \".(so|dylib)\",SO_SUFFIX = \".dylib\",s" tensorflow/core/platform/default/build_config.bzl
    bazel fetch //tensorflow/...
    bash ${BASEDIR}/travis/upload.sh bazelcache /private/var/tmp/_bazel_travis
} fi
