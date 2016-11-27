#!/bin/bash

set -x

cd ${BASEDIR}

# bash ${BASEDIR}/travis/download.sh bazelinstall /private/var/tmp/_bazel_travis
# if [[ ! -f ${BASEDIR}/tools/bazel.rc ]]; then {
cd ${BASEDIR}
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_CUDA=0
export PYTHON_BIN_PATH=/usr/local/bin/python3
echo /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages | util/python/python_config.sh --setup /Library/Frameworks/Python.framework/Versions/3.5/bin/python3
GEN_GIT_SOURCE=tensorflow/tools/git/gen_git_source.py
chmod a+x ${GEN_GIT_SOURCE}
SOURCE_BASE_DIR=${BASEDIR}
echo SOURCE_BASE_DIR ${SOURCE_BASE_DIR}
${PYTHON_BIN_PATH} ${GEN_GIT_SOURCE} --configure ${SOURCE_BASE_DIR}
ls tools
perl -pi -e "s,SO_SUFFIX = \".(so|dylib)\",SO_SUFFIX = \".dylib\",s" tensorflow/core/platform/default/build_config.bzl
bazel --batch fetch //tensorflow/...
# bash ${BASEDIR}/travis/upload.sh bazelinstall /private/var/tmp/_bazel_travis
# } fi
