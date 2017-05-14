#!/bin/bash

# set -x

CACHE_NAME=$1
CACHED_PATH=$2

echo =============
echo DOWNLOAD ${CACHE_NAME} ${CACHED_PATH}

PATH=~/Library/Python/2.7/bin:$PATH
S3_CACHE_DIR=s3://${TRAVIS_BUCKET}/cache/tensorflow-cl/${TRAVIS_BRANCH}

set -x
time aws s3 cp --quiet ${S3_CACHE_DIR}/${CACHE_NAME}.tar.gz /tmp/${CACHE_NAME}.tar.gz
set +x
echo FOUND cache for ${CACHE_NAME}, decompressing...
mkdir -p ${CACHED_PATH}
cd ${CACHED_PATH}
set -x
time tar -xf /tmp/${CACHE_NAME}.tar.gz
echo DOWNLOAD DONE ${CACHE_NAME}
echo ==========
