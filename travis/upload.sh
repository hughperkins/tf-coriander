#!/bin/bash

# set -x
set -e

CACHE_NAME=$1
CACHED_PATH=$2
EXCLUDES=$3
echo EXCLUDES ${EXCLUDES}
EXCLUDE_STR=
for EXCLUDE in ${EXCLUDES}; do {
    EXCLUDE_STR="${EXCLUDE_STR} --exclude ${EXCLUDE}"
} done
echo EXCLUDE_STR ${EXCLUDE_STR}

echo ===============
echo UPLOAD ${CACHE_NAME} ${CACHED_PATH}

PATH=~/Library/Python/2.7/bin:$PATH
S3_CACHE_DIR=s3://${TRAVIS_BUCKET}/cache/tensorflow-cl/${TRAVIS_BRANCH}

cd ${CACHED_PATH}
touch /tmp/${CACHE_NAME}.tar.bz2
rm /tmp/${CACHE_NAME}.tar.bz2
set -x
time tar -cjf /tmp/${CACHE_NAME}.tar.bz2 ${EXCLUDE_STR} *
ls -lh /tmp/${CACHE_NAME}.tar.bz2
time aws s3 cp --quiet /tmp/${CACHE_NAME}.tar.bz2 ${S3_CACHE_DIR}/${CACHE_NAME}.tar.bz2
echo UPLOAD DONE ${CACHE_NAME}
echo ========================
