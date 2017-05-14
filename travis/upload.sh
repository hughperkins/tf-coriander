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
touch /tmp/${CACHE_NAME}.tar.gz
rm /tmp/${CACHE_NAME}.tar.gz
set -x
time tar -czf /tmp/${CACHE_NAME}.tar.gz ${EXCLUDE_STR} *
ls -lh /tmp/${CACHE_NAME}.tar.gz
time aws s3 cp --quiet /tmp/${CACHE_NAME}.tar.gz ${S3_CACHE_DIR}/${CACHE_NAME}.tar.gz
echo UPLOAD DONE ${CACHE_NAME}
echo ========================
