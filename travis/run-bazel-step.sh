#!/bin/bash

MAX_BUILD_TIME=1200

TARGET=$1
echo TARGET ${TARGET}

PATH=~/Library/Python/2.7/bin:$PATH
S3_CACHE_DIR=s3://${TRAVIS_BUCKET}/cache/tensorflow-cl/${TRAVIS_BRANCH}

bash ${BASEDIR}/travis/download.sh ${GIT_COMMIT} /private/var/tmp/_bazel_travis install

bazel --batch build --verbose_failures ${TARGET} &

BAZEL_DONE=0
function wait_bazel {
    while true; do {
        BAZEL_COUNT=`ps -ef | grep bazel | grep -v grep | grep -v run-bazel-step | wc -l`
        # echo BAZEL_COUNT ${BAZEL_COUNT}
        if [ ${BAZEL_COUNT} -eq 0 ]; then {
            echo bazel done
            BAZEL_DONE=1
            return
        } fi
        NOW=`python ${BASEDIR}/travis/seconds.py`
        BUILD_TIME=`echo "${NOW} - ${BUILD_START}" | bc -l`
        echo BUILD_TIME ${BUILD_TIME}
        if [ `echo "${BUILD_TIME} > ${MAX_BUILD_TIME}" | bc -l` -eq 1 ]; then {
            echo timeout
            return
        } fi
        # echo sleep
        sleep 10
    } done
}

wait_bazel
echo after wait_bazel
bazel --batch shutdown
while [ `ps -ef | grep bazel | grep -v grep | grep -v run-bazel-step | wc -l` -ne 0 ]; do {
    echo waiting for bazel shutdown...
    sleep 1
} done
echo bazel down

bash ${BASEDIR}/travis/upload.sh ${GIT_COMMIT} /private/var/tmp/_bazel_travis install

if [ ${BAZEL_DONE} -eq 1 ]; then {
    echo exit SUCCESS
    touch /tmp/flg.ok
    aws s3 cp /tmp/flg.interrupted ${S3_CACHE_DIR}/${GIT_COMMIT}-ok.flg
    exit 0
} else {
    echo exit FAIL
    touch /tmp/flg.interrupted
    aws s3 cp /tmp/flg.interrupted ${S3_CACHE_DIR}/${GIT_COMMIT}-interrupted.flg
    exit 1
} fi
