#!/bin/bash

MAX_BUILD_TIME=1500

TARGET=$1
echo TARGET ${TARGET}

PATH=~/Library/Python/2.7/bin:$PATH
S3_CACHE_DIR=s3://${TRAVIS_BUCKET}/cache/tensorflow-cl/${TRAVIS_BRANCH}

bash ${BASEDIR}/travis/download.sh ${TRAVIS_BUILD_NUMBER}-${GIT_COMMIT} /private/var/tmp/_bazel_travis install

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
ps -ef | grep bazel | grep -v grep | grep -v run-bazel-step | awk '{print $2}' | xargs -L1 kill
# killall bazel
bazel --batch shutdown
while [ `ps -ef | grep bazel | grep -v grep | grep -v run-bazel-step | wc -l` -ne 0 ]; do {
    echo waiting for bazel shutdown...
    sleep 1
} done
echo bazel down
echo sleep 10
sleep 10

echo running upload...
bash ${BASEDIR}/travis/upload.sh ${TRAVIS_BUILD_NUMBER}-${GIT_COMMIT} /private/var/tmp/_bazel_travis install

touch /tmp/foo
aws s3 rm ${S3_CACHE_DIR}/${TRAVIS_BUILD_NUMBER}-${GIT_COMMIT}-running.flg || true
if [ ${BAZEL_DONE} -eq 1 ]; then {
    echo exit SUCCESS
    touch /tmp/flg.ok
    aws s3 cp /tmp/foo ${S3_CACHE_DIR}/${TRAVIS_BUILD_NUMBER}-${GIT_COMMIT}-ok.flg
    exit 0
} else {
    echo exit INCOMPLETE
    touch /tmp/flg.interrupted
    aws s3 cp /tmp/foo ${S3_CACHE_DIR}/${TRAVIS_BUILD_NUMBER}-${GIT_COMMIT}-interrupted.flg
    exit 1
} fi
