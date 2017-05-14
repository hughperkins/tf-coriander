#!/bin/bash

set -x

cd ${BASEDIR}

PYTHON_DIR=/Library/Frameworks/Python.framework/Versions/3.5

bash ${BASEDIR}/travis/download.sh pythoninstall /tmp/pycopy
ls /tmp/pycopy
sudo mkdir -p ${PYTHON_DIR}
sudo rsync -a /tmp/pycopy/ ${PYTHON_DIR}/

wget https://www.python.org/ftp/python/3.5.2/python-3.5.2-macosx10.6.pkg
sudo installer -pkg python-3.5.2-macosx10.6.pkg -target /

NUMPY_COUNT=`pip3 freeze | grep numpy | wc -l`
echo NUMPY_COUNT [${NUMPY_COUNT}]

if [ ${NUMPY_COUNT} -eq 0 ]; then {
    echo installing numpy
    set -x
    cd ${BASEDIR}
    python3 -V
    which python3
    ls -lh /usr/local/bin/python*
    ls -lhd /usr/local/lib/py*
    find /usr/local -name 'dist-packages'
    pip3 install --upgrade setuptools
    pip3 install --upgrade wheel
    pip3 install --upgrade pip
    pip3 install numpy
    rsync -a ${PYTHON_DIR}/ /tmp/pycopy/
    bash ${BASEDIR}/travis/upload.sh pythoninstall /tmp/pycopy
} fi
