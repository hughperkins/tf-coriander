#!/bin/bash

set -x

cd ${BASEDIR}

PYTHON_DIR=/Library/Frameworks/Python.framework/Versions/3.5

bash ${BASEDIR}/travis/download.sh pythoninstall ${PYTHON_DIR}

if [[ ! -f ${PYTHON_DIR}/bin/python3.5 ]]; then {
    cd ${BASEDIR}
    wget https://www.python.org/ftp/python/3.5.2/python-3.5.2-macosx10.6.pkg
    sudo installer -pkg python-3.5.2-macosx10.6.pkg -target /
    python3 -V
    which python3
    ls -lh /usr/local/bin/python*
    ls -lhd /usr/local/lib/py*
    find /usr/local -name 'dist-packages'
    pip3 install --upgrade setuptools
    pip3 install --upgrade wheel
    pip3 install --upgrade pip
    pip3 install numpy
    bash ${BASEDIR}/travis/upload.sh pythoninstall ${PYTHON_DIR}
} fi
