#!/bin/bash

set -x

cd ${BASEDIR}

echo BASEDIR ${BASEDIR}
bash ${BASEDIR}/travis/download.sh swig ${BASEDIR}/soft/swig-3.0.10
if [[ ! -f ${BASEDIR}/soft/swig-3.0.10/swig ]]; then {
    cd ${BASEDIR}/soft
    wget http://kent.dl.sourceforge.net/project/swig/swig/swig-3.0.10/swig-3.0.10.tar.gz
    tar -xf swig-3.0.10.tar.gz
    ls
    cd swig-3.0.10
    ./configure
    make -j 8
    bash ${BASEDIR}/travis/upload.sh swig ${BASEDIR}/soft/swig-3.0.10
} fi
cd ${BASEDIR}/soft/swig-3.0.10
sudo make install
ls -lh /usr/local/bin/swig*
