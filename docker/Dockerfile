FROM ubuntu:16.04

ARG GIT_BRANCH=master

RUN git clone --recursive https://github.com/hughperkins/tf-coriander -b ${GIT_BRANCH}

RUN cd tf-coriander && \
    ./install_deps.sh

RUN cd tf-coriander && \
    LOCALRESOURCES=" --local_resources 4000,2,1.0" ./build.sh

RUN cd tf-coriander && \
    . env3/bin/activate && \
    pip install soft/tensorflowpkg/tensorflow-0.11.0rc0-py3-none-any.whl

RUN cd tf-coriander && \
    . env3/bin/activate && \
    cd && \
    python -c 'import tensorflow'
