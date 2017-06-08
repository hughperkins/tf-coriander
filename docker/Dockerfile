FROM ubuntu:16.04

ARG GIT_BRANCH=master

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake cmake-curses-gui git gcc g++ libc6-dev zlib1g-dev \
    libtinfo-dev \
    curl ca-certificates build-essential wget xz-utils \
    bash-completion \
    python3 python3-virtualenv swig python \
    openjdk-8-jdk python3-dev \
    ocl-icd-opencl-dev

RUN git clone --recursive https://github.com/hughperkins/tf-coriander -b ${GIT_BRANCH}

RUN cd tf-coriander && \
    mkdir soft && \
    cd soft && \
    wget http://releases.llvm.org/4.0.0/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz

RUN cd tf-coriander/soft && \
    tar -xf clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
    mv clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04 llvm-4.0

RUN cd tf-coriander/soft && \
    wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel_0.4.5-linux-x86_64.deb

RUN cd tf-coriander/soft && \
    dpkg -i bazel_0.4.5-linux-x86_64.deb

RUN cd tf-coriander && \
    python3 -m virtualenv -p python3 env3

RUN cd tf-coriander && \
    . env3/bin/activate && \
    pip install numpy

RUN cd tf-coriander && \
    . env3/bin/activate && \
    util/run_configure.sh

RUN cd tf-coriander && \
    . env3/bin/activate && \
    CLANG_HOME=$PWD/soft/llvm-4.0 util/build_coriander.sh

RUN cd tf-coriander && \
    . env3/bin/activate && \
    LOCALRESOURCES=" --local_resources 4000,2,1.0" util/build_u1604.sh
