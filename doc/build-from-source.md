# Building from source

## Pre-requisites

### Ubuntu 16.04 64-bit:

- normal non-GPU tensorflow prerequisites for building from source
  - when you run `./configure`, you can put `n` for cuda, gpu etc
- you need an OpenCL-enabled GPU installed and OpenCL drivers for that GPU installed.  Currently, supported OpenCL version is 1.2 or better
  - To check this: run `clinfo`, and check you have at least one device with:
    - `Device Type`: 'GPU', and
    - `Device OpenCL C Version`: 1.2, or higher
  - If you do, then you're good :+1:
- install the following:
```
sudo apt-get install -y opencl-headers cmake clang-3.8 llvm-3.8 clinfo git gcc g++ python3-numpy python3-dev python3-wheel zlib1g-dev
sudo apt-get install -y git gcc g++ python3-numpy python3-dev python3-wheel zlib1g-dev virtualenv swig python3-setuptools
sudo apt-get install -y openjdk-8-jdk unzip zip

mkdir -p ~/git
cd ~/git
git clone https://github.com/bazelbuild/bazel.git
cd bazel
git checkout 0.3.2
./compile.sh
sudo cp output/bazel /usr/local/bin
```

### Mac Sierra (draft)

- normal Mac non-GPU tensorflow prerequisites for building from source
  - when you run `./configure`, you can put `n` for cuda, gpu etc
- also:
```
brew install bazel
brew install protobuf
brew install grpc
# the following brew installs might not be strictly necessary, but I did them:
brew install autoconf automake libtool shtool gflags
```

## Procedure

```
mkdir -p ~/git
cd ~/git

# prepare python virtual env
if [[ ! -d ~/env3 ]]; then { virtualenv -p python3 ~/env3; } fi
source ~/env3/bin/activate
pip install numpy
deactivate

# download tensorflow, and configure
git clone --recursive https://github.com/hughperkins/tensorflow-cl
cd tensorflow-cl
source ~/env3/bin/activate
./configure
# put python path: /usr/bin/python3
# 'no' for hadoop, gpu, cloud, etc

# build cuda-on-cl
pushd third_party/cuda-on-cl
mkdir build
cd build
cmake ..
make -j 4
# note: on Mac: following command might not need `sudo`:
sudo make install
popd

# on mac, do:
mkdir bazel-out/host/bin/external/grpc
ln -s /usr/local/bin/grpc_cpp_plugin bazel-out/host/bin/external/grpc/

# build tensorflow
bazel build --verbose_failures --logging 6 //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflowpkg
pip install --upgrade /tmp/tensorflowpkg/tensorflow-0.11.0rc0-py3-none-any.whl

# test/validate
cd
python -c 'import tensorflow'
# hopefully no errors :-)
# and the expected result:
# [[  4.   7.   9.]
# [  8.  10.  12.]]
python ~/git/tensorflow-cl/tensorflow/stream_executor/cl/test/test_simple.py
```

## Updating

- if you pull down new updates from the `tensorflow-cl` repository, you will almost certainly need to update the [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl) installation:
```
git submodule update
pushd third_party/cuda-on-cl
make -j 4
sudo make install
popd
```
- you will probably need to do also `bazel clean`
- and then, as before:
```
source ~/env3/bin/activate
bazel run --verbose_failures --logging 6 //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflowpkg
pip install --upgrade /tmp/tensorflowpkg/tensorflow-0.11.0rc0-py3-none-any.whl
```
