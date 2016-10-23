# Tensorflow-cl

Run Tensorflow on OpenCL™ devices.  UNDER CONSTRUCTION!!!

## Summary

This repo was created from the original Tensorflow repository at:

- https://github.com/tensorflow/tensorflow

Please see the main repository for full Tensorflow documentation.  This readme will only focus on the OpenCL porting aspects of Tensorflow.

## What works

### What's working
- per-element binary operators: `sub`, `mul`, `div`, `not_equal`, `minimum`, `maximum`, `pow`, `squared_difference` (test: [test_tf3.py](tensorflow/stream_executor/cl/test/test_tf3.py))
- per-element unary operator: `tanh`, `abs`, `acos`, `asin`, `atan`, `ceil`, `cos`, `exp`, `floor`, `inverse`, `isfinite`, `isinf`, `isnan`, `log`, `neg`, `sigmoid`, `sign`, `sin`, `sqrt`, square`, `tan` (test: [test_tf4.py](tensorflow/stream_executor/cl/test/test_tf4.py))
- comparison operators: `equal_to`, `greater`, `greater_equal`, `less`, `less_equal`

### To do
- add BLAS
- convolutions
- Variables
- gradients

## Build

### Pre-requisites

- Ubuntu 16.04 64-bit (might work on other platforms, but not tested)
  - I hope to target also Mac, and you can help me to tweak some of the `BUILD` rules for Mac if you want (specifically [this one](https://github.com/hughperkins/tensorflow-cl/blob/tensorflow-cl/tensorflow/workspace.bzl#L21-L25), used by [usr_lib_x8664linux.BUILD](https://github.com/hughperkins/tensorflow-cl/blob/tensorflow-cl/usr_lib_x8664linux.BUILD))
- normal non-GPU tensorflow prerequisites for building from source
  - when you run `./configure`, you can put `n` for cuda, gpu etc
- you need an OpenCL-enabled GPU installed and OpenCL drivers for that GPU installed.  Currently, supported OpenCL version is 1.2 or better
  - To check this: run `clinfo`, and check you have at least one device with:
    - `Device Type`: 'GPU', and
    - `Device OpenCL C Version`: 1.2, or higher
  - If you do, then you're good :+1:

### Installation:

```
sudo apt-get install -y opencl-headers cmake clang-3.8 llvm-3.8 clinfo git gcc g++ python3-numpy python3-dev python3-wheel zlib1g-dev
sudo apt-get install -y git gcc g++ python3-numpy python3-dev python3-wheel zlib1g-dev virtualenv swig python3-setuptools
sudo apt-get install -y openjdk-8-jdk unzip zip
mkdir -p ~/git
cd ~/git

# install bazel
git clone https://github.com/bazelbuild/bazel.git
cd bazel
git checkout 0.3.2
./compile.sh
sudo cp output/bazel /usr/local/bin

# download tensorflow, and configure
git clone --recursive https://github.com/hughperkins/tensorflow-cl
cd tensorflow-cl
./configure
# put python path: /usr/bin/python3
# 'no' for hadoop, gpu, cloud, etc

# build cuda-on-cl
pushd third_party/cuda-on-cl
make -j 4
sudo make install
popd

# build tensorflow
source ~/env3/bin/activate
bazel run --verbose_failures --logging 6 //tensorflow/tools/pip_package:build_pip_package
# (ignore error message about 'No destination dir provided')
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflowpkg
if [[ ! -d ~/env3 ]]; then { virtualenv -p python3 ~/env3; } fi
pip install --upgrade /tmp/tensorflowpkg/tensorflow-0.11.0rc0-py3-none-any.whl

# test/validate
cd
python -c 'import tensorflow'
# hopefully no errors :-)
python ~/git/tensorflow-cl/tensorflow/stream_executor/cl/test/test_tf2.py
```

### Updating:

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

## Run

- test per-element binary operations [tensorflow/stream_executor/cl/test/test_tf3.py](tensorflow/stream_executor/cl/test/test_tf3.py):
```
cd
source ~/env3/bin/activate
python ~/git/tensorflow-cl/tensorflow/stream_executor/cl/test/test_tf3.py
```
- test per-element unary operations [tensorflow/stream_executor/cl/test/test_tf4.py](tensorflow/stream_executor/cl/test/test_tf4.py):
```
cd
source ~/env3/bin/activate
python ~/git/tensorflow-cl/tensorflow/stream_executor/cl/test/test_tf4.py
```

## Design/architecture

- tensorflow code stays 100% [NVIDIA® CUDA™](https://www.nvidia.com/object/cuda_home_new.html)
- [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl) compiles the CUDA code into OpenCL

## Roadmap

- use [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl) to port the bulk of tensorflow
- use im2col for convolution (for now)
- use [CLBlast](https://github.com/CNugteren/CLBlast) to provide blas implementation

## Related projects

### DNN Libraries
- [OpenCL Torch](https://github.com/hughperkins/distro-cl)
- [DeepCL](https://github.com/hughperkins/DeepCL)

### OpenCL middleware
- [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl)
- [EasyCL](https://github.com/hughperkins/EasyCL)

## News

- Oct 23:
  - can use component wise addition from Python now :-)
  - fixed critical bug involving `float4`s, that meant that tensors larger than, say, 3 :-P, could not be added correctly
  - added following per-element binary operators: `sub`, `mul`, `div`, `not_equal`, `minimum`, `maximum`, `pow`, `squared_difference` (test: [test_tf3.py](tensorflow/stream_executor/cl/test/test_tf3.py))
  - added following per-element unary operator: `tanh`, `abs`, `acos`, `asin`, `atan`, `ceil`, `cos`, `exp`, `floor`, `inverse`, `isfinite`, `isinf`, `isnan`, `log`, `neg`, `sigmoid`, `sign`, `sin`, `sqrt`, square`, `tan` (test: [test_tf4.py](tensorflow/stream_executor/cl/test/test_tf4.py))
  - added following comparison operators: `equal_to`, `greater`, `greater_equal`, `less`, `less_equal`
- Oct 22:
  - componentwise addition working, when called from c++
  - commit `0db9cc2e`: re-enabled `-fPIC`, `-pie`
    - this is a pre-requisite for being able to run from python at some point
    - but if you built prior to this, you need to deeeeep clean, and rebuild from scratch:
    ```
    rm -Rf third_party/cuda-on-cl/build
    bazel clean --expunge
    ```
  - python working (as of commit 5e67304c3c)
    - you'll need to do `bazel clean`, and rebuild from scratch, if you already did a build prior to this commit
- Oct 20:
  - removed requirement for CUDA Toolkit
  - updated build slightly: added https://github.com/hughperkins/cuda-on-cl as a submodule
- Oct 18:
  - stream executor up
  - crosstool working
