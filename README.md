# Tensorflow-cl

Run Tensorflow on OpenCL devices.  UNDER CONSTRUCTION!!!

## Summary

This repo was created from the original Tensorflow repository at:

- https://github.com/tensorflow/tensorflow

Please see the main repository for full Tensorflow documentation.  This readme will only focus on the OpenCL porting aspects of Tensorflow.

## What works

- OpenCL stream executor up:

<img src="doc/img/contextcreated.png?raw=true" width="600" height="400" />

- crosstool working:

<img src="doc/img/testcu.png?raw=true" width="600" height="170" />

## How to run

### Pre-requisites

- Ubuntu 16.04 64-bit (might work on other platforms, but not tested)
  - I hope to target also Mac, and you can help me to tweak some of the `BUILD` rules for Mac if you want (specifically [this one](https://github.com/hughperkins/tensorflow-cl/blob/tensorflow-cl/tensorflow/workspace.bzl#L21-L25), used by [usr_lib_x8664linux.BUILD](https://github.com/hughperkins/tensorflow-cl/blob/tensorflow-cl/usr_lib_x8664linux.BUILD))
- NVIDIA® CUDA™ toolkit 7.5, at `/usr/local/cuda-7.5`
- normal non-GPU tensorflow prerequisites for building from source
  - when you run `./configure`, you can put `n` for cuda, gpu etc
- following needs to be installed, in addition to standard tensorflow non-gpu pre-requisites:
```
sudo apt-get install opencl-headers cmake clang-3.8 llvm-3.8 clinfo
git clone --recursive https://github.com/hughperkins/cuda-on-cl
cd cuda-on-cl
make -j 4
sudo make install
```
- you need an OpenCL-enabled GPU installed and OpenCL drivers for that GPU installed.  Currently, supported OpenCL version is 1.2 or better
  - To check this: run `clinfo`, and check you have at least one device with:
    - `Device Type`: 'GPU', and
    - `Device OpenCL C Version`: 1.2, or higher
  - If you do, then you're good :+1:

### Procedure

#### Stream executor test

Stream executor test: [tensorflow/stream_executor/cl/test/test.cc](https://github.com/hughperkins/tensorflow-cl/blob/tensorflow-cl/tensorflow/stream_executor/cl/test/test.cc) :
```
bazel run --verbose_failures //tensorflow/stream_executor:test_cl
```

#### Crosstool test

Crosstool test: [tensorflow/tools/cocl/test/testcu.cu.cc](https://github.com/hughperkins/tensorflow-cl/blob/tensorflow-cl/tensorflow/tools/cocl/test/testcu.cu.cc) :
```
bazel run --verbose_failures //tensorflow/tools/cocl:testcu
```

## Roadmap

- use [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl) to port the bulk of tensorflow
- use im2col for convolution (for now)
- use [CLBlast](https://github.com/CNugteren/CLBlast) to provide blas implementation

## FAQ

- Why fork the repo?  Why not make a standalone repo, that contains only the opencl bits, and uses tensorflow as a dependency?
  - well, I might.  In the future.  My bazel knowledge is a bit elementary for now :-P
- How can I see your contributions compared to all the mainstream Tensorflow stuff?
  - [this link](https://github.com/hughperkins/tensorflow-cl/compare/master...tensorflow-cl#files_bucket)

## News

- Oct 18:
  - stream executor up
  - crosstool working
