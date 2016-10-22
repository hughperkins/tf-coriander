# Tensorflow-cl

Run Tensorflow on OpenCL™ devices.  UNDER CONSTRUCTION!!!

## Summary

This repo was created from the original Tensorflow repository at:

- https://github.com/tensorflow/tensorflow

Please see the main repository for full Tensorflow documentation.  This readme will only focus on the OpenCL porting aspects of Tensorflow.

## What works

- OpenCL stream executor up:

<img src="doc/img/contextcreated.png?raw=true" width="600" />

- crosstool working:

<img src="doc/img/testcu.png?raw=true" width="600" />

- component-wise addition working:

<img src="doc/img/componentwiseaddition_tensorflow_opencl.png?raw=true" width="600"/>

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

### Initial setup:
```
sudo apt-get install opencl-headers cmake clang-3.8 llvm-3.8 clinfo
git submodule update --init
pushd third_party/cuda-on-cl
make -j 4
sudo make install
popd
```

### Updating:

- if you pull down new updates from the `tensorflow-cl` repository, please run the following, to update the [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl) installation:
```
git submodule update
pushd third_party/cuda-on-cl
make -j 4
sudo make install
popd
```
- note that you dont need to re-run configure.  Yay! :-)

## Run

### Stream executor test

Stream executor test: [tensorflow/stream_executor/cl/test/test.cc](https://github.com/hughperkins/tensorflow-cl/blob/tensorflow-cl/tensorflow/stream_executor/cl/test/test.cc) :
```
bazel run --verbose_failures //tensorflow/stream_executor:test_cl
```

### Crosstool test

Crosstool test: [tensorflow/tools/cocl/test/testcu.cu.cc](https://github.com/hughperkins/tensorflow-cl/blob/tensorflow-cl/tensorflow/tools/cocl/test/testcu.cu.cc) :
```
bazel run --verbose_failures //tensorflow/tools/cocl:testcu
```

### Component-wise addition

- uses protobuffers created using [tensorflow/stream_executor/cl/test/create_proto.py](tensorflow/stream_executor/cl/test/create_proto.py)
- copy to `/tmp` by doing:
```
cp tensorflow/stream_exeuctor/cl/test/graph.pb /tmp
```
- then, run [tensorflow/stream_executor/cl/test/loadproto.cc](tensorflow/stream_executor/cl/test/loadproto.cc), by doing:
```
bazel run --verbose_failures --logging 6 //tensorflow/stream_executor:test_cl_loadproto
```

## Design/architecture

- tensorflow code stays 100% [NVIDIA® CUDA™](https://www.nvidia.com/object/cuda_home_new.html)
- [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl) compiles the CUDA code into OpenCL

## Roadmap

- use [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl) to port the bulk of tensorflow
- use im2col for convolution (for now)
- use [CLBlast](https://github.com/CNugteren/CLBlast) to provide blas implementation

## FAQ

- Why fork the repo?  Why not make a standalone repo, that contains only the opencl bits, and uses tensorflow as a dependency?
  - well, I might.  In the future.  My bazel knowledge is a bit elementary for now :-P
- How can I see your contributions compared to all the mainstream Tensorflow stuff?
  - [this link](https://github.com/hughperkins/tensorflow-cl/compare/master...tensorflow-cl#files_bucket)

## Related projects

### DNN Libraries
- [OpenCL Torch](https://github.com/hughperkins/distro-cl)
- [DeepCL](https://github.com/hughperkins/DeepCL)

### OpenCL middleware

- [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl)
- [EasyCL](https://github.com/hughperkins/EasyCL)

## News

- Oct 22:
  - componentwise addition working, when called from c++
  - commit `0db9cc2e`: re-enabled `-fPIC`, `-pie`
    - this is a pre-requisite for being able to run from python at some point
    - but if you built prior to this, you need to deeeeep clean, and rebuild from scratch:
    ```
    rm -Rf third_party/cuda-on-cl/build
    bazel clean --expunge
    ```
- Oct 20:
  - removed requirement for CUDA Toolkit
  - updated build slightly: added https://github.com/hughperkins/cuda-on-cl as a submodule
- Oct 18:
  - stream executor up
  - crosstool working
