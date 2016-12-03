# Tensorflow-cl

Run Tensorflow on OpenCL™ 1.2 devices

## Summary

This repo was created from the original Tensorflow repository at:

- https://github.com/tensorflow/tensorflow

Please see the main repository for full Tensorflow documentation.  This readme will only focus on the OpenCL porting aspects of Tensorflow.

## Test results, on v0.14.0 wheel

| test | Intel HD5500, beignet 1.2.1 | NVIDIA 940M, driver v367.57 |
|----- |-------|-----|
| unit tests (`py.test -v`) | pass | pass |
| [linear_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/2_BasicModels/linear_regression.py) | slow, but works   | slow, but works   |
| [logistic_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/2_BasicModels/logistic_regression.py) | ok  | ok   |
| [nearest_neighbor.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/2_BasicModels/nearest_neighbor.py) | ok (accuracy 0.92)  | ok (accuracy 0.92)   |
| [multilayer_perceptron.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/multilayer_perceptron.py) | missing adam  | missing adam |
| [recurrent_network.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/recurrent_network.py)| missing adam   |  missing adam  |
| [autoencoder.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/autoencoder.py)|  missing rmsprop  |    |

## Installation 

- Ubuntu 16.04 is used for development
- Ubuntu 14.04 will probably work, but might need some small tweaking (which I can help with)
- Mac OS X builds ok :-)  Draft wheel (untested) is at https://s3.amazonaws.com/hughperkinstravis/cache/tensorflow-cl/travis/tensorflowpkg.tar.gz
- You need:
  - the tensorflow non-gpu installation pre-requisites,
   - an OpenCL 1.2-enabled GPU, and  OpenCL 1.2-enabled drivers
   - python 3
- Simply download https://github.com/hughperkins/tensorflow-cl/releases/download/v0.14.0/tensorflow-0.11.0rc0-py3-none-any.whl , and
- Install using pip:
```
pip install --upgrade tensorflow-0.11.0rc0-py3-none-any.whl
```

If you want, you can [build from source](doc/build-from-source.md)  [![Build Status](https://travis-ci.org/hughperkins/tensorflow-cl.svg?branch=travis)](https://travis-ci.org/hughperkins/tensorflow-cl/branches)

## Testing


### Setup

```
pip install -r tensorflow/stream_executor/cl/test/requirements.txt
```

### Run

```
py.test -v
```

## Design/architecture

- tensorflow code stays 100% [NVIDIA® CUDA™](https://www.nvidia.com/object/cuda_home_new.html)
- [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl) compiles the CUDA code into OpenCL
- Cedric Nugteren's [CLBlast](https://github.com/CNugteren/CLBlast) provides BLAS (matrix multiplications)

## Related projects

### DNN Libraries
- [OpenCL Torch](https://github.com/hughperkins/distro-cl)
- [DeepCL](https://github.com/hughperkins/DeepCL)

### OpenCL middleware
- [CLBlast](https://github.com/CNugteren/CLBlast) BLAS for OpenCL
- [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl)  Compile CUDA apps for OpenCL
- [EasyCL](https://github.com/hughperkins/EasyCL)   Handles running kernels, passing in arguments etc, on OpenCL

## News

- Dec 3:
  - BUILT A MAC WHEEL!!!  This is entirely untested.  But the wheel is here: https://s3.amazonaws.com/hughperkinstravis/cache/tensorflow-cl/travis/tensorflowpkg.tar.gz  (Simply untar it, and `pip install` it)
    - corresponding travis log is at https://travis-ci.org/hughperkins/tensorflow-cl/builds/180917138 and https://travis-ci.org/hughperkins/tensorflow-cl/builds/180410593
    - note that I had to built this in several stages, since it's a 3 hour build, and the logs for this are at https://s3.amazonaws.com/hughperkinstravis/cache/tensorflow-cl/travis/90-c520cc1-log.txt and https://s3.amazonaws.com/hughperkinstravis/cache/tensorflow-cl/travis/91-c55079d-log.txt
- Nov 29:
  - Mac build ran to completion!  On Travis.  Build output https://travis-ci.org/hughperkins/tensorflow-cl/builds/179727517  Yes, it didnt run, didnt create the wheel.  But the `build_pip_package` target built to completion.  which is a huge step forward :-)  Travis script here: [.travis.yml](.travis.yaml)
- Nov 25:
  - release wheel [v0.14.0](https://github.com/hughperkins/tensorflow-cl/releases/download/v0.14.0/tensorflow-0.11.0rc0-py3-none-any.whl)
    - this fixes `argmin`, `argmax`, and `softmax`
    - tons of changes under-the-hood
- Nov 10:
  - released wheel [v0.13.0](https://github.com/hughperkins/tensorflow-cl/releases/download/v0.13.0/tensorflow-0.11.0rc0-py3-none-any.whl)
     - beignet test results fairly solidly match K520 results now
     - fixed the regression on `not_equal` operator
     - removed the spam from memory copy  
- Nov 9:
  - fixed unary and binary operators on beignet
  - note that the tools/bazel.rc.templ has changed.  Please make sure to copy the new value into tools/bazel.rc, or re-run configure (probably need to do `bazel clean` anyway, so might as well do `./configure`)
