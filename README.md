# Tensorflow-cl

Run Tensorflow on OpenCL™ 1.2 devices

## Summary

This repo was created from the original Tensorflow repository at:

- https://github.com/tensorflow/tensorflow

Please see the main repository for full Tensorflow documentation.  This readme will only focus on the OpenCL porting aspects of Tensorflow.

## Good points of this compared to other OpenCL Tensorflows

- compatible with Mac, since doesnt need SPIR ingestor
- should work theoretically on any OpenCL 1.2 GPU device, without needing SPIR 1.2 extension, or OpenCL 2.0.

## What works, what doesnt

### What's working

- per-element unary operations
- per-element binary operations
- reductions
- BLAS, ie matrix multiplications, using Cedric Nugteren's [CLBlast](https://github.com/cnugteren/CLBlast) library
- `tf.random_normal`, `tf.random_uniform`
- trainers, including Adam

### What's missing

- `tf.split`
- Convolutions

## Test results, github, on [v0.17.2 wheel](https://github.com/hughperkins/tensorflow-cl/releases/tag/v0.17.2)

| test | Mac Sierra, using Radeon Pro 450 GPU (thank you [ASAPP](http://asapp.com) :-) ) | Ubuntu 16.04, using NVIDIA K520 |
|----- |-------|-------|
| unit tests (`py.test -v`) | All pass :-) | `tf.random_normal` fails. Others pass ok |
| [linear_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/2_BasicModels/linear_regression.py) |  Runs ok, loss decreases  | Runs ok, loss decreases |
| [logistic_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/2_BasicModels/logistic_regression.py) |  Runs ok, loss decreases | Runs ok, loss decreases |
| [nearest_neighbor.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/2_BasicModels/nearest_neighbor.py) |  Ok, accuracy 0.92 | Ok, accuracy 0.92 |
| [autoencoder.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/autoencoder.py)| Runs ok, loss decreases | Runs ok, loss decreases |
| [multilayer_perceptron.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/multilayer_perceptron.py) | Runs ok, loss decreases | Runs, but probably needs `tf.random_normal` working |
| [recurrent_network.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/recurrent_network.py)| Missing split | Missing split |
| [bidirectional_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/birectional_rnn.py)| Missing split | Missing split |
| [dynamic_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/dynamic_rnn.py) | Missing split, unpack | Missing split, unpack |
| [convolutional_network.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/convolutional_network.py) | Missing conv | Missing conv |

## Installation 

The environments used for testing/development are:
- Ubuntu 16.04, with:
  - NVIDIA K80, and
- Mac Sierra, with:
  - ~~Intel HD Graphics 530~~ I'm not testing on this currently, though I might, in the future
  - Radeon Pro 450
  (using a nice Mac Book Pro 4th generation that my employer [ASAPP](http://www.asapp.com) have provided me with recently :-) )

### Ubuntu 16.04

You can install from wheel:
- You will need:
  - the tensorflow non-gpu installation pre-requisites,
   - an OpenCL 1.2-enabled GPU, and  OpenCL 1.2-enabled drivers
   - python 3
- Simply download https://github.com/hughperkins/tensorflow-cl/releases/download/v0.17.2/tensorflow-cl-v0.17.2-ubuntu1604-python3.zip , and
- Install using pip:
```
unzip tensorflow-cl-v0.17.2-ubuntu1604-python3.zip
pip install --upgrade tensorflow-0.11.0rc0-py3-none-any.whl
```

### Mac Sierra

For Mac Sierra, python 3.6, there is a wheel at [https://github.com/hughperkins/tensorflow-cl/releases/download/v0.17.2/tensorflow-cl-v0.17.2-macsierra-python3.zip](https://github.com/hughperkins/tensorflow-cl/releases/download/v0.17.2/tensorflow-cl-v0.17.2-macsierra-python3.zip)
- tested on Mac Sierra, using Radeon Pro 450
- to select the Radeon, given that there's probably an Intel HD530 at gpu index 0, make sure to `export CL_GPUOFFSET=1`, which will select the gpu at index 1, ie the Radeon
- you'll need to install python 3.6, and create a virtualenv from it, activate it
- download the zip file from the link just above, and install by doing:
```
unzip tensorflow-cl-v0.17.2-macsierra-python3.zip
pip install --upgrade tensorflow-0.11.0rc0-py3-none-any.whl
```

Piccie of tests running on Mac Sierra:

<img src="doc/img/mac_sierra_tests.png" width="600" />

### Build from source

If you want, you can [build from source](doc/build-from-source.md)

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
- [Coriander](https://github.com/hughperkins/Coriander) compiles the CUDA code into OpenCL
- Cedric Nugteren's [CLBlast](https://github.com/CNugteren/CLBlast) provides BLAS (matrix multiplications)

## Hacking

If you want to enable new operations, please take a look at [enabling-operations.md](doc/enabling-operations.md).

## Related projects

### DNN Libraries
- [OpenCL Torch](https://github.com/hughperkins/distro-cl)
- [DeepCL](https://github.com/hughperkins/DeepCL)

### OpenCL middleware
- [CLBlast](https://github.com/CNugteren/CLBlast) BLAS for OpenCL
- [Coriander](https://github.com/hughperkins/coriander)  Compile NVIDIA® CUDA™ apps for OpenCL 1.2
- [EasyCL](https://github.com/hughperkins/EasyCL)   Handles running kernels, passing in arguments etc, on OpenCL

## News

- May 30 2017:
  - created [v0.17.2 release](https://github.com/hughperkins/tensorflow-cl/releases/tag/v0.17.2):
    - wheels available for both Ubuntu 16.04 and Mac Sierra, for Python 3.5
    - Aymeric Damien's [autoencoder.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/autoencoder.py) and [multilayer_perceptron.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/autoencoder.py) run ok now
    - `tf.random_normal` and `tf.random_uniform` working ok on Mac/Radeon
    - Adam works now
- May 27 2017:
  - upgraded LLVM, in Coriander, from 3.8.0 to 4.0.0. Thank you to @iame6162013 for inspiring me to do this
  - tons of operations are working now, on the github version:
    - `tf.random_normal` and `tf.random_uniform` work now
    - enabled a few operations like slicing, aggregation, concat, gather
