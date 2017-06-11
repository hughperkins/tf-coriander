# Tensorflow-cl

Run [Tensorflow](https://github.com/tensorflow/tensorflow) on OpenCL™ 1.2 devices

- tested on:
  - Mac Sierra using Radeon Pro 450 GPU (thank you [ASAPP](http://asapp.com) :-) )
  - Ubuntu 16.04, using NVIDIA K520
- should work theoretically on any OpenCL 1.2 GPU

## Piccie

On a Mac:

<img src="doc/img/multilayerperceptron.png" />

## Execution speed

- [Execution speed](doc/execution_speed.md)

## What's working

- [What's working](doc/whats_working.md)

## Installation

- [Installation](doc/installation.md)

## Tests

- [Tests](doc/testing.md)

## Design/architecture

- tensorflow code stays 100% [NVIDIA® CUDA™](https://www.nvidia.com/object/cuda_home_new.html)
- [Coriander](https://github.com/hughperkins/Coriander) compiles the NVIDIA® CUDA™ code into OpenCL
- Cedric Nugteren's [CLBlast](https://github.com/CNugteren/CLBlast) provides BLAS (matrix multiplications)

Presentation on [Coriander](https://github.com/hughperkins/Coriander) at this year's [IWOCL 2017](http://www.iwocl.org/iwocl-2017/conference-program/)

## Related projects

- [OpenCL Torch](https://github.com/hughperkins/distro-cl)
- [DeepCL](https://github.com/hughperkins/DeepCL)

## News

- June 11 2017:
  - set up Jenkins build, which makes build logs and a Ubuntu 16.04 wheel available for certain commits, https://github.com/hughperkins/tf-coriander/commits/example-jenkins-builds (click on one of the green ticks)
- June 7 2017:
  - created [v0.18.3](https://github.com/hughperkins/tf-coriander/releases/tag/v0.18.3) release:
    - `tf.split` enabled
    - following examples from Aymeric Damien's [Tensorflow-Examples](https://github.com/aymericdamien/TensorFlow-Examples) run now:
      - [recurrent_network.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/recurrent_network.py)
      - [bidirectional_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/bidirectional_rnn.py)
      - [dynamic_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/dynamic_rnn.py)
