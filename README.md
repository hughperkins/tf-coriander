# Tensorflow-cl

Run [Tensorflow](https://github.com/tensorflow/tensorflow) on OpenCL™ 1.2 devices

- tested on Mac Sierra and Ubuntu 16.04
- should work theoretically on any OpenCL 1.2 GPU

## What's working

- unary operations, eg `tanh`, `log`, `exp`, `sigmoid`, `sqrt`, `abs`, `ceil`, `floor`, `argmax`, `argmin`
- binary operations, eg `pow`, `mul`, `add`, `maximum`, `minimum`, `squared_difference`
- reductions, eg `reduce_sum`, `reduce_max`, `reduce_min`, `reduce_prod`, `reduce_mean`
- variables, `tf.Variable`
- matrix multiplication, `matmul`
- random numbers, eg `tf.random_uniform`, `tf.random_normal`
- softmax, eg `tf.nn.softmax_cross_entropy_with_logits`
- gradients, automatic differentiation, back-propagation
- trainers, eg `tf.train.AdamOptimizer`
- ReLU, `tf.nn.relu`

## Piccie

On a Mac:

<img src="doc/img/multilayerperceptron.png" />

## Execution speed

- [Execution speed](doc/execution_speed.md)

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

- June 7 2017:
  - created [v0.18.3](https://github.com/hughperkins/tf-coriander/releases/tag/v0.18.3) release:
    - `tf.split` enabled
    - following examples from Aymeric Damien's [Tensorflow-Examples](https://github.com/aymericdamien/TensorFlow-Examples) run now:
      - [recurrent_network.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/recurrent_network.py)
      - [bidirectional_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/bidirectional_rnn.py)
      - [dynamic_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/dynamic_rnn.py)
  - (note that earlier v0.18.2 release had a regression)
- June 2 2017:
  - created [v0.17.3 release](https://github.com/hughperkins/tf-coriander/releases/tag/v0.17.3):
    - bug fix release:
      - `tf.random_uniform` and `tf.random_normal` should give equal results to the cpu version, on both Mac and Ubuntu
      - `tf.random_normal` should no longer give all zeros results on Ubuntu, ie should fix https://github.com/hughperkins/tf-coriander/issues/35
      - the Mac wheel should have `RPATH` set correctly, ie hopefully should not give error messages about unable to load `libclew.dylib` or similar, ie should fix https://github.com/hughperkins/tf-coriander/issues/39
