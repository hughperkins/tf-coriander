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

## Test results

Latest results, [v0.18.3](https://github.com/hughperkins/tf-coriander/releases/tag/v0.18.3):

| test | Mac Sierra, using Radeon Pro 450 GPU (thank you [ASAPP](http://asapp.com) :-) ) | Ubuntu 16.04, using NVIDIA K520 |
|----- |-------|-------|
| unit tests (`py.test -v`) | All pass :-) | All pass :-) |
| [linear_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/2_BasicModels/linear_regression.py) |  Runs ok, loss decreases  | Runs ok, loss decreases |
| [logistic_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/2_BasicModels/logistic_regression.py) | Runs ok, loss decreases  | Runs ok, loss decreases |
| [nearest_neighbor.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/2_BasicModels/nearest_neighbor.py) | Runs ok, accuracy 0.92  | Runs ok, accuracy 0.92 |
| [autoencoder.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/autoencoder.py)| Runs ok, loss decreases | Runs ok, loss decreases |
| [multilayer_perceptron.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/multilayer_perceptron.py) | Runs ok, loss decreases | Runs ok, loss decreases |
| [recurrent_network.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/recurrent_network.py) | Runs ok, loss decreases | Runs ok, loss decreases |
| [bidirectional_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/bidirectional_rnn.py) | Runs ok, loss decreases | Runs ok, loss decreases |
| [dynamic_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/3_NeuralNetworks/dynamic_rnn.py) | Runs ok, loss decreases | Runs ok, loss decreases |

## Installation 

- You will need:
  - the tensorflow non-gpu installation pre-requisites
  - an OpenCL 1.2-enabled GPU, and OpenCL 1.2-enabled drivers (check that `clinfo` shows your GPU, and that is shows as a GPU device)
  - python 3

For Ubuntu 16.04 and Mac Sierra, there are pre-built wheels available:

- [Mac Sierra](https://github.com/hughperkins/tf-coriander/releases/download/v0.18.3/tensorflow-cl-v0.18.3-macsierra-python3.zip)
- [Ubuntu 16.04](https://github.com/hughperkins/tf-coriander/releases/download/v0.18.3/tensorflow-cl-v0.18.3-ubuntu1604-python3.zip)

Simply download, unzip, then install using `pip`:
```
pip install --upgrade tensorflow-0.11.0rc0-py3-none-any.whl
```

For other operating systems, please [build from source](doc/build-from-source.md)

By default, Tensorflow-cl will run using the first GPU available on your system. You can use the environment variable `CL_GPUOFFSET` to choose others:

- `export CL_GPUOFFSET=1` chooses the second GPU (ie, index 1)
- `export CL_GPUOFFSET=2` chooses the third GPU

## Testing

- [Testing](doc/testing.md)

## Design/architecture

- tensorflow code stays 100% [NVIDIA® CUDA™](https://www.nvidia.com/object/cuda_home_new.html)
- [Coriander](https://github.com/hughperkins/Coriander) compiles the NVIDIA® CUDA™ code into OpenCL
- Cedric Nugteren's [CLBlast](https://github.com/CNugteren/CLBlast) provides BLAS (matrix multiplications)

Presentation on [Coriander](https://github.com/hughperkins/Coriander) at this year's [IWOCL 2017](http://www.iwocl.org/iwocl-2017/conference-program/)

## Enabling new operations

- [Enabling operations](doc/enabling-operations.md).

## Related projects

### DNN Libraries
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
