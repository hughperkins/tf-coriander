# Execution speed

## Comparison with NVIDIA® CUDA™ native

We can run tests on the same GPU: once using NVIDIA® CUDA™ native, and once using Coriander, and compare the execution times

| Scenario | Coriander | NVIDIA® CUDA™ native |
|----- |-------|-------|
| [linear_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/2_BasicModels/linear_regression.py)  | 0.21s | 0.07s |
| [logistic_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/2_BasicModels/logistic_regression.py) | 9.5s | 3.7s |
| [multilayer_perceptron.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/multilayer_perceptron.py) | 15.8s | 15.1s |
| [recurrent_network.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/recurrent_network.py) | 0.84s | 0.23s |
| [dynamic_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/dynamic_rnn.py) | 0.9s | 0.23s |
| [bidirectional_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/bidirectional_rnn.py) | 0.9s | 0.24s |

We can see that:
- for multilayer_perceptron.py, epoch time is comparable between Coriander and NVIDIA® CUDA™, using the same GPU
- for the recurrent networks, Coriander is around 4 times slower than using NVIDIA® CUDA™ directly.

[methodology](speed_comparison_methodology.md)

## Kernel setup/compile time

Coriander writes the kernels to OpenCL at runtime, and compiles them on-the-fly.  This means the first iteration will take longer.  Here is the increase in execution time for the first iteration:

| Scenario | Kernel generation/compile time |
|----- |-------|
| [linear_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/2_BasicModels/linear_regression.py)  | 0.13s |
| [logistic_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/2_BasicModels/logistic_regression.py) | 0.9s |
| [multilayer_perceptron.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/multilayer_perceptron.py) | ~0s |
| [recurrent_network.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/recurrent_network.py) | 1.9s |
| [dynamic_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/dynamic_rnn.py) | 3.7s |
| [bidirectional_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/bidirectional_rnn.py) | 2.1s |

[methodology](speed_comparison_methodology.md)
