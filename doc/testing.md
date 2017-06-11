# Testing

There are two sets of tests:
- Unit tests, in [tensorflow/stream_executor/cl/test](tensorflow/stream_executor/cl/test)
- End to end tests, using a modified [Tensorflow-Examples](https://github.com/hughperkins/TensorFlow-Examples/tree/as-unit-tests)

## Unit tests


### Setup
```
pip install -r tensorflow/stream_executor/cl/test/requirements.txt
```

### Run

```
py.test -v
```

On a Mac:

<img src="img/mac_sierra_tests.png" width="600" />

## End to end tests

First clone the tests:
```
git clone https://github.com/hughperkins/TensorFlow-Examples/tree/as-unit-tests
```

Then to run them:
```
cd Tensorflow-Examples
bash ./run_tests.sh
```

## Results

Test results, using [wheel v0.18.3](https://github.com/hughperkins/tf-coriander/releases/tag/v0.18.3) :

| Test | Mac Sierra, using Radeon Pro 450 GPU (thank you [ASAPP](http://asapp.com) :-) ) | Ubuntu 16.04, using NVIDIA K520 |
|----- |-------|-------|
| Unit tests (`py.test -v`) | Pass | Pass |
| [linear_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/2_BasicModels/linear_regression.py) |  Ok | Ok |
| [logistic_regression.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/2_BasicModels/logistic_regression.py) | Ok | Ok |
| [nearest_neighbor.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/2_BasicModels/nearest_neighbor.py)  | Ok | Ok |
| [autoencoder.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/autoencoder.py)| Ok | Ok |
| [multilayer_perceptron.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/multilayer_perceptron.py)  | Ok | Ok |
| [recurrent_network.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/recurrent_network.py)  | Ok | Ok |
| [dynamic_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/dynamic_rnn.py)  | Ok | Ok |
| [bidirectional_rnn.py](https://github.com/hughperkins/TensorFlow-Examples/blob/as-unit-tests/examples/3_NeuralNetworks/bidirectional_rnn.py)  | Ok | Ok |
