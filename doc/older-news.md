# Older news

- Nov 1:
  - building clew, CLBlast, easycl, cocl as shared libraries now, rather than static
    - hopefully this will facilitate debugging things on the HD5500 on my laptop, since dont need to build/install entire wheel, for `libcocl` tweaks
  - turned on `clew`
    - this means no longer needs `libOpenCL.so` during build process
    - might facilitiate building on Mac, since no longer need to link to `libOpenCL.so`, which was outside the Bazel build tree
- Oct 30:
  - new wheel [v0.11.0](https://github.com/hughperkins/tensorflow-cl/releases/download/v0.11.0/tensorflow-0.11.0rc0-py3-none-any.whl)
    - fixes critical bug in v0.10.0 release, where the number of devices was hard-coded to be 0 :-P
    - Aymeric Damien's 2_BasicModels all run now, on NVIDIA K520.  Seem broken on Intel HD5500 for now
    - bunch of fixes underneath to get 2_BasicModels working ok on K520
- Oct 29:
  - `reduce_min` working now, and [test_reductions.py](tensorflow/stream_executor/cl/test/test_reductions.py) tests three types of reduction axes: inner, outer, all
  - Wheel [v0.10.0](https://github.com/hughperkins/tensorflow-cl/releases/download/v0.10.0/tensorflow-0.11.0rc0-py3-none-any.whl) released:
     - Aymeric Damien's [linear_regression](https://github.com/hughperkins/TensorFlow-Examples/blob/enforce-gpu/examples/2_BasicModels/linear_regression.py) runs fairly ok now (a bit slow, but not monstrously slow, maybe 3-4 times slower than on CUDA)
     - kernels cached between kernel launches (this gives a hugggeee speed boost, compared to earlier)
     - bunch of behind-the-scenes ops added, like Cast
     - memory and device name reported correctly now
     - `reduce_min` working now
     - `softmax` added
- Oct 28:
  - training working :-)  [test_gradients.py](tensorflow/stream_executor/cl/test/test_gradients.py)
  - `reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_mean` added, in beta [test_reductions.py](tensorflow/stream_executor/cl/test/test_reductions.py)
- Oct 25:
  - fixed BLAS wrapper, working now, on GPU, test script: [test_blas.py](tensorflow/stream_executor/cl/test/test_blas.py)
  - int32 constant works on gpu now, [test_ints.py](tensorflow/stream_executor/cl/test/test_ints.py)
- Oct 24:
  - hmmm, just discovered some new options, to ensure operations really are on the gpu, and ... many are not :-P, so back to the drawing board a bit
    - the good news is that component-wise add really is on the gpu
    - the bad news is that everything else is not :-P
  - (re-)added following per-element binary operators: `sub`, `mul`, `div`, `pow`, `minimum`, `maximum`, `squared_difference`.  This time, they actually are really running on the gpu :-)  (test: [test_tf3.py](tensorflow/stream_executor/cl/test/test_tf3.py))
  - (re-)added following per-element unary operators:, which really are running on gpu now :-), [test_tf4.py](tensorflow/stream_executor/cl/test/test_tf4.py): `tanh`, `abs`, `acos`, `asin`, `atan`, `ceil`, `cos`, `exp`, `floor`, `inverse`, `isfinite`, `isinf`, `isnan`, `log`, `neg`, `sign`, `sin`, `sqrt`, square`, `tan`
  - Variables can be placed on gpu now, [test_gradients.py](tensorflow/stream_executor/cl/test/test_gradients.py)
- Oct 23:
  - can use component wise addition from Python now :-)
  - fixed critical bug involving `float4`s, that meant that tensors larger than, say, 3 :-P, could not be added correctly
  - ~~added following per-element binary operators: `sub`, `mul`, `div`, `not_equal`, `minimum`, `maximum`, `pow`, `squared_difference` (test: [test_tf3.py](tensorflow/stream_executor/cl/test/test_tf3.py))~~
  - ~~added following per-element unary operator: `tanh`, `abs`, `acos`, `asin`, `atan`, `ceil`, `cos`, `exp`, `floor`, `inverse`, `isfinite`, `isinf`, `isnan`, `log`, `neg`, `sigmoid`, `sign`, `sin`, `sqrt`, square`, `tan` (test: [test_tf4.py](tensorflow/stream_executor/cl/test/test_tf4.py))~~
  - ~~added following comparison operators: `equal_to`, `greater`, `greater_equal`, `less`, `less_equal`~~
  - ~~added in BLAS (using Cedric Nugteren's [CLBlast](https://github.com/CNugteren/CLBlast) ).  Not very tested yet.  Test script [test_blas.py](tensorflow/stream_executor/cl/test/test_blas.py)~~
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
  