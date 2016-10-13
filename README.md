# Tensorflow-cl

Run Tensorflow on OpenCL devices.  UNDER CONSTRUCTION!!!

## Summary

This repo was created from the original Tensorflow repository at:

- https://github.com/tensorflow/tensorflow

Please see the main repository for full Tensorflow documentation.  This readme will only focus on the OpenCL porting aspects of Tensorflow.

## Work works

- opencl context created.  :-P

## Roadmap

- use [cuda-on-cl](https://github.com/hughperkins/cuda-on-cl) to port the bulk of tensorflow
- use im2col for convolution (for now)
- use [CLBlast](https://github.com/CNugteren/CLBlast) to provide blas implementation

## FAQ

- Why fork the repo?  Why not make a standalone repo, that contains only the opencl bits, and uses tensorflow as a dependency?
  - well, I might.  In the future.  My bazel knowledge is a bit elementary for now :-P
- How can I see your contributions compared to all the mainstream Tensorflow stuff?
  - [this link](https://github.com/hughperkins/tensorflow-cl/compare/master...tensorflow-cl#files_bucket)
