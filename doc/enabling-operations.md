# Enabling Tensorflow operations

This document is mostly for developers/maintainers.  But, anyone can be a developer/maintainer.  It suffices to change a line or two, here or there, and submit a pull request :-)

## GPU Operations: Philosophy

Tensorflow is a big project. In theory, I could just point Coriander at it, and it will all build, and work perfectly :-)

In practice, when I looked at Tensorflow, back in October, I hadnt written Coriander yet, and I wanted to start off with one or two tiny nibbles,
rather than watching as the entire world failed to build in one go :-D

So, the way operations on the gpu work is:
- by default, Tensorflow 'guards' all GPU operations in an `#if GOOGLE_CUDA ... #endif` section
- I left `GOOGLE_CUDA` undefined, so initially all GPU operations were disabled
- then, I uncomment these bit by bit, leaving `GOOGLE_CUDA` forever undefined

## Enabling GPU operations

What this means:
- not defining `GOOGLE_CUDA` is why when we run `./configure`, we put 'No' for 'GPU': thats the bit that sets `GOOGLE_CUDA`, or not.  We want: not
- I have activated a bunch of operations, but not all
- to add new operations, in many cases it is almost sufficient just to find the relevant file, in `tensorflow/core/kernels`, and comment out the `#if GOOGLE_CUDA ... #endif` guards

## Types

Nuance:
- currently Coriander is targeted primarily at int32s, and single floats.  That's partly because this is mostly sufficient/good for machine learning. And partly to keep things simple initially. We can dabble in other types later
- sooo... when you uncomment stuff, you'll see stuff inside the sections like:
```
#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32); \
  DEFINE_GPU_SPECS_INDEX(T, int64);
```
or:
```
#if GOOGLE_CUDA
REGISTER3(SimpleBinaryOp, GPU, "TanhGrad", functor::tanh_grad, float,
          Eigen::half, double);
#endif
```

What we want to do is to change these to only register/instantiate the float32 and int32 types. All other types should be commented out for now.  So,
if/when you comment-out the `GOOGLE_CUDA` bit, you'll want to remove pretty much anything that's not a single float, or a 32-bit integer, eg:
```
#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32);
  // DEFINE_GPU_SPECS_INDEX(T, int64);
```
and
```
// #if GOOGLE_CUDA
// REGISTER3(SimpleBinaryOp, GPU, "TanhGrad", functor::tanh_grad, float,
//           Eigen::half, double);
REGISTER(SimpleBinaryOp, GPU, "TanhGrad", functor::tanh_grad, float);
// #endif
```

As you can see, this might need a small-ish amount of hacking, eg to change `REGISTER3` into `REGISTER` in this case, since we are only registering
one type (`float`), instead of three (`float`, `half`, `double`)

## Testing

Ideally, it would be good to create new unit-tests, in tensorflow/stream_executor/cl/test directory, for any new operations enabled. That's because
there's a realistic (> 10% chance) that the newly enabled operations have some bug in them that needs to be fixed.

If you add such a test:
- please create a pull request to add the test to main `tensorflow-cl` branch.  Writing tests takes a bunch of time, and this would be super helpful :-)
- if the tests are failing, please create a branch somewhere, with the failing test, and the uncommented operations on it, and point me at it, by creating a new issue. I'll want to know:
  - which operation(s) you've enabled
  - where is your test
  - how to run the test
  - expected test results
  - the results you are seeing
  - what operating system you are using
  - what hardware you are using (primarily: which GPUs? which GPU driver(s)?)
- enabling operations and testing them, reporting any bugs, and pull requesting any operations that are tested and seem to be working ok, would be super enormously useful :-)
