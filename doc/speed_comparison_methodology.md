# Speed comparison methodology

The scripts by Aymeric Damien, at [Tensorflow-Examples](https://github.com/aymericdamien/TensorFlow-Examples) were instrumented, to measure per epoch/iteration times.

It was then assumed that:
- the first epoch/iteration time would be longer, because of kernel build/compilation time
- subsequent epochs/iterations should have approximately similar execution times

Therefore, if we have iteration times `t0`, `t1`, `t2`, in a list `iteration_times`, we calculate the average iteration time as:
```
average_time = numpy.average(iteration_times[1:])
```
Then, the iteration time for the first iteration is assumed to be approximately:
```
t0 ~= kernel_compilation_time + average_iteration_time
```
...and therefore we approximate the kernel_compilation_time as:
```
kernel_compilation_time = t0 - average_iteration_time
```
