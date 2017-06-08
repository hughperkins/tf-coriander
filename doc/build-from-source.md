# Building from source

## Pre-requisites

- you need an OpenCL-enabled GPU installed and OpenCL drivers for that GPU installed.  Currently, supported OpenCL version is 1.2 or better
  - To check this: run `clinfo`, and check you have at least one device with:
    - `Device Type`: 'GPU', and
    - `Device OpenCL C Version`: 1.2, or higher
  - If you do, then you're good :+1:

- normal non-GPU tensorflow prerequisites for building from source
- then do:
```
git clone --recursive https://github.com/hughperkins/tf-coriander
cd tf-coriander
bash ./install_deps.sh
```

## Build

From the root of the cloned `tf-coriander` repo, do:
```
bash ./build.sh
```

## Install

```
pip install --upgrade soft/tensorflowpkg/tensorflow-0.11.0rc0-py3-none-any.whl
```

## Test/validate

```
source env3/bin/activate
( cd; python -c 'import tensorflow' )
# check no errors
python tensorflow/stream_executor/cl/test/test_simple.py
# hopefully no errors :-)
# expected result:
# [[  4.   7.   9.]
# [  8.  10.  12.]]
py.test -v
# hopefully no errors :-)
```

If you do get errors, please collect together as much inforamtion as possible, and save to a gist, then create an issue on the github.  I'll want:
  - the github commit of tf-coriander and coriander, that you are using
  - which operating system
  - which GPU(s)
  - the output of `clinfo`
  - the full output of the command that produced the error, as well as the command

## Updating

- if you pull down new updates from the `tf-coriander` repository, you need to update the [coriander](https://github.com/hughperkins/coriander) installation:
```
git submodule update --init --recursive
```
- .. and then redo the build

## Dockerfile

There is a dockerfile, based on Ubuntu 16.04, at [docker](../docker). Whilst this probalby wont run as such, it provides a useful reference to how to build,
possibly.
