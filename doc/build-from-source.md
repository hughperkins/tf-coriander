# Building from source

## Pre-requisites

- you need an OpenCL-enabled GPU installed and OpenCL drivers for that GPU installed.  Currently, supported OpenCL version is 1.2 or better
  - To check this: run `clinfo`, and check you have at least one device with:
    - `Device Type`: 'GPU', and
    - `Device OpenCL C Version`: 1.2, or higher
  - If you do, then you're good :+1:

### Ubuntu 16.04 64-bit:

- normal non-GPU tensorflow prerequisites for building from source
- then do:
```
mkdir ~/Downloads
cd ~/Downloads
wget http://releases.llvm.org/4.0.0/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
sudo mkdir -p /usr/local/opt
cd /usr/local/opt
sudo tar -xf ~/Downloads/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
sudo mv clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04 llvm-4.0

cd
sudo apt-get install -y opencl-headers cmake clinfo git gcc g++ python3-numpy python3-dev python3-wheel zlib1g-dev
sudo apt-get install -y git gcc g++ python3-numpy python3-dev python3-wheel zlib1g-dev virtualenv swig python3-setuptools
sudo apt-get install -y default-jdk unzip zip
sudo apt-get install -y protobuf-c-compiler protobuf-compiler libprotobuf-dev libprotoc-dev

# bazel
cd ~/Downloads
wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel_0.4.5-linux-x86_64.deb
sudo dpkg -i bazel_0.4.5-linux-x86_64.deb
```

### Mac Sierra

- normal Mac non-GPU tensorflow prerequisites for building from source
- then do:

Download/install llvm-4.0:
```
cd ~
wget http://llvm.org/releases/4.0.0/clang+llvm-4.0.0-x86_64-apple-darwin.tar.xz
tar -xf clang+llvm-4.0.0-x86_64-apple-darwin.tar.xz
mv clang+llvm-4.0.0-x86_64-apple-darwin /usr/local/opt
ln -s /usr/local/opt/clang+llvm-4.0.0-x86_64-apple-darwin /usr/local/opt/llvm-4.0
```

Install other pre-requisites:
```
cd ~/Downloads
wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-installer-darwin-x86_64.sh
sh ./bazel-0.4.5-installer-darwin-x86_64.sh --user
brew install autoconf automake libtool shtool gflags
```

## Procedure

### Create Python virtualenv

Note that you dont strictly *need* a virtualenv, but it's mildly easier for me to document using a virtualenv, and I use a virtualenv for dev/testing.

```
# (this might be slightly different on mac)
if [[ ! -d ~/env3 ]]; then { virtualenv -p python3 ~/env3; } fi
source ~/env3/bin/activate
pip install numpy
deactivate
```

### Download Tensorflow

```
mkdir -p ~/git
cd ~/git
git clone --recursive https://github.com/hughperkins/tf-coriander
```

### Configure Tensorflow

```
util/run_configure.sh
```

### Build Coriander

```
cd ~/git/tf-coriander/third_party/coriander
mkdir build
cd build
cmake ..
make -j 8
# note: on Mac: dont use `sudo` in following command:
sudo make install
```

### Build grpc_cpp_plugin and protobuf

(these should probably be in the BUILD dependencies somehow, but I didnt figure out how to do this yet)
```
cd ~/git/tf-coriander
bazel build @grpc//:grpc_cpp_plugin
bazel build @protobuf//:protoc

if [[ ! -h bazel-out ]]; then { echo ERROR: bazel-out should be a link; } fi

# ^^^ make sure bazel-out is a link, if it's not, then stop, cos nothing
# else will work if it's not :-)

mkdir -p bazel-out/host/bin/external/grpc
mkdir -p bazel-out/host/bin/external/protobuf
ln -sf $PWD/bazel-bin/external/grpc/grpc_cpp_plugin bazel-out/host/bin/external/grpc/grpc_cpp_plugin
ln -sf $PWD/bazel-bin/external/protobuf/protoc bazel-out/host/bin/external/protobuf/protoc
```

### Build Tensorflow

```
cd ~/git/tf-coriander
bazel build --verbose_failures --logging 6 //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflowpkg
pip install --upgrade /tmp/tensorflowpkg/tensorflow-0.11.0rc0-py3-none-any.whl
```

### Test/validate

```
cd
python -c 'import tensorflow'
python ~/git/tf-coriander/tensorflow/stream_executor/cl/test/test_simple.py
# hopefully no errors :-)
# expected result:
# [[  4.   7.   9.]
# [  8.  10.  12.]]
cd ~/git/tf-coriander
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
cd ~/git/tf-coriander
git submodule update --init --recursive
```
- .. and then redo the build process, starting at section `Configure Tensorflow`
