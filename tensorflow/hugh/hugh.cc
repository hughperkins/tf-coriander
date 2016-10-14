#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/common_runtime/device_factory.h"

#include <iostream>
#include <vector>

// #include <cuda_runtime.h>

using namespace std;
using namespace tensorflow;

// void assure_initialized();

int main(int argc, char *argv[]) {
    cout << "hugh" << endl;
    // assure_initialized();
    Graph graph(OpRegistry::Global());
    // float *gpuFloats;
    // const int N = 1024;
    //cudaMalloc((void **)(&gpuFloats), N * sizeof(float));

    // from common_runtime/graph_runner_test.cc
    Scope root = Scope::NewRootScope();
    std::cout << "1" << std::endl;
    auto c = ops::Const(root, 42.0f);
    std::cout << "2" << std::endl;
    std::vector<Tensor> outputs;
    std::cout << "3" << std::endl;
    //Status s = GraphRunner::Run(root.graph(), nullptr, Env::Default(), {},
    //                            {c.name()}, &outputs);
    std::cout << "4" << std::endl;

    SessionOptions opts;
    std::cout << "5" << std::endl;
    // lets try things from core/common_runtime/kernel_benchmark_testlib.cc next?
    string device = "gpu";  // this kind of from core/kernels/basic_ops_benchmark_test.cc
    string t = str_util::Uppercase(device);
    std::cout << "6" << std::endl;
    Device* device_ =
        DeviceFactory::NewDevice(t, opts, "/job:localhost/replica:0/task:0");
    std::cout << "7" << std::endl;
    CHECK(device_) << "Could not create a " << device << " device";
    std::cout << "8" << std::endl;

    cout << "all done" << endl;
    return 0;
}
