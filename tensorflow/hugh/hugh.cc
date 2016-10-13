#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/public/session_options.h"

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
    auto c = ops::Const(root, 42.0f);
    std::vector<Tensor> outputs;
    Status s = GraphRunner::Run(root.graph(), nullptr, Env::Default(), {},
                                {c.name()}, &outputs);

    SessionOptions opts;
    // lets try things from core/common_runtime/kernel_benchmark_testlib.cc next?

    cout << "all done" << endl;
    return 0;
}
