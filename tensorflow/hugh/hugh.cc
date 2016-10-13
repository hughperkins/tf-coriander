#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include <iostream>

#include <cuda_runtime.h>

using namespace std;
using namespace tensorflow;

void assure_initialized();

int main(int argc, char *argv[]) {
    cout << "hugh" << endl;
    assure_initialized();
    Graph graph(OpRegistry::Global());
    float *gpuFloats;
    const int N = 1024;
    cudaMalloc((void **)(&gpuFloats), N * sizeof(float));
    cout << "all done" << endl;
    return 0;
}

