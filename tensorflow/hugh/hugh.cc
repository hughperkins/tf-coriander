#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include <iostream>

using namespace std;
using namespace tensorflow;

int main(int argc, char *argv[]) {
    cout << "hugh" << endl;
    Graph graph(OpRegistry::Global());
    cout << "all done" << endl;
    return 0;
}

