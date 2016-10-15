#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/graph/testlib.h"  // gives test:;graph::Var
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/rendezvous.h"

#include <iostream>
#include <vector>
#include <functional>

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

    // Node* var = test::graph::Var(&g, DT_FLOAT, TensorShape({10}));
    // return test::graph::Var(g, DT_FLOAT, TensorShape({n}));

    auto var = test::graph::Var(&graph, DT_FLOAT, TensorShape({1}));

    Tensor data(DT_FLOAT, TensorShape({1}));
    data.flat<float>().setZero();
    auto zeros = test::graph::Constant(&graph, data);

    test::graph::Assign(&graph, var, zeros);

    // Graph* g = new Graph(OpRegistry::Global());
    // auto var = Var(g, 1);
    // test::graph::Assign(g, var, Zeros(g, 1));

    // test::Benchmark("cpu", run, GetOptions(), init).Run(iters);

    // from core/common_runtime/kernel_benchmark_testlib.cc Benchmark::Benchmark():
        // Benchmark::Benchmark(const string& device, Graph* g,
        //                      const SessionOptions* options, Graph* init) {
      // SessionOptions default_options;
      // if (!options) {
      //   options = &default_options;
      // }

      // testing::StopTiming();
      // // string t = str_util::Uppercase(device);
      // // device_ =
      // //     DeviceFactory::NewDevice(t, *options, "/job:localhost/replica:0/task:0");
      // CHECK(device_) << "Could not create a " << device << " device";

    thread::ThreadPool* pool = new thread::ThreadPool(opts.env, "blocking", 1);

    auto runner = [pool](std::function<void()> closure) {
        pool->Schedule(closure);
    };

    Rendezvous *rendez = NewLocalRendezvous();

    //   const int graph_def_version = g->versions().producer();

    //   LocalExecutorParams params;
    //   params.device = device_;
    //   params.function_library = nullptr;
    //   params.create_kernel = [this, graph_def_version](const NodeDef& ndef,
    //                                                    OpKernel** kernel) {
    //     return CreateNonCachedKernel(device_, nullptr, ndef, graph_def_version,
    //                                  kernel);
    //   };
    //   params.delete_kernel = [](OpKernel* kernel) {
    //     DeleteNonCachedKernel(kernel);
    //   };

    //   if (init) {
    //     Executor* init_exec;
    //     TF_CHECK_OK(NewLocalExecutor(params, init, &init_exec));
    //     Executor::Args args;
    //     args.rendezvous = rendez_;
    //     args.runner = runner;
    //     TF_CHECK_OK(init_exec->Run(args));
    //     delete init_exec;
    //   }

    //   TF_CHECK_OK(NewLocalExecutor(params, g, &exec_));
    // }

    // from core/common_runtime/kernel_benchmark_testlib.cc Benchmark::Run() :
    // void Benchmark::RunWithArgs(
    //     const std::vector<std::pair<const Node*, Tensor>>& inputs,
    //     const std::vector<const Node*>& outputs, int iters)

    // if (!device_) {
    //    return;
    // }
  // // Gets inputs' and outputs' rendezvous keys.
  // std::vector<std::pair<string, Tensor>> in;
  // for (const auto& p : inputs) {
  //   in.push_back({GetRendezvousKey(p.first), p.second});
  // }
  // std::vector<string> out;
  // for (const auto& n : outputs) {
  //   out.push_back(GetRendezvousKey(n));
  // }
  // Tensor unused;  // In benchmark, we don't care the return value.
  // bool is_dead;

  // // Warm up
  // Executor::Args args;
  // args.rendezvous = rendez_;
  // args.runner = [this](std::function<void()> closure) {
  //   pool_->Schedule(closure);
  // };
  // static const int kWarmupRuns = 3;
  // for (int i = 0; i < kWarmupRuns; ++i) {
  //   for (const auto& p : in) {
  //     Rendezvous::ParsedKey parsed;
  //     TF_CHECK_OK(Rendezvous::ParseKey(p.first, &parsed));
  //     rendez_->Send(parsed, Rendezvous::Args(), p.second, false);
  //   }
  //   TF_CHECK_OK(exec_->Run(args));
  //   for (const string& key : out) {
  //     Rendezvous::ParsedKey parsed;
  //     TF_CHECK_OK(Rendezvous::ParseKey(key, &parsed));
  //     rendez_->Recv(parsed, Rendezvous::Args(), &unused, &is_dead);
  //   }
  // }
  // TF_CHECK_OK(device_->Sync());
  // VLOG(3) << kWarmupRuns << " warmup runs done.";

  // testing::StartTiming();
  // while (iters-- > 0) {
  //   for (const auto& p : in) {
  //     Rendezvous::ParsedKey parsed;
  //     TF_CHECK_OK(Rendezvous::ParseKey(p.first, &parsed));
  //     rendez_->Send(parsed, Rendezvous::Args(), p.second, false);
  //   }
  //   TF_CHECK_OK(exec_->Run(args));
  //   for (const string& key : out) {
  //     Rendezvous::ParsedKey parsed;
  //     TF_CHECK_OK(Rendezvous::ParseKey(key, &parsed));
  //     rendez_->Recv(parsed, Rendezvous::Args(), &unused, &is_dead);
  //   }
  // }

  // TF_CHECK_OK(device_->Sync());
  // testing::StopTiming();


    cout << "all done" << endl;
    return 0;
}
