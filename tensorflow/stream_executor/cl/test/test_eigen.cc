#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/const_op.h"
// #include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/graph/testlib.h"  // gives test:;graph::Var
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/common_runtime/executor.h"  // gives LocalExecutorParams
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
// #include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"

#include <iostream>
#include <vector>
#include <functional>

// #include <cuda_runtime.h>

using namespace std;
using namespace tensorflow;

// void assure_initialized();

// void test2() {
//   // from core/common_runtime/graph_runner_test.cc
//   Scope root = Scope::NewRootScope();
//   auto p1 = ops::Placeholder(root.WithOpName("p1"), DT_FLOAT);
//   auto p2 = ops::Placeholder(root.WithOpName("p2"), DT_FLOAT);
//   auto add = ops::Add(root.WithOpName("add"), p1, p2);

//   Tensor p1_data(DT_FLOAT, TensorShape({}));
//   Tensor p2_data(DT_FLOAT, TensorShape({}));
//   p1_data.scalar<float>()() = 1.0f;
//   p2_data.scalar<float>()() = 2.0f;
//   std::vector<std::pair<string, Tensor>> inputs = {{"p1:0", p1_data},
//                                                    {"p2:0", p2_data}};

//   std::vector<Tensor> outputs;
//   // seems this runs on cpu...
//   Status s = GraphRunner::Run(root.graph(), nullptr, Env::Default(), inputs,
//                               {"add:0"}, &outputs);
//   TF_ASSERT_OK(s);
//   tensorflow::test::internal::ExpectEqual(3.0f, outputs[0].scalar<float>()());
// }

// from kernel_benchmark_testlib.cc
string GetRendezvousKey(const Node* node) {
  string send_device;
  TF_CHECK_OK(GetNodeAttr(node->def(), "send_device", &send_device));
  string recv_device;
  TF_CHECK_OK(GetNodeAttr(node->def(), "recv_device", &recv_device));
  string tensor_name;
  TF_CHECK_OK(GetNodeAttr(node->def(), "tensor_name", &tensor_name));
  uint64 send_device_incarnation;
  TF_CHECK_OK(GetNodeAttr(node->def(), "send_device_incarnation",
                          reinterpret_cast<int64*>(&send_device_incarnation)));
  return Rendezvous::CreateKey(send_device, send_device_incarnation,
                               recv_device, tensor_name, FrameAndIter(0, 0));
}

void test3() {
  int iters = 1;
  SessionOptions options;
  // Graph* g = new Graph(OpRegistry::Global());

  // Scope root = Scope::NewRootScope();
  // auto p1 = ops::Placeholder(root.WithOpName("p1"), DT_FLOAT);
  // auto p2 = ops::Placeholder(root.WithOpName("p2"), DT_FLOAT);
  // auto add = ops::Add(root.WithOpName("add"), p1, p2);

  Graph* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({3, 4}));

  // Tensor p1_data(DT_FLOAT, TensorShape({}));
  // Tensor p2_data(DT_FLOAT, TensorShape({}));
  // p1_data.scalar<float>()() = 1.0f;
  // p2_data.scalar<float>()() = 2.0f;
  // std::vector<std::pair<string, Tensor>> inputs = {{"p1:0", p1_data},
  //                                                  {"p2:0", p2_data}};

  // std::vector<Tensor> outputs;

  tensorflow::test::Benchmark("gpu", g, &options).Run(iters);
}

int main(int argc, char *argv[]) {
    cout << "hugh start" << endl;

    // test3();

    // cout << "hugh done" << endl;
    // return 0;

    // from common_runtime/graph_runner_test.cc
    Scope root = Scope::NewRootScope();
    auto c = ops::Const(root, 42.0f);
    std::vector<Tensor> outputs;
    std::cout << "3" << std::endl;
    //Status s = GraphRunner::Run(root.graph(), nullptr, Env::Default(), {},
    //                            {c.name()}, &outputs);

    SessionOptions opts;
    // lets try things from core/common_runtime/kernel_benchmark_testlib.cc next?
    // string device = "gpu";  // this kind of from core/kernels/basic_ops_benchmark_test.cc
    Device* device =
        DeviceFactory::NewDevice("GPU", opts, "/job:localhost/replica:0/task:0");
    std::cout << "7" << std::endl;
    CHECK(device) << "Could not create a " << device << " device";
    std::cout << "8" << std::endl;

    Graph init_graph(OpRegistry::Global());
    // auto var = test::graph::Var(&init_graph, DT_FLOAT, TensorShape({1}));
    Tensor data(DT_FLOAT, TensorShape({1}));
    data.flat<float>().setZero();
    auto zeros = test::graph::Constant(&init_graph, data);

    // test::graph::Assign(&init_graph, var, zeros);

    Graph exec_graph(OpRegistry::Global());
    // auto var_exec = test::graph::Var(&exec_graph, DT_FLOAT, TensorShape({1}));
    Tensor data_exec(DT_FLOAT, TensorShape({1}));
    data_exec.flat<float>().setZero();
    auto zeros_exec = test::graph::Constant(&exec_graph, data_exec);

    auto afteradd = test::graph::Multi(&exec_graph, "Add", {zeros_exec, zeros_exec});

    // test::graph::Assign(&exec_graph, var_exec, zeros_exec);

    // from core/common_runtime/kernel_benchmark_testlib.cc Benchmark::Benchmark():
        // Benchmark::Benchmark(const string& device, Graph* g,
        //                      const SessionOptions* options, Graph* init) {
    thread::ThreadPool* pool = new thread::ThreadPool(opts.env, "blocking", 1);

    auto runner = [pool](std::function<void()> closure) {
        pool->Schedule(closure);
    };

    Rendezvous *rendez = NewLocalRendezvous();

    const int graph_def_version = exec_graph.versions().producer();

    LocalExecutorParams params;
    params.device = device;
    params.function_library = nullptr;
    Executor* exec = 0;
    params.create_kernel = [device, graph_def_version]
            (const NodeDef& ndef, OpKernel** kernel) {
        return CreateNonCachedKernel(device, nullptr, ndef, graph_def_version,
                                     kernel);
    };
    params.delete_kernel = [](OpKernel* kernel) {
        DeleteNonCachedKernel(kernel);
    };

    // if (init) {
        Executor* init_exec;
        TF_CHECK_OK(NewLocalExecutor(params, &init_graph, &init_exec));
        cout << "after first executor" << endl;
        Executor::Args args;
        args.rendezvous = rendez;
        args.runner = runner;
        auto initexecrunres = init_exec->Run(args);
        cout << "after init_exec->Run" << endl;
        // delete init_exec;
    // }
    auto execres = NewLocalExecutor(params, &exec_graph, &exec);
    cout << "after second newlocalexecutor; got execres" << endl;
    cout << "execres: " << execres << endl;
    TF_CHECK_OK(execres);

    // from core/common_runtime/kernel_benchmark_testlib.cc Benchmark::Run() :
    // void Benchmark::RunWithArgs(
    //     const std::vector<std::pair<const Node*, Tensor>>& inputs,
    //     const std::vector<const Node*>& outputs, int iters)

    // if (!device_) {
    //    return;
    // }
  // Gets inputs' and outputs' rendezvous keys.
  // std::vector<std::pair<string, Tensor>> in;
  // for (const auto& p : inputs) {
  //   in.push_back({GetRendezvousKey(p.first), p.second});
  // }
  std::vector<string> out;
  // for (const auto& n : outputs) {
    out.push_back(GetRendezvousKey(outputs));
  // }
  Tensor unused;  // In benchmark, we don't care the return value.
  bool is_dead;

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

    // test2();

    cout << "all done" << endl;
    return 0;
}
