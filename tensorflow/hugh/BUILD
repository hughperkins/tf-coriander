cc_binary(
    name = "hugh",
    srcs = [
        "hugh.cc",
        #"../stream_executor/cl_platform.cc",
        #"../stream_executor/cl_platform.h",
    ],
    deps = [
        "@cuda_for_cocl//:cuda_headers",
        "@local_usr_lib//:libopencl",
        "@cocl//:cocl-lib",
        "@clew//:clew-lib",
        "@EasyCL//:easycl-lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:state_ops_op_lib",   # gives REGISTER_OP("Variable") ?
        # "//tensorflow/core:user_ops_op_lib",  # gives REGISTER_OP("Variable") ?
        "//tensorflow/core:testlib",  # for test::graph::Var
    ]
)
