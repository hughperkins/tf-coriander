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
        "//tensorflow/core/kernels:state",  # this triggers registration of Variable kernels

        "//tensorflow/core:ops",  # added this to get "Add" op
        "//tensorflow/core/kernels:math",  # adding this to try to get "Add" gpu kernel; adds a loootttt to compile time

        # all these are from deps for core/kernels:small
        # ":math",
        # ":ops_util",
        # "//tensorflow/core:core_cpu",
        # "//tensorflow/core:framework",
        # "//tensorflow/core:test",
        # "//tensorflow/core:test_main",
        # "//tensorflow/core:testlib",
    ]
)

cc_binary(
    name = "hughcross",
    srcs = [
        'hughcross.cc',
    ],
    deps = [
        "@cocl//:cocl-lib",
    ],
)

cc_binary(
    name = "testcu",
    srcs = [
        'testcu.cu.cc',
    ],
    deps = [
        "@cocl//:cocl-lib",
    ],
    copts = [
        '-x', 'cuda',
    ],
)
