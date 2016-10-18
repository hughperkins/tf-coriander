licenses(["restricted"])

load("//tensorflow:tensorflow.bzl", "if_cuda")

cc_library(
    name = "stream_executor",
    srcs = glob(
        [
            "*.cc",
            "lib/*.cc",
            "platform/default/*.cc",
            "cl/*.cc",
            # "cuda/cuda_driver.cc",
            # "cuda/cuda_diagnostics.cc",
            # "cuda/cuda_activation.cc",
            # "cl/cl_gpu_executor.cc",
            # "cuda/cuda_timer.cc",
            # "cuda/cuda_stream.cc",
        ],
        exclude = [
            "**/*_test.cc",
        ],
    # ) + if_cuda(
    #     glob([
    #         "cuda/*.cc",
    #     ]),
    ),
    hdrs = glob([
        "*.h",
        "cuda/*.h",
        "cl/*.h",
        "lib/*.h",
        "lib/gtl/*.h",
        "platform/**/*.h",
    ]),
    data = [
        "//tensorflow/core:cuda",
        "@usr_local_lib//:libcocl",
        "@usr_lib_x8664linux//:libopencl",
        # "@local_config_cuda//cuda:cublas",
        # "@local_config_cuda//cuda:cudnn",
        # "@local_config_cuda//cuda:cufft",
        # "@local_config_cuda//cuda:curand",
    ],
    linkopts = [
        "-ldl",
        '-lcocl',
        '-lOpenCL',
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@cuda_for_cocl//:cuda_headers",
        # "@local_usr_lib//:libopencl",
        # "@cocl//:cocl-lib",
        # "@clew//:clew-lib",
        # "@EasyCL//:easycl-lib",
        "//tensorflow/core:lib",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "test_cl",
    srcs = [
        "cl/test/test.cc",
    ],
    linkopts = [
        '-lOpenCL'
    ],
    deps = [
        "@cuda_for_cocl//:cuda_headers",
        "@usr_lib_x8664linux//:libopencl",
        "@usr_local_lib//:libcocl",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:state_ops_op_lib",   # gives REGISTER_OP("Variable") ?
        "//tensorflow/core:testlib",  # for test::graph::Var
        "//tensorflow/core/kernels:state",  # this triggers registration of Variable kernels
    ]
)

cc_binary(
    name = "test_cl_eigen",
    srcs = [
        "cl/test/test_eigen.cc",
    ],
    linkopts = [
        '-lOpenCL'
    ],
    deps = [
        "@cuda_for_cocl//:cuda_headers",
        "@usr_lib_x8664linux//:libopencl",
        "@usr_local_lib//:libcocl",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/cc:const_op",
        "//tensorflow/core:testlib",  # for test::graph::Var
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core/kernels:cwise_op",
    ]
)

        # "//tensorflow/core/kernels:math",  # adding this to try to get "Add" gpu kernel; adds a loootttt to compile time
        # "//tensorflow/core:ops",  # added this to get "Add" op
        # "//tensorflow/core/kernels:state",  # this triggers registration of Variable kernels
        # "//tensorflow/core:state_ops_op_lib",   # gives REGISTER_OP("Variable") ?
        # "//tensorflow/core:user_ops_op_lib",  # gives REGISTER_OP("Variable") ?
        # "//tensorflow/cc:cc_ops",

        # all these are from deps for core/kernels:small
        # ":math",
        # ":ops_util",
        # "//tensorflow/core:core_cpu",
        # "//tensorflow/core:framework",
        # "//tensorflow/core:test",
        # "//tensorflow/core:test_main",
        # "//tensorflow/core:testlib",
