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
        ],
        exclude = [
            "**/*_test.cc",
            "dso_loader.cc",
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
        # "@usr_local_lib//:libcocl",
        "@usr_lib_x8664linux//:libopencl",
    ],
    copts = [
        '-Ithird_party/cocl_links/include/cocl'
    ],
    linkopts = [
        "-ldl",
        '-lcocl',
        '-lOpenCL',
    ],
    visibility = ["//visibility:public"],
    deps = [
        #"@cocl//:cocl-headers",
        #"@cocl_links//:libcocl",
        "//third_party/cocl_links:headers",
        "//third_party/cocl_links:libcocl",

        # "@local_usr_lib//:libopencl",
        # "@cocl//:cocl-lib",
        # "@clew//:clew-lib",
        # "@EasyCL//:easycl-lib",
        "//tensorflow/core:lib",
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
        "@usr_lib_x8664linux//:libopencl",
        # "@usr_local_lib//:libcocl",
        "//third_party/cocl_links:libcocl",
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
    copts = [
        # '-Ithird_party/eigen3',
        '-Iexternal/eigen_archive',
    ],
    linkopts = [
        '-lOpenCL'
    ],
    deps = [
        "@usr_lib_x8664linux//:libopencl",
        # "@usr_local_lib//:libcocl",
        "//third_party/cocl_links:libcocl",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/cc:const_op",
        "//tensorflow/core:testlib",  # for test::graph::Var
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:array_ops_op_lib",  # trying to get Placeholder
        "//tensorflow/core/kernels:cwise_op",
        #"//tensorflow/core/kernels:cwise_op_add",
        #"//tensorflow/core/kernels:cwise_ops",

        "//tensorflow/cc:cc_ops",  # trying to get placeholder

        "//third_party/eigen3",
    ]
)

cc_binary(
    name = "test_cl_trainer",
    srcs = [
        "cl/test/example_trainer.cc",
    ],
    copts = [
        # '-Ithird_party/eigen3',
        '-Iexternal/eigen_archive',
    ],
    linkopts = [
        '-lOpenCL'
    ],
    deps = [
        "@usr_lib_x8664linux//:libopencl",
        # "@usr_local_lib//:libcocl",
        "//third_party/cocl_links:libcocl",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/cc:const_op",
        "//tensorflow/core:testlib",  # for test::graph::Var
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:array_ops_op_lib",  # trying to get Placeholder
        "//tensorflow/core/kernels:cwise_op",
        #"//tensorflow/core/kernels:cwise_op_add",
        #"//tensorflow/core/kernels:cwise_ops",

        "//tensorflow/cc:cc_ops",  # trying to get placeholder

        "//third_party/eigen3",
    ]
)

cc_binary(
    name = "test_cl_loadproto",
    srcs = [
        "cl/test/loadproto.cc",
    ],
    copts = [
        # '-Ithird_party/eigen3',
        '-Iexternal/eigen_archive',
    ],
    linkopts = [
        '-lOpenCL'
    ],
    deps = [
        "//tensorflow/core:tensorflow",

        "@usr_lib_x8664linux//:libopencl",
        # "@usr_local_lib//:libcocl",
        "//third_party/cocl_links:libcocl",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/cc:const_op",
        "//tensorflow/core:testlib",  # for test::graph::Var
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:array_ops_op_lib",  # trying to get Placeholder
        "//tensorflow/core/kernels:cwise_op",
        #"//tensorflow/core/kernels:cwise_op_add",
        #"//tensorflow/core/kernels:cwise_ops",

        "//tensorflow/cc:cc_ops",  # trying to get placeholder

        "//third_party/eigen3",
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
