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
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cudnn",
        "@local_config_cuda//cuda:cufft",
        "@local_config_cuda//cuda:curand",
    ],
    linkopts = [
        "-ldl",
        # "-lcocl",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@cuda_for_cocl//:cuda_headers",
        "@local_usr_lib//:libopencl",
        "@cocl//:cocl-lib",
        "@clew//:clew-lib",
        "@EasyCL//:easycl-lib",
        "//tensorflow/core:lib",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = 1,
)
