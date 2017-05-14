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
    copts = [
        '-Ithird_party/cuda-on-cl/include/cocl',
        '-Ithird_party/cuda-on-cl/src/EasyCL',
        '-Ithird_party/cuda-on-cl/src/EasyCL/third_party/clew/include',
    ],
    linkopts = [
        "-ldl",
        '-lcocl',
        '-lclblast',
        '-leasycl',
        '-lclew',
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:cocl",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)
