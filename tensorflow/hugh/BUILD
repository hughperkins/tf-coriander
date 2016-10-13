cc_binary(
    name = "hugh",
    srcs = [
        "hugh.cc",
        #"../stream_executor/cl_platform.cc",
        #"../stream_executor/cl_platform.h",
    ],
    deps = [
        "@cocl//:cocl-lib",
        "@clew//:clew-lib",
        "@EasyCL//:easycl-lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
    ]
)
