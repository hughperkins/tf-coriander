licenses([
    # cuda-on-cl is under the Apache license
    "reciprocal",  # Apache License
    "notice",  # Apache License
])

cc_library(
    name = 'libcocl',
    srcs = [
      'lib/libcocl.a',
    ],
    visibility = ["//visibility:public"],
     linkstatic = 1,
)

cc_library(
    name = 'headers',
    srcs = [
    ],
    hdrs = glob([
      'include/cocl/*.h',
      'include/cocl/cuda/*.h',
    ]),
    visibility = ["//visibility:public"],
)
