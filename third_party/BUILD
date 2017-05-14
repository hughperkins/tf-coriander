load("//tensorflow/core:platform/default/build_config.bzl", "tf_add_so_suffix")

cc_library(
    name = 'cocl',
    copts = [
      '-Icuda-on-cl/include',
      '-Icuda-on-cl/include/cocl/cuda',
      '-Icuda-on-cl/src/EasyCL/thirdparty/clew/include',
    ],
    data = [
        tf_add_so_suffix(":cuda-on-cl/build/libclew"),
        tf_add_so_suffix(":cuda-on-cl/build/libeasycl"),
        tf_add_so_suffix(":cuda-on-cl/build/libcocl"),
        tf_add_so_suffix(":cuda-on-cl/build/libclblast"),
    ],
    srcs = [
        tf_add_so_suffix('cuda-on-cl/build/libclew'),
        tf_add_so_suffix('cuda-on-cl/build/libeasycl'),
        tf_add_so_suffix('cuda-on-cl/build/libcocl'),
        tf_add_so_suffix('cuda-on-cl/build/libclblast'),
    ],
    hdrs = glob([
        'cuda-on-cl/src/EasyCL/thirdparty/clew/include/clew.h',
        'cuda-on-cl/src/EasyCL/*.h',
        'cuda-on-cl/include/cocl/*.h',
        'cuda-on-cl/include/cocl/cuda/*.h',
    ]),
    visibility = ["//visibility:public"],
)
