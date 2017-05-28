load("//tensorflow/core:platform/default/build_config.bzl", "tf_add_so_suffix")

cc_library(
    name = 'cocl',
    copts = [
      '-Icoriander/include',
      '-Icoriander/include/cocl/cuda',
      '-Icoriander/src/EasyCL/thirdparty/clew/include',
    ],
    data = [
        tf_add_so_suffix(":coriander/build/libclew"),
        tf_add_so_suffix(":coriander/build/libeasycl"),
        tf_add_so_suffix(":coriander/build/libcocl"),
        tf_add_so_suffix(":coriander/build/libclblast"),
    ],
    srcs = [
        tf_add_so_suffix('coriander/build/libclew'),
        tf_add_so_suffix('coriander/build/libeasycl'),
        tf_add_so_suffix('coriander/build/libcocl'),
        tf_add_so_suffix('coriander/build/libclblast'),
    ],
    hdrs = glob([
        'coriander/src/EasyCL/thirdparty/clew/include/clew.h',
        'coriander/src/EasyCL/*.h',
        'coriander/include/cocl/*.h',
        'coriander/include/cocl/cuda/*.h',
        'coriander/include/cocl/proxy_includes/*.h',
    ]),
    visibility = ["//visibility:public"],
)
