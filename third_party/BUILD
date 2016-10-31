cc_library(
    name = 'cocl',
    copts = [
      '-Icuda-on-cl/include',
      '-Icuda-on-cl/include/cocl/cuda',
    ],
    data = [
        ":cuda-on-cl/build/libcocl.so",
        ":cuda-on-cl/build/clblast/libclblast.so",
    ],
    srcs = [
        'cuda-on-cl/build/libcocl.so',
        'cuda-on-cl/build/clblast/libclblast.so',
    ],
    hdrs = glob([
        'cuda-on-cl/include/cocl/*.h',
        'cuda-on-cl/include/cocl/cuda/*.h',
    ]),
    visibility = ["//visibility:public"],
)
