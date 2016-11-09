cc_library(
    name = 'cocl',
    copts = [
      '-Icuda-on-cl/include',
      '-Icuda-on-cl/include/cocl/cuda',
      '-Icuda-on-cl/src/EasyCL/thirdparty/clew/include',
    ],
    data = [
        ":cuda-on-cl/build/libclew.so",
        ":cuda-on-cl/build/libeasycl.so",
        ":cuda-on-cl/build/libcocl.so",
        ":cuda-on-cl/build/libclblast.so",
    ],
    srcs = [
        'cuda-on-cl/build/libclew.so',
        'cuda-on-cl/build/libeasycl.so',
        'cuda-on-cl/build/libcocl.so',
        'cuda-on-cl/build/libclblast.so',
    ],
    hdrs = glob([
        'cuda-on-cl/src/EasyCL/thirdparty/clew/include/clew.h',
        'cuda-on-cl/src/EasyCL/*.h',
        'cuda-on-cl/include/cocl/*.h',
        'cuda-on-cl/include/cocl/cuda/*.h',
    ]),
    visibility = ["//visibility:public"],
)
