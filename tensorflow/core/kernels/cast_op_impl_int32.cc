/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/cast_op_impl.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromInt32(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, CPUDevice, int32);
  return nullptr;
}

// #if GOOGLE_CUDA

#undef CURRY_TYPES3
#define CURRY_TYPES3(FN, arg0, arg1)   \
  FN(arg0, arg1, bool);                \
  FN(arg0, arg1, int32);               \
  FN(arg0, arg1, float);               

//  FN(arg0, arg1, uint8);               \
  FN(arg0, arg1, int8);                \
  FN(arg0, arg1, uint16);              \
  FN(arg0, arg1, int16);               \
  FN(arg0, arg1, int64);               \
  FN(arg0, arg1, Eigen::half);         \
  FN(arg0, arg1, double);              \
  FN(arg0, arg1, std::complex<float>); \
  FN(arg0, arg1, std::complex<double>)

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromInt32(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, GPUDevice, int32);
  return nullptr;
}
// #endif  // GOOGLE_CUDA

}  // namespace tensorflow

