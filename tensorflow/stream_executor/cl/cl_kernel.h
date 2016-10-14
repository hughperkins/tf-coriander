/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// The CL implementation of the StreamExecutorInterface functionality.
// CL inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the CL streams
// programming model provided by the libcl.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef TENSORFLOW_STREAM_EXECUTOR_CL_CL_KERNEL_H_
#define TENSORFLOW_STREAM_EXECUTOR_CL_CL_KERNEL_H_

#include "cuda.h"

#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/cl/cl_driver.h"
#include "tensorflow/stream_executor/lib/casts.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/logging.h"

#ifdef PLATFORMS_GPUS_CL_DYNAMIC_LIBCL_DYNAMIC_LIBCL_H_
#error \
    "No driver calls in this file, wrap driver functionality in cl_driver.cc."
#endif

#ifdef __CL_RUNTIME_H__
#error \
    "CL runtime being included into CL GPU executor; should be driver only."
#endif

namespace perftools {
namespace gputools {
namespace cl {

// Wraps a CUfunction to implement the platform-independent KernelInterface.
class CLKernel : public internal::KernelInterface {
 public:
  CLKernel() : cl_function_(nullptr), arity_(0),
                 preferred_cache_config_(KernelCacheConfig::kNoPreference) {}

  // Note that the function is unloaded when the module is unloaded, and the
  // module that the function is contained in is owned by the CLExecutor.
  ~CLKernel() override {}

  // As arity cannot be reflected upon using the CL API, the arity is
  // explicitly set during the CLExecutor::GetKernel initialization process.
  void set_arity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const override { return arity_; }

  // Returns the CUfunction value for passing to the CL API.
  CUfunction AsCLFunctionValue() const {
    DCHECK(cl_function_ != nullptr);
    return const_cast<CUfunction>(cl_function_);
  }

  // Returns the slot that the CUfunction is stored within for this object,
  // for the CL API which wants to load into a CUfunction*.
  CUfunction *cl_function_ptr() { return &cl_function_; }

  // CL supports setting the preferred cache configuration of a CUfunction
  // (more-or-less equivalent to a CLKernel). We support this via the below
  // functions; users can set a preference, and that is applied when the kernel
  // is [lazy-]loaded (in CLExecutor::Launch). The alternative would be to
  // load the kernel & set the preference when the user calls the setter below;
  // either approach is valid.
  // Sets the current kernel cache configuration preference.
  void SetPreferredCacheConfig(KernelCacheConfig config) override {
    preferred_cache_config_ = config;
  }

  // Returns the current kernel cache configuration preference.
  KernelCacheConfig GetPreferredCacheConfig() const override {
    return preferred_cache_config_;
  }

  // Returns the current kernel cache configuration preference as a
  // CUfunc_cache.
  CUfunc_cache GetCLCacheConfig() const {
    switch (preferred_cache_config_) {
      case KernelCacheConfig::kNoPreference:
        return CU_FUNC_CACHE_PREFER_NONE;
      case KernelCacheConfig::kPreferShared:
        return CU_FUNC_CACHE_PREFER_SHARED;
      case KernelCacheConfig::kPreferL1:
        return CU_FUNC_CACHE_PREFER_L1;
      case KernelCacheConfig::kPreferEqual:
        return CU_FUNC_CACHE_PREFER_EQUAL;
      default:
        LOG(FATAL) << "Unknown KernelCacheConfig"
                   << static_cast<int32>(preferred_cache_config_);
    }
  }

 private:
  CUfunction cl_function_;  // Wrapped CL kernel handle.
  unsigned arity_;            // Number of formal parameters the kernel takes.

  // Preferred (but not required) cache configuration for this kernel.
  KernelCacheConfig preferred_cache_config_;
};

// Given a platform-independent kernel datatype, returns the (const) internal
// CL platform implementation pointer.
inline const CLKernel *AsCLKernel(const KernelBase *kernel) {
  return static_cast<const CLKernel *>(kernel->implementation());
}

// Given a platform-independent kernel datatype, returns the (non-const)
// internal CL platform implementation pointer.
inline CLKernel *AsCLKernel(KernelBase *kernel) {
  return static_cast<CLKernel *>(kernel->implementation());
}

}  // namespace cl
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CL_CL_KERNEL_H_
