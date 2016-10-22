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

#include "tensorflow/core/common_runtime/gpu/gpu_init.h"

#include <string>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/stream_executor_util.h"

#include <iostream>

namespace gpu = ::perftools::gputools;

namespace tensorflow {

Status ValidateGPUMachineManager() {
  // std::cout << "ValidateGPUMachineManager" << std::endl;
  auto result = gpu::MultiPlatformManager::PlatformWithName("CL");
  if (!result.ok()) {
    return StreamExecutorUtil::ConvertStatus(result.status());
  }

  // std::cout << "ValidateGPUMachineManager found Platform with name CL" << std::endl;
  return Status::OK();
}

gpu::Platform* GPUMachineManager() {
  auto result = gpu::MultiPlatformManager::PlatformWithName("CL");
  if (!result.ok()) {
    std::cout << "ValidateGPUMachineManager Could not find Platform with name CL" << std::endl;
    LOG(FATAL) << "Could not find Platform with name CL";
    return nullptr;
  }

  return result.ValueOrDie();
}

}  // namespace tensorflow
