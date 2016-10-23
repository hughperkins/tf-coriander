// copyright Hugh Perkins 2016, The Tensorflow Authors 2015.  All rights reserved
/*

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

#include "cuda.h"

#include "tensorflow/stream_executor/cl/cl_platform.h"

#include "tensorflow/stream_executor/cl/cl_platform_id.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/cl/cl_gpu_executor.h"
// #include "EasyCL.h"

// namespace perftools {
// namespace gputools {
// namespace 
// using namespace perftools::gputools::cuda;
// #include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include "tensorflow/stream_executor/cl/cl_driver.h"

#include <iostream>

namespace perftools {
namespace gputools {

namespace cl {

// PLATFORM_DEFINE_ID(kClPlatformId);
const std::string name = "CL";

extern "C" {
    void hostside_opencl_funcs_assure_initialized();
}

ClPlatform::ClPlatform() {
    // std::cout << "ClPlatform()" << std::endl;
    hostside_opencl_funcs_assure_initialized();
    // port::Printf("using port clplatform");
    // EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
    // std::cout << "end of ClPlatform() constructor" << std::endl;
}

ClPlatform::~ClPlatform() {
    // std::cout << "~ClPlatform()" << std::endl;
}

Platform::Id ClPlatform::id() const {
    // std::cout << "ClPlatform::id()" << std::endl;
    return kClPlatformId;
}

// int ClPlatform::VisibleDeviceCount() const {
//     std::cout << "ClPlatform::VisibleDeviceCount" << std::endl;
//     return 1;
// }
int ClPlatform::VisibleDeviceCount() const {
  // Throw away the result - it logs internally, and this [containing] function
  // isn't in the path of user control. It's safe to call this > 1x.
  // std::cout << "ClPlatform::VisibleDeviceCount()" << std::endl;
  if (!CLDriver::Init().ok()) {
    // std::cout << "soi-disant CLDriver failed to initialize" << std::endl;
    return -1;
  }
  // std::cout << "soi-disant CLDriver initialized ok." << std::endl;
  // std::cout << "num devices " << CLDriver::GetDeviceCount() << std::endl;

  return CLDriver::GetDeviceCount();
}

const string& ClPlatform::Name() const {
    // std::cout << "ClPlatform::name()" << std::endl;
    return name;
}

port::StatusOr<StreamExecutor*> ClPlatform::ExecutorForDevice(int ordinal) {
    // std::cout << "ClPlatform::ExecutorForDevice(" << ordinal << ")" << std::endl;
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> ClPlatform::ExecutorForDeviceWithPluginConfig(
  int ordinal, const PluginConfig& plugin_config) {
    // std::cout << "ClPlatform::ExecutorForDeviceWithPluginConfig()" << std::endl;

  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);

    // return port::Status{
    //     port::error::INTERNAL,
    //     port::Printf(
    //         "failed initializing StreamExecutor for cl device")};
}

port::StatusOr<StreamExecutor*> ClPlatform::GetExecutor(
  const StreamExecutorConfig& config) {
    // std::cout << "ClPlatform::GetExecutor()" << std::endl;
  mutex_lock lock(mu_);

  port::StatusOr<StreamExecutor*> status = executor_cache_.Get(config);
  if (status.ok()) {
    return status.ValueOrDie();
  }

  port::StatusOr<std::unique_ptr<StreamExecutor>> executor =
      GetUncachedExecutor(config);
  if (!executor.ok()) {
    return executor.status();
  }

  StreamExecutor* naked_executor = executor.ValueOrDie().get();
  executor_cache_.Insert(config, executor.ConsumeValueOrDie());
  return naked_executor;
}

port::StatusOr<std::unique_ptr<StreamExecutor>> ClPlatform::GetUncachedExecutor(
  const StreamExecutorConfig& config) {
    // std::cout << "ClPlatform::GetUncachedExecutor()" << std::endl;
  auto executor = port::MakeUnique<StreamExecutor>(
      this, new CLExecutor(config.plugin_config));
  // std::cout << "cl_platform.cc GetUncachedExecutor() created new CUDAExecutor" << std::endl;
  auto init_status = executor->Init(config.ordinal, config.device_options);
  if (!init_status.ok()) {
    std::cout << "cl_platform.cc GetUncachedExecutor() CUDAExecutor->init() failed" << std::endl;
    return port::Status{
        port::error::INTERNAL,
        port::Printf(
            "failed initializing StreamExecutor for CL device ordinal %d: %s",
            config.ordinal, init_status.ToString().c_str())};
  }
  // std::cout << "created executor ok" << std::endl;

  return std::move(executor);
}

void ClPlatform::UnregisterTraceListener(TraceListener* listener) {
    // std::cout << "ClPlatform::UnregisterTraceListener()" << std::endl;
}

} // namespace cl

static void InitializeClPlatform() {
    std::unique_ptr<cl::ClPlatform> platform(new cl::ClPlatform);
    SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

} // namespace gputools
} // namespace perftools

REGISTER_MODULE_INITIALIZER(cl_platform,
                            perftools::gputools::InitializeClPlatform());
