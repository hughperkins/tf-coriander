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

#include "tensorflow/hugh/cl_platform.h"

#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

#include <iostream>

namespace perftools {
namespace gputools {
namespace cl {

PLATFORM_DEFINE_ID(kClPlatformId);


ClPlatform::ClPlatform() {
    std::cout << "ClPlatform()" << std::endl;
    port::Printf("using port clplatform");
}

ClPlatform::~ClPlatform() {}

Platform::Id ClPlatform::id() const { return kClPlatformId; }

int ClPlatform::VisibleDeviceCount() const {
return 1;
}

const string& ClPlatform::Name() const { return "OpenCL"; }

port::StatusOr<StreamExecutor*> ClPlatform::ExecutorForDevice(int ordinal) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf(
            "failed initializing StreamExecutor for cl device")};
}
port::StatusOr<StreamExecutor*> ClPlatform::ExecutorForDeviceWithPluginConfig(
  int ordinal, const PluginConfig& plugin_config) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf(
            "failed initializing StreamExecutor for cl device")};
}
port::StatusOr<StreamExecutor*> ClPlatform::GetExecutor(
  const StreamExecutorConfig& config) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf(
            "failed initializing StreamExecutor for cl device")};
}
port::StatusOr<std::unique_ptr<StreamExecutor>> ClPlatform::GetUncachedExecutor(
  const StreamExecutorConfig& config) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf(
            "failed initializing StreamExecutor for cl device")};
}
void ClPlatform::UnregisterTraceListener(TraceListener* listener) {
    
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

