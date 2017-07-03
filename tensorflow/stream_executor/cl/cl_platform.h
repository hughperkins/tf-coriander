// copyright Hugh Perkins 2016, Tensorflow authors 2015. All rights reserved

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
#pragma once

#include <memory>
#include "tensorflow/stream_executor/platform/port.h"
#include <vector>

#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/trace_listener.h"

namespace perftools {
namespace gputools {
namespace cl {

    extern const Platform::Id kClPlatformId;

class ClPlatform : public Platform {
public:
    ClPlatform();
    ~ClPlatform() override;
    int VisibleDeviceCount() const override;
    Platform::Id id() const override;
    const string& Name() const override;
    virtual port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal);
    virtual port::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& plugin_config);
    virtual port::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config);
    virtual port::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config);
    virtual void UnregisterTraceListener(TraceListener* listener);
    virtual void RegisterTraceListener(std::unique_ptr<TraceListener> listener) {}

private:
    // mutex that guards internal state.
    mutable mutex mu_;

    // Cache of created executors.
    ExecutorCache executor_cache_;

    SE_DISALLOW_COPY_AND_ASSIGN(ClPlatform);
};

}
}
}

