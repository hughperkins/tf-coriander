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

#include "cuda.h"

#include "tensorflow/stream_executor/cl/cl_activation.h"

#include "tensorflow/stream_executor/cl/cl_driver.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace perftools {
namespace gputools {
namespace cl {

cl::ClContext* ExtractClContext(CLExecutor *cl_exec);
CLExecutor *ExtractClExecutor(StreamExecutor *stream_exec);

ScopedActivateExecutorContext::ScopedActivateExecutorContext(
    CLExecutor *cl_exec):
      driver_scoped_activate_context_(
          new ScopedActivateContext{ExtractClContext(cl_exec)}) { }

ScopedActivateExecutorContext::ScopedActivateExecutorContext(
    StreamExecutor *stream_exec)
    : ScopedActivateExecutorContext(ExtractClExecutor(stream_exec)) {}

ScopedActivateExecutorContext::~ScopedActivateExecutorContext() {
  delete static_cast<ScopedActivateContext *>(driver_scoped_activate_context_);
}

}  // namespace cl
}  // namespace gputools
}  // namespace perftools
