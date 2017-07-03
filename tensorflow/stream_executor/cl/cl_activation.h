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

// This file contains APIs that assume a StreamExecutor is backed by CL.
// It reaches into the CL implementation to activate an underlying CL
// context.
//
// Having this file separate from cl_gpu_executor.h means that dependent
// code does not also have to depend on cl.h.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CL_CL_ACTIVATION_H_
#define TENSORFLOW_STREAM_EXECUTOR_CL_CL_ACTIVATION_H_

#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

class StreamExecutor;

namespace cl {

class CLExecutor;
class ScopedActivateContext;

// Activates a CL context within an enclosing scope.
class ScopedActivateExecutorContext {
 public:
  // Form that takes a CL executor implementation.
  explicit ScopedActivateExecutorContext(CLExecutor* cl_exec);

  // Form that takes a pImpl executor and extracts a CL implementation --
  // fatal failure if it is not CL inside.
  explicit ScopedActivateExecutorContext(StreamExecutor* stream_exec);

  ~ScopedActivateExecutorContext();

 private:

  // The cl.h-using datatype that we wrap.
  ScopedActivateContext* driver_scoped_activate_context_;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedActivateExecutorContext);
};

}  // namespace cl
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CL_CL_ACTIVATION_H_
