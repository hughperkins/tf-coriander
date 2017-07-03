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

#ifndef TENSORFLOW_STREAM_EXECUTOR_CL_CL_EVENT_H_
#define TENSORFLOW_STREAM_EXECUTOR_CL_CL_EVENT_H_

#include "cuda.h"

#include "tensorflow/stream_executor/cl/cl_driver.h"
#include "tensorflow/stream_executor/cl/cl_stream.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/status.h"

namespace perftools {
namespace gputools {
namespace cl {

// CLEvent wraps a CUevent in the platform-independent EventInterface
// interface.
class CLEvent : public internal::EventInterface {
 public:
  explicit CLEvent(CLExecutor* parent);

  ~CLEvent() override;

  // Populates the CL-platform-specific elements of this object.
  port::Status Init();

  // Deallocates any platform-specific elements of this object. This is broken
  // out (not part of the destructor) to allow for error reporting.
  port::Status Destroy();

  // Inserts the event at the current position into the specified stream.
  port::Status Record(CLStream* stream);

  // Polls the CL platform for the event's current status.
  Event::Status PollForStatus();

  // The underyling CL event element.
  // CUevent is from cuda includes
  const CUevent& cl_event();

 private:
  // The Executor used to which this object and CUevent are bound.
  CLExecutor* parent_;

  // The underlying CL event element.
  CUevent cl_event_;
};

}  // namespace cl
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CL_CL_EVENT_H_
