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

#include "tensorflow/stream_executor/cl/cl_event.h"

#include "tensorflow/stream_executor/cl/cl_gpu_executor.h"
#include "tensorflow/stream_executor/cl/cl_stream.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace perftools {
namespace gputools {
namespace cl {

CLEvent::CLEvent(CLExecutor* parent)
    : parent_(parent), cl_event_(nullptr) {}

CLEvent::~CLEvent() {}

port::Status CLEvent::Init() {
  return CLDriver::CreateEvent(parent_->cl_context(), &cl_event_,
                                 CLDriver::EventFlags::kDisableTiming);
}

port::Status CLEvent::Destroy() {
  return CLDriver::DestroyEvent(parent_->cl_context(), &cl_event_);
}

port::Status CLEvent::Record(CLStream* stream) {
  return CLDriver::RecordEvent(parent_->cl_context(), cl_event_,
                                 stream->cl_stream());
}

Event::Status CLEvent::PollForStatus() {
  port::StatusOr<CUresult> status =
      CLDriver::QueryEvent(parent_->cl_context(), cl_event_);
  if (!status.ok()) {
    LOG(ERROR) << "Error polling for event status: "
               << status.status().error_message();
    return Event::Status::kError;
  }

  switch (status.ValueOrDie()) {
    case CUDA_SUCCESS:
      return Event::Status::kComplete;
    case CUDA_ERROR_NOT_READY:
      return Event::Status::kPending;
    default:
      LOG(INFO) << "Error condition returned for event status: "
                << status.ValueOrDie();
      return Event::Status::kError;
  }
}

const CUevent& CLEvent::cl_event() {
  return cl_event_;
}

}  // namespace cl
}  // namespace gputools
}  // namespace perftools
