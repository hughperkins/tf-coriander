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

#include "tensorflow/stream_executor/cl/cl_stream.h"

#include "tensorflow/stream_executor/cl/cl_gpu_executor.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/stream.h"

namespace perftools {
namespace gputools {
namespace cl {

bool CLStream::Init() {
  if (!CLDriver::CreateStream(parent_->cl_context(), &cl_stream_)) {
    return false;
  }
  return CLDriver::CreateEvent(parent_->cl_context(), &completed_event_,
                                 CLDriver::EventFlags::kDisableTiming)
      .ok();
}

void CLStream::Destroy() {
  if (completed_event_ != nullptr) {
    port::Status status =
        CLDriver::DestroyEvent(parent_->cl_context(), &completed_event_);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
    }
  }

  CLDriver::DestroyStream(parent_->cl_context(), &cl_stream_);
}

bool CLStream::IsIdle() const {
  return CLDriver::IsStreamIdle(parent_->cl_context(), cl_stream_);
}

CLStream *AsCLStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return static_cast<CLStream *>(stream->implementation());
}

CUstream AsCLStreamValue(Stream *stream) {
  DCHECK(stream != nullptr);
  return AsCLStream(stream)->cl_stream();
}

}  // namespace cl
}  // namespace gputools
}  // namespace perftools
