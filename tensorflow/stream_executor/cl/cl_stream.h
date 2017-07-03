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

// Defines the CLStream type - the CL-specific implementation of the generic
// StreamExecutor Stream interface.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CL_CL_STREAM_H_
#define TENSORFLOW_STREAM_EXECUTOR_CL_CL_STREAM_H_

#include "tensorflow/stream_executor/cl/cl_driver.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

#include <iostream>

namespace perftools {
namespace gputools {
namespace cl {

class CLExecutor;

// Wraps a CUstream in order to satisfy the platform-independent
// StreamInterface.
//
// Thread-safe post-initialization.
class CLStream : public internal::StreamInterface {
 public:
  explicit CLStream(CLExecutor *parent)
      : parent_(parent), cl_stream_(nullptr), completed_event_(nullptr) {}

  // Note: teardown is handled by a parent's call to DeallocateStream.
  ~CLStream() override {}

  void *CudaStreamHack() override {
    std::cout << "cl_stream.h CLStream::CudaStreamHack()" << std::endl;
    return cl_stream_;
  }
  void **CudaStreamMemberHack() override {
    // std::cout << "cl_stream.h CLStream::CudaStreamMemberHack()" << std::endl;
    return reinterpret_cast<void **>(&cl_stream_);
  }

  // Explicitly initialize the CL resources associated with this stream, used
  // by StreamExecutor::AllocateStream().
  bool Init();

  // Explicitly destroy the CL resources associated with this stream, used by
  // StreamExecutor::DeallocateStream().
  void Destroy();

  // Returns true if no work is pending or executing on the stream.
  bool IsIdle() const;

  // Retrieves an event which indicates that all work enqueued into the stream
  // has completed. Ownership of the event is not transferred to the caller, the
  // event is owned by this stream.
  CUevent* completed_event() { return &completed_event_; }

  // Returns the CUstream value for passing to the CL API.
  //
  // Precond: this CLStream has been allocated (otherwise passing a nullptr
  // into the NVIDIA library causes difficult-to-understand faults).
  CUstream cl_stream() const {
    // std::cout << "cl_stream.h CLStream::cl_stream()" << std::endl;
    DCHECK(cl_stream_ != nullptr);
    return const_cast<CUstream>(cl_stream_);
  }

  CLExecutor *parent() const { return parent_; }

 private:
  CLExecutor *parent_;  // Executor that spawned this stream.
  CUstream cl_stream_;  // Wrapped CL stream handle.

  // Event that indicates this stream has completed.
  CUevent completed_event_ = nullptr;
};

// Helper functions to simplify extremely common flows.
// Converts a Stream to the underlying CLStream implementation.
CLStream *AsCLStream(Stream *stream);

// Extracts a CUstream from a CLStream-backed Stream object.
CUstream AsCLStreamValue(Stream *stream);

}  // namespace cl
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CL_CL_STREAM_H_
